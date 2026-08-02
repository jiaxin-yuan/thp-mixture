#!/usr/bin/env python
"""Uncertainty-aware remaining-time prediction from THP-Mixture via MC rollout.

For every test prefix (case length L, prefix p in [1, L-1]) we draw S
ancestral samples from the learned one-step law p(dt, k | H): at each step,
sample dt from the zero-inflated LogNormal mixture and the next activity from
the mark softmax, feed both back, and stop at the ORACLE remaining length
K = L - p (THP has no EOS token; consistent with run_suffix_eval.py). The S
suffix sums give an empirical p(RT | prefix), summarized as
(pred_mean, pred_std) and scored with the BPM2025 / UQ4PPM metric quartet
(MAE / MA / MPIW / AURG; see uq_metrics.py). A distribution-free MA from the
empirical sample quantiles is reported alongside the Gaussian benchmark MA.

This MC estimator is the principled replacement for greedy mean-path rollout
(Jensen bias; docs/thp_suffix_feasibility.tex, Obstacle 3).

Usage (mpp env):
    conda run -n mpp python run_rt_uncertainty.py --dataset UQ_BPIC13I
    conda run -n mpp python run_rt_uncertainty.py --all   [--samples 50]
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)          # THP_for_PPM/
THP_DIR = REPO_ROOT                              # code, data/ and saved_models/
PER_DATASET = os.path.join(REPO_ROOT, "paper_results", "per_dataset")

sys.path.insert(0, SCRIPT_DIR)
import uq_metrics as um

FOLD = 0
N_MIX = 5
SEED = 0
MAX_ROWS = 32768         # instance*sample rows per AR chunk (seqs are short)
MAX_SCORE_ELEMS = 2 ** 26  # cap rows*(p+maxK)^2: attention scores are
                           # [rows, n_head, L, L]; long-case logs (Sepsis,
                           # BPIC12) OOM under the flat MAX_ROWS cap
LOG_DT_CLAMP = 20.0      # clamp on mu + sigma*z before exp (e^20 days, finite)

UQ_DATASETS = [
    "UQ_HelpDesk", "UQ_Sepsis", "UQ_BPIC13I", "UQ_BPIC20DD", "UQ_BPIC20RFP",
    "UQ_BPIC20PTC", "UQ_BPIC20ID", "UQ_BPIC20TPD", "UQ_BPIC12", "UQ_BPIC15_1",
]


def out_dir(ds):
    d = os.path.join(PER_DATASET, ds)
    os.makedirs(d, exist_ok=True)
    return d


def _load_thp(ds, device):
    sys.path.insert(0, THP_DIR)
    cwd = os.getcwd()
    os.chdir(THP_DIR)
    from preprocess.dataset import df_to_dict, EventData
    from transformer.model import get_non_pad_mask, Transformer

    fold_file = f"fold{FOLD}_variation0_{ds}"
    train_out, _, test_out = df_to_dict(
        directory=os.path.join(THP_DIR, "data"), fold_filename=fold_file,
    )
    num_types = train_out["dim_process"]
    ck = os.path.join(THP_DIR, "saved_models",
                      f"{fold_file}_mixture_best_model_best_mae.pth")
    if not os.path.exists(ck):
        os.chdir(cwd)
        return None
    model = Transformer(num_types=num_types, d_model=36, d_rnn=256, d_inner=128,
                        n_layers=4, n_head=4, d_k=16, d_v=16, dropout=0.1,
                        n_mix=N_MIX, model_type="mixture")
    model.load_state_dict(torch.load(ck, map_location=device)["model_state_dict"])
    model.to(device).eval()
    ed = EventData(test_out["test"])
    os.chdir(cwd)
    return model, ed, num_types


def _sample_step(tie_logit, time_params, mark_logit, gen):
    """One ancestral sample of (dt, mark) per row from the last position."""
    pi = torch.sigmoid(tie_logit[:, -1])                       # [R]
    params = time_params[:, -1]                                # [R, 3K]
    w = F.softmax(params[:, :N_MIX], dim=-1)
    mu = params[:, N_MIX:2 * N_MIX]
    sigma = F.softplus(params[:, 2 * N_MIX:]) + 1e-4
    comp = torch.multinomial(w, 1, generator=gen).squeeze(-1)  # [R]
    r = torch.arange(comp.size(0), device=comp.device)
    z = torch.randn(comp.size(0), device=comp.device, generator=gen)
    log_dt = (mu[r, comp] + sigma[r, comp] * z).clamp(max=LOG_DT_CLAMP)
    dt = torch.exp(log_dt)
    tie = torch.rand(pi.size(0), device=pi.device, generator=gen) < pi
    dt = torch.where(tie, torch.zeros_like(dt), dt)
    probs = F.softmax(mark_logit[:, -1, :], dim=-1)
    mark0 = torch.multinomial(probs, 1, generator=gen).squeeze(-1)   # 0-indexed
    return dt, mark0


def run_dataset(ds, S):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = _load_thp(ds, device)
    if loaded is None:
        print(f"[{ds}] SKIP: mixture checkpoint not found")
        return
    model, ed, _ = loaded
    from transformer.model import get_non_pad_mask  # noqa: F401 (path set)

    gen = torch.Generator(device=device); gen.manual_seed(SEED)

    # Prefix instances grouped by prefix length (cf. run_suffix_eval.py)
    groups = defaultdict(list)   # p -> list of (types[p], times[p], K, rt_true)
    for ci in range(len(ed.time)):
        t = np.asarray(ed.time[ci], dtype=np.float64)
        a = np.asarray(ed.activity[ci], dtype=np.int64)
        L = len(a)
        for p in range(1, L):
            groups[p].append((a[:p], t[:p], L - p, float(t[-1] - t[p - 1])))

    n_inst = sum(len(v) for v in groups.values())
    print(f"[{ds}] {n_inst} prefixes, S={S}", flush=True)
    import time as _time
    t0, done = _time.time(), 0

    means, stds, gts, plens, medians = [], [], [], [], []
    q_lo, q_hi = [], []                      # 5% / 95% sample quantiles
    sample_rows = []                          # for distribution-free MA
    with torch.no_grad():
        for p, items in sorted(groups.items()):
            # Longest suffixes first so each chunk is sized for its own
            # worst-case final length, not the group's.
            items = sorted(items, key=lambda x: -x[2])
            s0 = 0
            while s0 < len(items):
                lfin = p + items[s0][2]
                rows = min(MAX_ROWS, max(S, MAX_SCORE_ELEMS // (lfin * lfin)))
                chunk_inst = max(1, rows // S)
                sub = items[s0:s0 + chunk_inst]
                s0 += chunk_inst
                G = len(sub)
                Ks = np.array([x[2] for x in sub], dtype=np.int64)
                maxK = int(Ks.max())
                seq_type = torch.tensor(np.stack([x[0] for x in sub]),
                                        dtype=torch.long, device=device)
                seq_time = torch.tensor(np.stack([x[1] for x in sub]),
                                        dtype=torch.float32, device=device)
                # expand to S sample chains: [G*S, p]
                seq_type = seq_type.repeat_interleave(S, dim=0)
                seq_time = seq_time.repeat_interleave(S, dim=0)
                rt_acc = torch.zeros(G * S, device=device)
                ks_t = torch.tensor(Ks, device=device).repeat_interleave(S)
                for step in range(maxK):
                    _, (tie, tparams, mark) = model(seq_type, seq_time)
                    dt, mark0 = _sample_step(tie, tparams, mark, gen)
                    active = (step < ks_t)
                    rt_acc = rt_acc + dt * active.float()
                    seq_type = torch.cat([seq_type, (mark0 + 1).unsqueeze(1)], dim=1)
                    seq_time = torch.cat([seq_time, (seq_time[:, -1] + dt).unsqueeze(1)], dim=1)
                rt = rt_acc.view(G, S)
                means.extend(rt.mean(1).cpu().numpy())
                stds.extend(rt.std(1, unbiased=True).cpu().numpy())
                medians.extend(rt.median(1).values.cpu().numpy())
                q = torch.quantile(rt, torch.tensor([0.05, 0.95], device=device), dim=1)
                q_lo.extend(q[0].cpu().numpy()); q_hi.extend(q[1].cpu().numpy())
                sample_rows.append(rt.cpu().numpy())
                gts.extend(x[3] for x in sub)
                plens.extend([p] * G)
                done += G
                if done % 20000 < G:
                    print(f"  ...{done}/{n_inst} prefixes "
                          f"({_time.time() - t0:.0f}s)", flush=True)

    pred_mean = np.asarray(means); pred_std = np.asarray(stds)
    y = np.asarray(gts); rt_samples = np.concatenate(sample_rows, axis=0)

    res = um.quartet(pred_mean, pred_std, y)
    res["mae_median_point"] = um.mae(np.asarray(medians), y)
    ma_s, picp = um.ma_sample(rt_samples, y)
    res["ma_sample"] = ma_s
    res["picp_curve"] = picp
    res["mpiw_90_sample_days"] = float(np.mean(np.asarray(q_hi) - np.asarray(q_lo)))
    res["n_samples_per_prefix"] = S
    res["stop_rule"] = "oracle remaining length (no EOS token)"

    out = {"config": {"fold": FOLD, "seed": SEED, "S": S, "dataset": ds},
           "models": {"THP-Mixture (MC rollout)": res}}
    with open(os.path.join(out_dir(ds), "rt_uncertainty_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(os.path.join(out_dir(ds), "rt_uncertainty_arrays.npz"),
                        pred_mean=pred_mean, pred_std=pred_std, y=y,
                        pref_len=np.asarray(plens),
                        q05=np.asarray(q_lo), q95=np.asarray(q_hi))
    print(f"  MAE={res['mae']:.3f}d (median-point {res['mae_median_point']:.3f}) "
          f"MA={res['ma']:.3f} (sample {res['ma_sample']:.3f}) "
          f"MPIW(rms σ)={res['mpiw_rms_sigma']:.2f}d AURG={res['aurg']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--samples", type=int, default=50)
    args = ap.parse_args()
    targets = UQ_DATASETS if args.all else ([args.dataset] if args.dataset else [])
    if not targets:
        ap.error("provide --dataset or --all")
    for ds in targets:
        run_dataset(ds, args.samples)
