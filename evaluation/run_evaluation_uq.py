#!/usr/bin/env python3
"""Generalized Joint Event Accuracy (JEA) evaluation across the 10 UQ4PPM logs.

Methods:
  THP Mixture (ours)  — argmax(mark_head) for activity, E[Δt] from the
                        zero-inflated LogNormal mixture for time. Predicts both.
  THP Baseline (MLE)  — intensity/MAE model, time only (no activity head).
  Frequency / Last-Event baselines — activity+time reference points.
  SuTraN / ED-LSTM    — decoupled PPM, step-0 of the suffix prediction.

PGTNet is dropped. The remaining-time SOTA reference is the BPM2025 UQ4PPM
paper (Amiri Elyasi, van der Aa, Stuckenschmidt — "A Simple and Calibrated
Approach for Uncertainty-Aware Remaining Time Prediction"), Table 3 MAE (days).

All methods use the same chronological 80/20 split + prefixes [2, n-1].
JEA(ε) = fraction of next-event predictions where activity is correct AND the
time error < ε days. ε grid is shared across methods.

Usage (THP needs the `mpp` env; PPM needs `fasutran`; report needs either):
  conda run -n mpp      python run_evaluation_uq.py --thp    --dataset UQ_HelpDesk
  conda run -n fasutran python run_evaluation_uq.py --ppm    --dataset UQ_HelpDesk
  python run_evaluation_uq.py --report --plots --dataset UQ_HelpDesk
  # or operate on every dataset:
  ... --thp --all   /   --ppm --all   /   --report --plots --all
"""

import os
import sys
import json
import argparse
import pickle
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)          # THP_for_PPM/
THP_DIR = REPO_ROOT                              # code, data/ and saved_models/
PER_DATASET = os.path.join(REPO_ROOT, "paper_results", "per_dataset")
# SuTraN / ED-LSTM are not vendored; point this at a checkout to run --ppm.
SUTRAN_DIR = os.environ.get("SUTRAN_DIR", os.path.join(REPO_ROOT, "..", "SuffixTransformerNetwork"))

FOLD = 0
EPS_GRID = np.geomspace(1e-6, 5.0, 25)  # days

UQ_DATASETS = [
    "UQ_HelpDesk", "UQ_Sepsis", "UQ_BPIC13I", "UQ_BPIC20DD",
    "UQ_BPIC20RFP", "UQ_BPIC20PTC", "UQ_BPIC20ID", "UQ_BPIC20TPD",
    "UQ_BPIC12", "UQ_BPIC15_1",
]

# BPM2025 UQ4PPM paper, Table 3 — remaining-time MAE in days.
# Columns: DA+H, CDA+H, HR, DE, BE+H, CARD, LA+I, LA+S.
PAPER_RT_TABLE = {
    "UQ_BPIC20DD":  {"DA+H": 5.38, "CDA+H": 6.32, "HR": 4.13, "DE": 5.92, "BE+H": 4.70, "CARD": 4.50, "LA+I": 4.12, "LA+S": 4.12},
    "UQ_BPIC20ID":  {"DA+H": 14.62, "CDA+H": 17.09, "HR": 15.40, "DE": 17.26, "BE+H": 14.08, "CARD": 22.26, "LA+I": 21.72, "LA+S": 21.72},
    "UQ_BPIC20PTC": {"DA+H": 9.45, "CDA+H": 12.83, "HR": 8.36, "DE": 11.08, "BE+H": 10.45, "CARD": 11.86, "LA+I": 7.73, "LA+S": 7.73},
    "UQ_BPIC20RFP": {"DA+H": 5.09, "CDA+H": 6.30, "HR": 5.59, "DE": 7.35, "BE+H": 6.62, "CARD": 5.71, "LA+I": 5.19, "LA+S": 5.19},
    "UQ_BPIC20TPD": {"DA+H": 25.02, "CDA+H": 25.98, "HR": 26.01, "DE": 27.12, "BE+H": 27.80, "CARD": 30.37, "LA+I": 27.79, "LA+S": 27.79},
    "UQ_BPIC15_1":  {"DA+H": 30.29, "CDA+H": 32.23, "HR": 26.93, "DE": 24.71, "BE+H": 27.23, "CARD": 100.43, "LA+I": 25.28, "LA+S": 25.28},
    "UQ_BPIC13I":   {"DA+H": 3.67, "CDA+H": 4.19, "HR": 3.15, "DE": 3.36, "BE+H": 3.06, "CARD": 4.69, "LA+I": 5.53, "LA+S": 5.53},
    "UQ_BPIC12":    {"DA+H": 6.60, "CDA+H": 8.02, "HR": 5.86, "DE": 6.05, "BE+H": 5.95, "CARD": 7.03, "LA+I": 7.84, "LA+S": 7.84},
    "UQ_Sepsis":    {"DA+H": 16.35, "CDA+H": 18.08, "HR": 15.32, "DE": 15.44, "BE+H": 15.69, "CARD": 24.50, "LA+I": 15.73, "LA+S": 15.73},
    "UQ_HelpDesk":  {"DA+H": 11.28, "CDA+H": 10.09, "HR": 9.45, "DE": 11.18, "BE+H": 18.65, "CARD": 12.87, "LA+I": 9.45, "LA+S": 9.45},
}


def out_dir(ds):
    d = os.path.join(PER_DATASET, ds)
    os.makedirs(d, exist_ok=True)
    return d


def dump_persample(ds, tag, correct, abs_err, prefix_len, true_gap, conf=None):
    """Persist flat per-event next-event records for the (Act%, MAE) Pareto plot.

    One row per scored next-event: `correct` (activity hit, 0/1), `abs_err`
    (next-gap |error| in days, the MAE contribution under each model's reported
    decode), `prefix_len`, `true_gap` (ground-truth next gap in days), and an
    optional `conf` (max next-activity softmax). Non-destructive: writes a new
    persample_<tag>.npz next to the existing *_results.json. Binning is left to
    the plotting script so the bin variable can be chosen without re-running.
    """
    n = len(correct)
    if conf is None:
        conf = np.full(n, np.nan)
    np.savez(
        os.path.join(out_dir(ds), f"persample_{tag}.npz"),
        correct=np.asarray(correct, dtype=np.float32),
        abs_err=np.asarray(abs_err, dtype=np.float32),
        prefix_len=np.asarray(prefix_len, dtype=np.int32),
        true_gap=np.asarray(true_gap, dtype=np.float32),
        conf=np.asarray(conf, dtype=np.float32),
    )
    print(f"    dumped persample_{tag}.npz (N={n})")


# =========================================================================
# Shared JEA bookkeeping
# =========================================================================
def _finalize_from_arrays(correct, dt_err, ranks, prefix_lens, num_types):
    """Build the THP-method result dict from flat per-event arrays."""
    correct = np.asarray(correct, dtype=bool)
    dt_err = np.asarray(dt_err, dtype=float)
    per_event = [float((correct & (dt_err < eps)).mean()) for eps in EPS_GRID]
    eps_mid = EPS_GRID[len(EPS_GRID) // 2]
    pl = np.asarray(prefix_lens, dtype=int)
    per_prefix = {}
    for k in sorted(set(pl.tolist())):
        m = pl == k
        per_prefix[str(int(k))] = float((correct[m] & (dt_err[m] < eps_mid)).mean())
    out = {
        "per_event_teacher": per_event,
        "activity_accuracy_teacher": float(correct.mean()) * 100,
        "time_mae_teacher": float(dt_err.mean()),
        "per_prefix_teacher": per_prefix,
    }
    if ranks is not None and num_types:
        ranks = np.asarray(ranks, dtype=int)
        topk = {}
        for ei, eps in enumerate(EPS_GRID):
            time_ok = dt_err < eps
            topk[str(ei)] = [float(((ranks <= k) & time_ok).mean() * 100)
                             for k in range(1, num_types + 1)]
        out["topk_jea"] = topk
        out["num_types"] = int(num_types)
    return out


def _finalize_from_arrays_matched(correct, mae_err, time_ok, ranks, prefix_lens, num_types):
    """Like _finalize_from_arrays, but with decoupled decodes per metric:
    `mae_err` from the median decode and `time_ok` [E, N] from the per-eps
    eps-mode decode (the time hit depends on the tolerance it was decoded for).
    """
    correct = np.asarray(correct, dtype=bool)
    mae_err = np.asarray(mae_err, dtype=float)
    per_event = [float((correct & time_ok[ei]).mean()) for ei in range(len(EPS_GRID))]
    mid = len(EPS_GRID) // 2
    pl = np.asarray(prefix_lens, dtype=int)
    per_prefix = {}
    for k in sorted(set(pl.tolist())):
        m = pl == k
        per_prefix[str(int(k))] = float((correct[m] & time_ok[mid][m]).mean())
    out = {
        "per_event_teacher": per_event,
        "activity_accuracy_teacher": float(correct.mean()) * 100,
        "time_mae_teacher": float(mae_err.mean()),
        "per_prefix_teacher": per_prefix,
    }
    if ranks is not None and num_types:
        ranks = np.asarray(ranks, dtype=int)
        topk = {}
        for ei in range(len(EPS_GRID)):
            topk[str(ei)] = [float(((ranks <= k) & time_ok[ei]).mean() * 100)
                             for k in range(1, num_types + 1)]
        out["topk_jea"] = topk
        out["num_types"] = int(num_types)
    return out


# =========================================================================
# 1. THP evaluation (mpp env). Mixture = activity+time; baseline = time only.
#    decode="matched" (default): metric-matched point estimates from the
#      mixture density — median for MAE, eps-ball mode (re-decoded per eps)
#      for JEA. See generative_eval.py.
#    decode="mean": legacy E[dt] decode (ablation; reproduces old numbers).
# =========================================================================
def run_thp(ds, decode="matched"):
    import torch
    sys.path.insert(0, THP_DIR)
    os.chdir(THP_DIR)  # saved_models / data are relative to THP_for_PPM
    from preprocess.dataset import df_to_dict, get_dataloader
    from transformer.model import get_non_pad_mask, Transformer
    import Utils
    sys.path.insert(0, SCRIPT_DIR)
    import generative_eval as ge

    try:
        from main import build_model
    except ImportError:
        # main.py pulls in trainer/train.py (needs sklearn, absent from mpp);
        # replicate main.build_model's fixed hyper-parameters.
        def build_model(num_types, device, n_mix=5, model_type="mixture"):
            model = Transformer(
                num_types=num_types, d_model=36, d_rnn=256, d_inner=128,
                n_layers=4, n_head=4, d_k=16, d_v=16, dropout=0.1,
                n_mix=n_mix, model_type=model_type)
            return model.to(device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_file = f"fold{FOLD}_variation0_{ds}"
    data_dir = os.path.join(THP_DIR, "data")
    train_out, val_out, test_out = df_to_dict(
        directory=data_dir, fold_filename=fold_file)
    num_types = train_out["dim_process"]
    test_loader = get_dataloader(test_out["test"], batch_size=64, shuffle=False)
    print(f"  [{ds}] test seqs={len(test_out['test'])}, marks={num_types}")

    methods = {}
    n_mix = 5

    # ---- THP Mixture (ours): activity + time ----
    mix_ckpt = os.path.join(THP_DIR, "saved_models",
                            f"{fold_file}_mixture_best_model_best_mae.pth")
    if os.path.exists(mix_ckpt):
        model = build_model(num_types, device, n_mix=n_mix, model_type="mixture")
        model.load_state_dict(torch.load(mix_ckpt, map_location=device)["model_state_dict"])
        model.eval()
        cor, derr, rk, pls = [], [], [], []
        tg, cf = [], []
        gdts, pis, log_ws, mus, sigmas = [], [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                event_time, _, _, event_type = (x.to(device) for x in batch)
                non_pad = get_non_pad_mask(event_type).squeeze(-1)  # [B,S]
                _, (tie_logit, time_params, mark_logit) = model(event_type, event_time)
                pred_dt = Utils.expected_dt(tie_logit, time_params, non_pad, n_mix)  # [B,S]
                # position t predicts t+1
                pred_mark = mark_logit[:, :-1, :].argmax(-1)        # [B,S-1] 0-idx
                gt_mark = event_type[:, 1:] - 1                     # [B,S-1] 0-idx, PAD->-1
                pdt = pred_dt[:, :-1]
                gdt = event_time[:, 1:] - event_time[:, :-1]
                valid = (non_pad[:, 1:] > 0) & (event_type[:, 1:] > 0)
                # rank of gt mark in logits (1-indexed)
                logit_t = mark_logit[:, :-1, :]                    # [B,S-1,C]
                gt_safe = gt_mark.clamp(min=0).unsqueeze(-1)
                gt_logit = torch.gather(logit_t, 2, gt_safe).squeeze(-1)
                rank = (logit_t > gt_logit.unsqueeze(-1)).sum(-1) + 1  # [B,S-1]

                vm = valid.reshape(-1)
                cor.extend(((pred_mark == gt_mark).reshape(-1)[vm]).cpu().numpy().tolist())
                derr.extend((torch.abs(pdt - gdt).reshape(-1)[vm]).cpu().numpy().tolist())
                rk.extend((rank.reshape(-1)[vm]).cpu().numpy().tolist())
                idx = torch.arange(1, event_type.size(1), device=device).unsqueeze(0).expand(event_type.size(0), -1)
                pls.extend((idx.reshape(-1)[vm]).cpu().numpy().tolist())
                tg.extend((gdt.reshape(-1)[vm]).cpu().numpy().tolist())
                conf_b = torch.softmax(logit_t, -1).max(-1).values    # [B,S-1]
                cf.extend((conf_b.reshape(-1)[vm]).cpu().numpy().tolist())
                if decode == "matched":
                    pi_b, lw_b, mu_b, sg_b = ge.extract_params(
                        tie_logit[:, :-1], time_params[:, :-1], n_mix)
                    gdts.append(gdt.reshape(-1)[vm])
                    pis.append(pi_b.reshape(-1)[vm])
                    log_ws.append(lw_b.reshape(-1, n_mix)[vm])
                    mus.append(mu_b.reshape(-1, n_mix)[vm])
                    sigmas.append(sg_b.reshape(-1, n_mix)[vm])

        if decode == "matched":
            with torch.no_grad():
                dt = torch.cat(gdts); pi = torch.cat(pis); log_w = torch.cat(log_ws)
                mu = torch.cat(mus); sigma = torch.cat(sigmas)
                # MAE: median decode (MAE-optimal point of the density)
                derr = (ge.quantile(0.5, pi, log_w, mu, sigma) - dt).abs().cpu().numpy()
                # JEA: eps-ball mode, re-decoded for each tolerance
                time_ok = np.stack([
                    ((ge.eps_mode(float(e), pi, log_w, mu, sigma) - dt).abs() < e)
                    .cpu().numpy() for e in EPS_GRID])              # [E, N]
            methods["THP Mixture"] = _finalize_from_arrays_matched(
                cor, derr, time_ok, rk, pls, num_types)
        else:
            methods["THP Mixture"] = _finalize_from_arrays(cor, derr, rk, pls, num_types)
        methods["THP Mixture"]["decode"] = decode
        print(f"    THP Mixture: acc={methods['THP Mixture']['activity_accuracy_teacher']:.1f}% "
              f"MAE={methods['THP Mixture']['time_mae_teacher']:.3f}d")
        dump_persample(ds, "thp_mixture", cor, np.asarray(derr), pls, tg, cf)
    else:
        print(f"    SKIP THP Mixture: {mix_ckpt} not found")

    # ---- THP Baseline: intensity MLE + regression heads. Prefer the
    # baseline_act checkpoint (categorical next-activity head added), which
    # makes Act/JEA defined; fall back to the legacy time-only baseline. ----
    bact_ckpt = os.path.join(THP_DIR, "saved_models",
                             f"{fold_file}_baseline_act_best_model_best_mae.pth")
    bl_ckpt = os.path.join(THP_DIR, "saved_models",
                           f"{fold_file}_baseline_best_model_best_mae.pth")
    if not os.path.exists(bl_ckpt):
        # legacy naming (no "_baseline" infix) — only if truly baseline-shaped
        legacy = os.path.join(THP_DIR, "saved_models",
                              f"{fold_file}_best_model_best_mae.pth")
        if os.path.exists(legacy):
            sd = torch.load(legacy, map_location="cpu")["model_state_dict"]
            if "time_predictor.linear.weight" in sd:
                bl_ckpt = legacy
    has_act = os.path.exists(bact_ckpt)
    use_ckpt = bact_ckpt if has_act else bl_ckpt
    if os.path.exists(use_ckpt):
        model = build_model(num_types, device, n_mix=n_mix,
                            model_type="baseline_act" if has_act else "baseline")
        bl_state = torch.load(use_ckpt, map_location=device)
        model.load_state_dict(bl_state["model_state_dict"])
        model.eval()
        # De-standardize Δt predictions if the checkpoint was trained on
        # z-scored time targets (legacy raw-days ckpts have time_stats=None).
        ts = bl_state.get("time_stats")
        dt_mean = ts["dt_mean"] if ts else 0.0
        dt_std = ts["dt_std"] if ts else 1.0
        cor, derr, rk, pls = [], [], [], []
        tg, cf = [], []
        with torch.no_grad():
            for batch in test_loader:
                event_time, _, _, event_type = (x.to(device) for x in batch)
                non_pad = get_non_pad_mask(event_type).squeeze(-1)
                _, heads = model(event_type, event_time)
                time_pred = heads[0].squeeze(-1) * dt_std + dt_mean
                pdt = time_pred[:, :-1]
                gdt = event_time[:, 1:] - event_time[:, :-1]
                valid = (non_pad[:, 1:] > 0) & (event_type[:, 1:] > 0)
                vm = valid.reshape(-1)
                derr.extend((torch.abs(pdt - gdt).reshape(-1)[vm]).cpu().numpy().tolist())
                idx = torch.arange(1, event_type.size(1), device=device).unsqueeze(0).expand(event_type.size(0), -1)
                pls.extend((idx.reshape(-1)[vm]).cpu().numpy().tolist())
                tg.extend((gdt.reshape(-1)[vm]).cpu().numpy().tolist())
                if has_act:
                    mark_logit = heads[2]
                    pred_mark = mark_logit[:, :-1, :].argmax(-1)
                    gt_mark = event_type[:, 1:] - 1
                    logit_t = mark_logit[:, :-1, :]
                    gt_safe = gt_mark.clamp(min=0).unsqueeze(-1)
                    gt_logit = torch.gather(logit_t, 2, gt_safe).squeeze(-1)
                    rank = (logit_t > gt_logit.unsqueeze(-1)).sum(-1) + 1
                    cor.extend(((pred_mark == gt_mark).reshape(-1)[vm]).cpu().numpy().tolist())
                    rk.extend((rank.reshape(-1)[vm]).cpu().numpy().tolist())
                    conf_b = torch.softmax(logit_t, -1).max(-1).values
                    cf.extend((conf_b.reshape(-1)[vm]).cpu().numpy().tolist())
        if has_act:
            methods["THP Baseline"] = _finalize_from_arrays(cor, derr, rk, pls, num_types)
            methods["THP Baseline"]["arch"] = "baseline_act"
            print(f"    THP Baseline (+act): acc={methods['THP Baseline']['activity_accuracy_teacher']:.1f}% "
                  f"MAE={methods['THP Baseline']['time_mae_teacher']:.3f}d")
            dump_persample(ds, "thp_baseline", cor, np.asarray(derr), pls, tg, cf)
        else:
            derr = np.asarray(derr)
            pl = np.asarray(pls, dtype=int)
            methods["THP Baseline"] = {
                "per_event_teacher": [0.0 for _ in EPS_GRID],
                "activity_accuracy_teacher": 0.0,
                "time_mae_teacher": float(derr.mean()),
                "per_prefix_teacher": {str(int(k)): 0.0 for k in sorted(set(pl.tolist()))},
                "no_activity": True,
            }
            print(f"    THP Baseline: MAE={methods['THP Baseline']['time_mae_teacher']:.3f}d (time only)")
    else:
        print(f"    SKIP THP Baseline: {use_ckpt} not found")

    # ---- Frequency / Last-Event baselines (activity+time reference) ----
    from collections import Counter
    train_loader = get_dataloader(train_out["train"], batch_size=64, shuffle=False)
    next_marks, dts, per_mark = [], [], {}
    for batch in train_loader:
        times, _, _, types = batch
        for b in range(types.size(0)):
            mask = types[b] > 0
            t = times[b][mask]; m = types[b][mask]
            for i in range(1, len(m)):
                next_marks.append(int(m[i])); d = float(t[i] - t[i-1]); dts.append(d)
                per_mark.setdefault(int(m[i-1]), []).append(d)
    if next_marks:
        most_common = Counter(next_marks).most_common(1)[0][0]
        gmean = float(np.mean(dts))
        pmed = {k: float(np.median(v)) for k, v in per_mark.items()}
        fc, fd, lc, ld, pl = [], [], [], [], []
        for batch in test_loader:
            times, _, _, types = batch
            for b in range(types.size(0)):
                mask = types[b] > 0
                t = times[b][mask]; m = types[b][mask]
                for i in range(1, len(m)):
                    gt_m = int(m[i]); gt_d = float(t[i] - t[i-1])
                    fc.append(int(most_common == gt_m)); fd.append(abs(gmean - gt_d))
                    lm = int(m[i-1]); lc.append(int(lm == gt_m))
                    ld.append(abs(pmed.get(lm, gmean) - gt_d)); pl.append(i)
        methods["Frequency BL"] = _finalize_from_arrays(fc, fd, None, pl, None)
        methods["Last-Event BL"] = _finalize_from_arrays(lc, ld, None, pl, None)

    output = {"config": {"fold": FOLD, "dataset": ds, "eps_grid": EPS_GRID.tolist()},
              "methods": methods}
    with open(os.path.join(out_dir(ds), "thp_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"    Saved thp_results.json")


# =========================================================================
# 2. SuTraN / ED-LSTM evaluation (fasutran env)
# =========================================================================
def run_ppm(ds):
    import torch
    data_dir = os.path.join(SUTRAN_DIR, ds)
    means = pickle.load(open(os.path.join(data_dir, f"{ds}_train_means_dict.pkl"), "rb"))
    stds = pickle.load(open(os.path.join(data_dir, f"{ds}_train_std_dict.pkl"), "rb"))
    ttne_mean, ttne_std = means["timeLabel_df"][0], stds["timeLabel_df"][0]

    results = {}
    for name, sub in [("SuTraN", "SUTRAN_NDA_results"), ("ED-LSTM", "ED_LSTM_results")]:
        rdir = os.path.join(data_dir, sub, "TEST_SET_RESULTS")
        if not os.path.exists(rdir):
            print(f"  SKIP {name}: {rdir} not found")
            continue
        acts = torch.load(os.path.join(rdir, "suffix_acts_decoded.pt"), map_location="cpu", weights_only=False)
        ttne = torch.load(os.path.join(rdir, "suffix_ttne_preds.pt"), map_location="cpu", weights_only=False)
        pref_len = torch.load(os.path.join(rdir, "pref_len.pt"), map_location="cpu", weights_only=False)
        labels = torch.load(os.path.join(rdir, "labels.pt"), map_location="cpu", weights_only=False)

        gt_acts = labels[2][:, 0].numpy()
        gt_ttne_std = labels[0][:, 0, 0].numpy()
        pred_acts = acts[:, 0].numpy()
        pred_ttne_std = ttne[:, 0].numpy()
        valid = gt_acts > 0
        gt_a, pr_a = gt_acts[valid], pred_acts[valid]
        pr_t = (pred_ttne_std[valid] * ttne_std + ttne_mean) / 86400.0
        gt_t = (gt_ttne_std[valid] * ttne_std + ttne_mean) / 86400.0
        pref_v = pref_len[valid].numpy()
        act_ok = pr_a == gt_a
        t_err = np.abs(pr_t - gt_t)

        jea_curve = [float((act_ok & (t_err < eps)).mean() * 100) for eps in EPS_GRID]
        eps_mid = EPS_GRID[len(EPS_GRID) // 2]
        per_prefix = {int(k): float((act_ok[pref_v == k] & (t_err[pref_v == k] < eps_mid)).mean() * 100)
                      for k in sorted(set(pref_v))}

        # Remaining-time MAE (days) — for comparison vs paper RT-SOTA
        rrt_mae_days = None
        rrt_p = os.path.join(rdir, "MAE_rrt_minutes.pt")
        if os.path.exists(rrt_p):
            rrt = torch.load(rrt_p, map_location="cpu", weights_only=False)
            rrt_mae_days = float(rrt.float().mean()) / 1440.0

        topk_jea, num_act, conf = None, None, None
        lp = os.path.join(rdir, "suffix_act_logits.pt")
        if os.path.exists(lp):
            logits = torch.load(lp, map_location="cpu", weights_only=False)[:, 0, :].numpy()[valid]
            num_act = logits.shape[1]
            ranks = (np.argsort(-logits, axis=1) == gt_a[:, None]).argmax(axis=1) + 1
            ex = np.exp(logits - logits.max(axis=1, keepdims=True))
            conf = (ex / ex.sum(axis=1, keepdims=True)).max(axis=1)
            topk_jea = {}
            for ei, eps in enumerate(EPS_GRID):
                time_ok = t_err < eps
                topk_jea[str(ei)] = [float(((ranks <= k) & time_ok).mean() * 100)
                                     for k in range(1, num_act + 1)]
        dump_persample(ds, "sutran" if name == "SuTraN" else "edlstm",
                       act_ok.astype(np.float32), t_err, pref_v, gt_t, conf)

        results[name] = {
            "n_instances": int(valid.sum()),
            "act_accuracy": float(act_ok.mean() * 100),
            "time_mae_days": float(t_err.mean()),
            "jea_curve": jea_curve,
            "eps_grid": EPS_GRID.tolist(),
            "jea_at_eps_mid": float(jea_curve[len(EPS_GRID) // 2]),
            "rrt_mae_days": rrt_mae_days,
            "per_prefix_jea": per_prefix,
        }
        if topk_jea is not None:
            results[name]["topk_jea"] = topk_jea
            results[name]["num_types"] = num_act
        print(f"  {name}: acc={results[name]['act_accuracy']:.1f}% "
              f"MAE={results[name]['time_mae_days']:.3f}d N={results[name]['n_instances']}")

    with open(os.path.join(out_dir(ds), "ppm_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved ppm_results.json")


# =========================================================================
# 3. Report + plots
# =========================================================================
def _load(ds):
    thp, ppm, eps = {}, {}, EPS_GRID
    p = os.path.join(out_dir(ds), "thp_results.json")
    if os.path.exists(p):
        d = json.load(open(p)); thp = d["methods"]; eps = np.array(d["config"]["eps_grid"])
    p = os.path.join(out_dir(ds), "ppm_results.json")
    if os.path.exists(p):
        ppm = json.load(open(p))
    return thp, ppm, eps


def generate_report(ds):
    thp, ppm, eps = _load(ds)
    if not thp and not ppm:
        print(f"  [{ds}] no results yet"); return
    mid = len(eps) // 2
    lines = []
    lines.append("=" * 92)
    lines.append(f"JEA REPORT — {ds}")
    lines.append("=" * 92)
    lines.append(f"\n{'Method':<20} {'Source':<10} {'JEA@mid%':>9} {'Act.Acc%':>9} {'MAE(d)':>8} {'N':>8}")
    lines.append("-" * 70)
    for n, r in thp.items():
        lines.append(f"{n:<20} {'THP':<10} {r['per_event_teacher'][mid]*100:>9.2f} "
                     f"{r['activity_accuracy_teacher']:>9.1f} {r['time_mae_teacher']:>8.3f} {'-':>8}")
    for n, r in ppm.items():
        lines.append(f"{n:<20} {'PPM':<10} {r['jea_at_eps_mid']:>9.2f} "
                     f"{r['act_accuracy']:>9.1f} {r['time_mae_days']:>8.3f} {r['n_instances']:>8}")
    # Remaining-time SOTA reference (paper)
    if ds in PAPER_RT_TABLE:
        tbl = PAPER_RT_TABLE[ds]
        best_m = min(tbl, key=tbl.get)
        lines.append(f"\nRemaining-time SOTA reference (BPM2025 UQ4PPM paper, MAE days):")
        lines.append("  " + "  ".join(f"{k}={v}" for k, v in tbl.items()))
        lines.append(f"  best = {best_m} ({tbl[best_m]:.2f} d)   "
                     f"[NB: paper MAE = remaining-time -> compare vs THP mae_rt, not next-event MAE]")
    report = "\n".join(lines)
    with open(os.path.join(out_dir(ds), "report.txt"), "w") as f:
        f.write(report)
    print(report)

    unified = {"thp": thp, "ppm": ppm, "eps_grid": eps.tolist(),
               "paper_rt_sota": PAPER_RT_TABLE.get(ds)}
    with open(os.path.join(out_dir(ds), "unified_results.json"), "w") as f:
        json.dump(unified, f, indent=2, default=_np_enc)


def _style():
    C = {"THP Mixture": "#e41a1c", "THP Baseline": "#377eb8",
         "Frequency BL": "#4daf4a", "Last-Event BL": "#984ea3",
         "SuTraN": "#a65628", "ED-LSTM": "#f781bf"}
    M = {"THP Mixture": "o", "THP Baseline": "^", "Frequency BL": "D",
         "Last-Event BL": "v", "SuTraN": "s", "ED-LSTM": "P"}
    return C, M


def make_plots(ds):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    thp, ppm, eps = _load(ds)
    if not thp and not ppm:
        print(f"  [{ds}] no results to plot"); return
    C, M = _style()

    # --- jea_vs_eps ---
    fig, ax = plt.subplots(figsize=(12, 7))
    for n, r in thp.items():
        if r.get("no_activity"):
            continue
        ax.plot(eps, np.array(r["per_event_teacher"]) * 100, color=C.get(n, "#333"),
                lw=2, marker=M.get(n, "o"), markersize=5, markevery=max(1, len(eps)//6),
                label=f"{n} (MAE={r['time_mae_teacher']:.2f}d)")
    for n, r in ppm.items():
        ax.plot(r["eps_grid"], r["jea_curve"], color=C.get(n, "#333"), lw=2.5,
                marker=M.get(n, "s"), markersize=5, markevery=max(1, len(r["eps_grid"])//6),
                linestyle="--", label=f"{n} (MAE={r['time_mae_days']:.2f}d)")
    ax.set_xscale("log"); ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0)
    ax.set_xlabel("Time tolerance ε (days)", fontsize=13); ax.set_ylabel("JEA (%)", fontsize=13)
    ax.set_title(f"Joint Event Accuracy vs ε — {ds}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir(ds), "jea_vs_eps.png"), dpi=150, bbox_inches="tight"); plt.close()

    # --- comparison bars: JEA@mid, Act.Acc, MAE ---
    names = [n for n in thp if not thp[n].get("no_activity")] + list(ppm)
    mid = len(eps) // 2
    jea_v = [thp[n]["per_event_teacher"][mid]*100 for n in thp if not thp[n].get("no_activity")] + [ppm[n]["jea_at_eps_mid"] for n in ppm]
    acc_v = [thp[n]["activity_accuracy_teacher"] for n in thp if not thp[n].get("no_activity")] + [ppm[n]["act_accuracy"] for n in ppm]
    # MAE includes THP Baseline + paper SOTA reference
    mae_names = [n for n in thp] + list(ppm)
    mae_v = [thp[n]["time_mae_teacher"] for n in thp] + [ppm[n]["time_mae_days"] for n in ppm]
    bc = [C.get(n, "#333") for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, vals, yl, tt in [(axes[0], jea_v, "JEA (%)", f"JEA (ε={eps[mid]:.4f}d)"),
                             (axes[1], acc_v, "Activity Acc (%)", "Activity Accuracy")]:
        x = np.arange(len(names)); bars = ax.bar(x, vals, color=bc, alpha=0.85, edgecolor="white")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(yl); ax.set_title(tt, fontweight="bold")
        for b, v in zip(bars, vals):
            if v > 0.3: ax.text(b.get_x()+b.get_width()/2, v+0.3, f"{v:.1f}", ha="center", fontsize=8)
    xm = np.arange(len(mae_names)); bcm = [C.get(n, "#333") for n in mae_names]
    bars = axes[2].bar(xm, mae_v, color=bcm, alpha=0.85, edgecolor="white")
    axes[2].set_xticks(xm); axes[2].set_xticklabels(mae_names, rotation=35, ha="right", fontsize=9)
    axes[2].set_ylabel("Next-event time MAE (days)"); axes[2].set_title("Time MAE (next-event)", fontweight="bold")
    for b, v in zip(bars, mae_v): axes[2].text(b.get_x()+b.get_width()/2, v, f"{v:.2f}", ha="center", fontsize=7)
    fig.suptitle(f"Comparison — {ds}", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir(ds), "comparison.png"), dpi=150, bbox_inches="tight"); plt.close()

    # --- heatmap ---
    hn = [n for n in thp if not thp[n].get("no_activity")] + list(ppm)
    mat = [np.array(thp[n]["per_event_teacher"])*100 for n in thp if not thp[n].get("no_activity")]
    for n in ppm:
        mat.append(np.interp(eps, ppm[n]["eps_grid"], ppm[n]["jea_curve"]))
    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=(14, max(3.5, len(hn)*0.7+1)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", origin="lower")
    ax.set_yticks(range(len(hn))); ax.set_yticklabels(hn, fontsize=10)
    tk = np.linspace(0, len(eps)-1, 6, dtype=int)
    ax.set_xticks(tk); ax.set_xticklabels([f"{eps[i]:.3f}" for i in tk], fontsize=9)
    ax.set_xlabel("Time tolerance ε (days)"); ax.set_title(f"JEA heatmap (%) — {ds}", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="JEA (%)"); plt.tight_layout()
    plt.savefig(os.path.join(out_dir(ds), "heatmap.png"), dpi=150, bbox_inches="tight"); plt.close()

    # --- jea_vs_topk ---
    has_topk = any("topk_jea" in r for r in thp.values()) or any("topk_jea" in r for r in ppm.values())
    if has_topk:
        eps_idx = [len(eps)//4, len(eps)//2, 3*len(eps)//4]
        fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
        for ai, ei in enumerate(eps_idx):
            ax = axes[ai]
            for n, r in thp.items():
                if "topk_jea" not in r: continue
                c = r["topk_jea"][str(ei)]
                ax.plot(range(1, len(c)+1), c, color=C.get(n, "#333"), lw=2, marker=M.get(n, "o"),
                        markersize=4, markevery=max(1, len(c)//6), label=n)
            for n, r in ppm.items():
                if "topk_jea" not in r: continue
                pe = int(np.argmin(np.abs(np.array(r["eps_grid"]) - eps[ei])))
                c = r["topk_jea"][str(pe)]
                ax.plot(range(1, len(c)+1), c, color=C.get(n, "#333"), lw=2.5, marker=M.get(n, "s"),
                        markersize=4, markevery=max(1, len(c)//6), linestyle="--", label=n)
            ax.set_xlabel("Top-k activity tolerance", fontsize=12)
            if ai == 0: ax.set_ylabel("JEA (%)", fontsize=13)
            ax.set_title(f"ε = {eps[ei]:.4f} d", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3); ax.set_ylim(bottom=0); ax.legend(fontsize=8, loc="upper left")
        fig.suptitle(f"JEA vs activity top-k — {ds}", fontsize=14, fontweight="bold")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir(ds), "jea_vs_topk.png"), dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [{ds}] plots saved to {out_dir(ds)}/")


def _np_enc(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(str(type(o)))


# =========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="UQ dataset, e.g. UQ_HelpDesk")
    ap.add_argument("--all", action="store_true", help="apply to all 10 UQ datasets")
    ap.add_argument("--thp", action="store_true")
    ap.add_argument("--ppm", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--decode", choices=["matched", "mean"], default="matched",
                    help="THP-Mixture point estimates: 'matched' = median for MAE "
                         "+ eps-mode for JEA (metric-optimal, default); "
                         "'mean' = legacy E[dt] decode (ablation)")
    args = ap.parse_args()

    targets = UQ_DATASETS if args.all else ([args.dataset] if args.dataset else [])
    if not targets:
        ap.error("provide --dataset <name> or --all")

    for ds in targets:
        if args.thp:
            print(f"\n=== THP eval: {ds} ({args.decode} decode) ==="); run_thp(ds, decode=args.decode)
        if args.ppm:
            print(f"\n=== PPM eval: {ds} ==="); run_ppm(ds)
        if args.report:
            print(f"\n=== Report: {ds} ==="); generate_report(ds)
        if args.plots:
            make_plots(ds)
    if not any([args.thp, args.ppm, args.report, args.plots]):
        ap.print_help()
