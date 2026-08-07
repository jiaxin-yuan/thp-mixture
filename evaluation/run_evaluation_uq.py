#!/usr/bin/env python3
"""Next event evaluation across the 10 UQ4PPM logs: the activity accuracy and
the next gap MAE of Table 3.

  THP Mixture       argmax(mark_head), point decode of the zero-inflated
                    LogNormal mixture for time
  THP Baseline      intensity MLE model, regression time heads + activity
  Frequency / Last-Event   activity+time reference points
  SuTraN / ED-LSTM  decoupled PPM, step-0 of the suffix prediction

All methods share the chronological 80/20 split and prefixes [2, n-1], scored
per position under teacher-forcing.

Usage (--thp needs the `mpp` env; --ppm needs `fasutran`; --report either),
with --all in place of --dataset to sweep every log:
  conda run -n mpp      python run_evaluation_uq.py --thp    --dataset UQ_HelpDesk
  conda run -n fasutran python run_evaluation_uq.py --ppm    --dataset UQ_HelpDesk
  python run_evaluation_uq.py --report --dataset UQ_HelpDesk
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

UQ_DATASETS = [
    "UQ_HelpDesk", "UQ_Sepsis", "UQ_BPIC13I", "UQ_BPIC20DD",
    "UQ_BPIC20RFP", "UQ_BPIC20PTC", "UQ_BPIC20ID", "UQ_BPIC20TPD",
    "UQ_BPIC12", "UQ_BPIC15_1",
]

def out_dir(ds):
    d = os.path.join(PER_DATASET, ds)
    os.makedirs(d, exist_ok=True)
    return d


def dump_persample(ds, tag, correct, abs_err, prefix_len, true_gap, conf=None):
    """Dump one row per scored next event (hit, |gap error| in days, prefix length, true gap, max activity softmax) to persample_<tag>.npz, leaving the binning to the plotting script."""
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


def _finalize_from_arrays(correct, dt_err):
    """Activity hit rate (%) and next gap MAE (days) from flat per-event arrays."""
    correct = np.asarray(correct, dtype=bool)
    dt_err = np.asarray(dt_err, dtype=float)
    return {
        "activity_accuracy_teacher": float(correct.mean()) * 100,
        "time_mae_teacher": float(dt_err.mean()),
    }


def run_thp(ds, decode="median"):
    """Score both THP variants, THP-M decoded by the median of its density (the paper's column) or by E[dt]."""
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
        # main.py pulls in sklearn through trainer/train.py, absent from mpp
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

    mix_ckpt = os.path.join(THP_DIR, "saved_models",
                            f"{fold_file}_mixture_best_model_best_mae.pth")
    if os.path.exists(mix_ckpt):
        model = build_model(num_types, device, n_mix=n_mix, model_type="mixture")
        model.load_state_dict(torch.load(mix_ckpt, map_location=device)["model_state_dict"])
        model.eval()
        cor, derr, pls = [], [], []
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
                logit_t = mark_logit[:, :-1, :]                    # [B,S-1,C]

                vm = valid.reshape(-1)
                cor.extend(((pred_mark == gt_mark).reshape(-1)[vm]).cpu().numpy().tolist())
                derr.extend((torch.abs(pdt - gdt).reshape(-1)[vm]).cpu().numpy().tolist())
                idx = torch.arange(1, event_type.size(1), device=device).unsqueeze(0).expand(event_type.size(0), -1)
                pls.extend((idx.reshape(-1)[vm]).cpu().numpy().tolist())
                tg.extend((gdt.reshape(-1)[vm]).cpu().numpy().tolist())
                conf_b = torch.softmax(logit_t, -1).max(-1).values    # [B,S-1]
                cf.extend((conf_b.reshape(-1)[vm]).cpu().numpy().tolist())
                if decode == "median":
                    pi_b, lw_b, mu_b, sg_b = ge.extract_params(
                        tie_logit[:, :-1], time_params[:, :-1], n_mix)
                    gdts.append(gdt.reshape(-1)[vm])
                    pis.append(pi_b.reshape(-1)[vm])
                    log_ws.append(lw_b.reshape(-1, n_mix)[vm])
                    mus.append(mu_b.reshape(-1, n_mix)[vm])
                    sigmas.append(sg_b.reshape(-1, n_mix)[vm])

        if decode == "median":
            with torch.no_grad():
                dt = torch.cat(gdts); pi = torch.cat(pis); log_w = torch.cat(log_ws)
                mu = torch.cat(mus); sigma = torch.cat(sigmas)
                # the MAE-optimal point of the density
                derr = (ge.quantile(0.5, pi, log_w, mu, sigma) - dt).abs().cpu().numpy()
        methods["THP Mixture"] = _finalize_from_arrays(cor, derr)
        methods["THP Mixture"]["decode"] = decode
        print(f"    THP Mixture: acc={methods['THP Mixture']['activity_accuracy_teacher']:.1f}% "
              f"MAE={methods['THP Mixture']['time_mae_teacher']:.3f}d")
        dump_persample(ds, "thp_mixture", cor, np.asarray(derr), pls, tg, cf)
    else:
        print(f"    SKIP THP Mixture: {mix_ckpt} not found")

    use_ckpt = os.path.join(THP_DIR, "saved_models",
                            f"{fold_file}_baseline_act_best_model_best_mae.pth")
    if os.path.exists(use_ckpt):
        model = build_model(num_types, device, n_mix=n_mix,
                            model_type="baseline_act")
        bl_state = torch.load(use_ckpt, map_location=device)
        model.load_state_dict(bl_state["model_state_dict"])
        model.eval()
        # the heads regress z-scored targets, de-standardize to days
        ts = bl_state["time_stats"]
        dt_mean, dt_std = ts["dt_mean"], ts["dt_std"]
        cor, derr, pls = [], [], []
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
                mark_logit = heads[2]
                pred_mark = mark_logit[:, :-1, :].argmax(-1)
                gt_mark = event_type[:, 1:] - 1
                logit_t = mark_logit[:, :-1, :]
                cor.extend(((pred_mark == gt_mark).reshape(-1)[vm]).cpu().numpy().tolist())
                conf_b = torch.softmax(logit_t, -1).max(-1).values
                cf.extend((conf_b.reshape(-1)[vm]).cpu().numpy().tolist())
        methods["THP Baseline"] = _finalize_from_arrays(cor, derr)
        methods["THP Baseline"]["arch"] = "baseline_act"
        print(f"    THP Baseline: acc={methods['THP Baseline']['activity_accuracy_teacher']:.1f}% "
              f"MAE={methods['THP Baseline']['time_mae_teacher']:.3f}d")
        dump_persample(ds, "thp_baseline", cor, np.asarray(derr), pls, tg, cf)
    else:
        print(f"    SKIP THP Baseline: {use_ckpt} not found")

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
        methods["Frequency BL"] = _finalize_from_arrays(fc, fd)
        methods["Last-Event BL"] = _finalize_from_arrays(lc, ld)

    output = {"config": {"fold": FOLD, "dataset": ds}, "methods": methods}
    with open(os.path.join(out_dir(ds), "thp_results.json"), "w") as f:
        json.dump(output, f, indent=2)
    print(f"    Saved thp_results.json")


def run_ppm(ds):
    """Score step-0 of each baseline's cached suffix predictions."""
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

        rrt_mae_days = None
        rrt_p = os.path.join(rdir, "MAE_rrt_minutes.pt")
        if os.path.exists(rrt_p):
            rrt = torch.load(rrt_p, map_location="cpu", weights_only=False)
            rrt_mae_days = float(rrt.float().mean()) / 1440.0

        conf = None            # max activity softmax, for the per-sample dump
        lp = os.path.join(rdir, "suffix_act_logits.pt")
        if os.path.exists(lp):
            logits = torch.load(lp, map_location="cpu", weights_only=False)[:, 0, :].numpy()[valid]
            ex = np.exp(logits - logits.max(axis=1, keepdims=True))
            conf = (ex / ex.sum(axis=1, keepdims=True)).max(axis=1)
        dump_persample(ds, "sutran" if name == "SuTraN" else "edlstm",
                       act_ok.astype(np.float32), t_err, pref_v, gt_t, conf)

        results[name] = {
            "n_instances": int(valid.sum()),
            "act_accuracy": float(act_ok.mean() * 100),
            "time_mae_days": float(t_err.mean()),
            "rrt_mae_days": rrt_mae_days,
        }
        print(f"  {name}: acc={results[name]['act_accuracy']:.1f}% "
              f"MAE={results[name]['time_mae_days']:.3f}d N={results[name]['n_instances']}")

    with open(os.path.join(out_dir(ds), "ppm_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved ppm_results.json")


def _load(ds):
    """Read back the two per-dataset result files."""
    thp, ppm = {}, {}
    p = os.path.join(out_dir(ds), "thp_results.json")
    if os.path.exists(p):
        thp = json.load(open(p))["methods"]
    p = os.path.join(out_dir(ds), "ppm_results.json")
    if os.path.exists(p):
        ppm = json.load(open(p))
    return thp, ppm


def generate_report(ds):
    """Print and store the per-method table."""
    thp, ppm = _load(ds)
    if not thp and not ppm:
        print(f"  [{ds}] no results yet"); return
    lines = []
    lines.append("=" * 92)
    lines.append(f"NEXT EVENT REPORT — {ds}")
    lines.append("=" * 92)
    lines.append(f"\n{'Method':<20} {'Source':<10} {'Act.Acc%':>9} {'MAE(d)':>8} {'N':>8}")
    lines.append("-" * 60)
    for n, r in thp.items():
        lines.append(f"{n:<20} {'THP':<10} "
                     f"{r['activity_accuracy_teacher']:>9.1f} {r['time_mae_teacher']:>8.3f} {'-':>8}")
    for n, r in ppm.items():
        lines.append(f"{n:<20} {'PPM':<10} "
                     f"{r['act_accuracy']:>9.1f} {r['time_mae_days']:>8.3f} {r['n_instances']:>8}")
    report = "\n".join(lines)
    with open(os.path.join(out_dir(ds), "report.txt"), "w") as f:
        f.write(report)
    print(report)

    unified = {"thp": thp, "ppm": ppm}
    with open(os.path.join(out_dir(ds), "unified_results.json"), "w") as f:
        json.dump(unified, f, indent=2, default=_np_enc)


def _np_enc(o):
    """JSON encoder for numpy scalars and arrays."""
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(str(type(o)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="UQ dataset, e.g. UQ_HelpDesk")
    ap.add_argument("--all", action="store_true", help="apply to all 10 UQ datasets")
    ap.add_argument("--thp", action="store_true")
    ap.add_argument("--ppm", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--decode", choices=["median", "mean"], default="median",
                    help="THP-Mixture time point estimate: 'median' = the "
                         "MAE-optimal point of the density (default, the paper's "
                         "decode); 'mean' = E[dt] (ablation, not a paper column)")
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
    if not any([args.thp, args.ppm, args.report]):
        ap.print_help()
