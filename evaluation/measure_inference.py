#!/usr/bin/env python
"""Inference-cost benchmark for the timing table (tab:timing).

Times each model's standard test-set inference, GPU-synchronized, after one
warm-up pass. The measured quantity is what each model must run to produce its
reported next-event predictions:

  THP-B / THP-M    one teacher-forced forward over the padded test sequences
                   (all prefixes of a case in a single pass, non-AR)
  SuTraN / ED-LSTM their own inference_loop on test_tensordataset.pt, i.e. the
                   full autoregressive suffix decode whose step-0 yields the
                   next event (the models have no cheaper next-event mode).
                   Their code is not vendored here; set SUTRAN_DIR to a checkout
                   to run --baselines

Appends rows to paper_results/inference_timings.csv:
    dataset,model,wall_s,n_prefixes,ms_per_prefix

Usage:
    python measure_inference.py --thp
    SUTRAN_DIR=<checkout> python measure_inference.py --baselines
"""

import os
import sys
import csv
import time
import shutil
import argparse
import tempfile

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)          # THP_for_PPM/
THP_DIR = REPO_ROOT
# SuTraN / ED-LSTM are not vendored here; point this at a checkout of
# https://github.com/BrechtWts/SuffixTransformerNetwork to time them.
SUTRAN_DIR = os.environ.get("SUTRAN_DIR", os.path.join(REPO_ROOT, "..", "SuffixTransformerNetwork"))
OUT_CSV = os.path.join(REPO_ROOT, "paper_results", "inference_timings.csv")

FOLD = 0
N_MIX = 5
SEED = 0

UQ_DATASETS = [
    "UQ_HelpDesk", "UQ_Sepsis", "UQ_BPIC13I", "UQ_BPIC20DD", "UQ_BPIC20RFP",
    "UQ_BPIC20PTC", "UQ_BPIC20ID", "UQ_BPIC20TPD", "UQ_BPIC12", "UQ_BPIC15_1",
]


def append_row(dataset, model, wall_s, n_prefixes):
    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["dataset", "model", "wall_s", "n_prefixes", "ms_per_prefix"])
        w.writerow([dataset, model, f"{wall_s:.3f}", n_prefixes,
                    f"{1000.0 * wall_s / max(n_prefixes, 1):.4f}"])
    print(f"  {dataset} {model}: {wall_s:.2f}s over {n_prefixes} prefixes "
          f"({1000.0 * wall_s / max(n_prefixes, 1):.3f} ms/prefix)", flush=True)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
# THP (mpp env)
# --------------------------------------------------------------------------- #

def bench_thp():
    sys.path.insert(0, THP_DIR)
    cwd = os.getcwd()
    os.chdir(THP_DIR)
    from preprocess.dataset import df_to_dict, get_dataloader
    from transformer.model import Transformer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ds in UQ_DATASETS:
        fold_file = f"fold{FOLD}_variation0_{ds}"
        train_out, _, test_out = df_to_dict(
            directory=os.path.join(THP_DIR, "data"), fold_filename=fold_file,
        )
        num_types = train_out["dim_process"]
        loader = get_dataloader(test_out["test"], batch_size=64, shuffle=False)
        n_prefixes = sum(max(len(seq) - 1, 0) for seq in test_out["test"])

        for mtype, tag in (("baseline", "THP-B"), ("baseline_act", "THP-B-act"),
                           ("mixture", "THP-M")):
            ck = os.path.join(THP_DIR, "saved_models",
                              f"{fold_file}_{mtype}_best_model_best_mae.pth")
            if not os.path.exists(ck):
                print(f"  {ds} {tag}: checkpoint missing, skipped", flush=True)
                continue
            model = Transformer(num_types=num_types, d_model=36, d_rnn=256,
                                d_inner=128, n_layers=4, n_head=4, d_k=16,
                                d_v=16, dropout=0.1, n_mix=N_MIX,
                                model_type=mtype)
            model.load_state_dict(
                torch.load(ck, map_location=device)["model_state_dict"])
            model.to(device).eval()

            with torch.no_grad():
                for batch in loader:                      # warm-up pass
                    event_time, _, _, event_type = (x.to(device) for x in batch)
                    model(event_type, event_time)
                    break
                _sync(device)
                t0 = time.perf_counter()
                for batch in loader:
                    event_time, _, _, event_type = (x.to(device) for x in batch)
                    model(event_type, event_time)
                _sync(device)
                wall = time.perf_counter() - t0
            append_row(ds, tag, wall, n_prefixes)
    os.chdir(cwd)


# --------------------------------------------------------------------------- #
# SuTraN / ED-LSTM (fasutran env) — reuses fit_baseline_wrapper's loaders
# --------------------------------------------------------------------------- #

def bench_baselines():
    sys.path.insert(0, SCRIPT_DIR)
    import fit_baseline_wrapper as fbw

    os.chdir(SUTRAN_DIR)
    sys.path.insert(0, SUTRAN_DIR)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ds in UQ_DATASETS:
        meta = fbw.get_metadata(ds)
        test_dataset = torch.load(os.path.join(ds, "test_tensordataset.pt"))
        test_sliced = fbw.slice_nda(test_dataset, meta["num_categoricals_pref"])
        n_prefixes = test_sliced[0].shape[0]

        for name, sub, kind in fbw.MODELS:
            backup = os.path.join(SUTRAN_DIR, ds, sub)
            ckpt_path = fbw.best_checkpoint(backup)
            if ckpt_path is None:
                print(f"  {ds} {name}: no checkpoint, skipped", flush=True)
                continue
            if kind == "sutran_nda":
                from SuTraN.SuTraN import SuTraN_no_context
                from SuTraN.inference_procedure import inference_loop
                model = SuTraN_no_context(
                    num_activities=meta["num_activities"], d_model=32,
                    num_prefix_encoder_layers=4, num_decoder_layers=4,
                    num_heads=8, d_ff=128, dropout=0.2,
                    remaining_runtime_head=True, layernorm_embeds=True,
                    outcome_bool=False).to(device)
            else:
                from LSTM_seq2seq.enc_dec_lstm import EncDecLSTM_no_context
                from LSTM_seq2seq.inference_procedure import inference_loop
                model = EncDecLSTM_no_context(
                    num_activities=meta["num_activities"], d_model=64,
                    n_layers=4, dropout=0.2).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            def run_once(res_path):
                with torch.no_grad():
                    if kind == "sutran_nda":
                        inference_loop(model, test_sliced, True, False, 1,
                                       meta["mean_std_ttne"], meta["mean_std_tsp"],
                                       meta["mean_std_tss"], meta["mean_std_rrt"],
                                       results_path=res_path, val_batch_size=2048)
                    else:
                        inference_loop(model, test_sliced, 1,
                                       meta["mean_std_ttne"], meta["mean_std_tsp"],
                                       meta["mean_std_tss"], meta["mean_std_rrt"],
                                       results_path=res_path, val_batch_size=2048)

            tmp = tempfile.mkdtemp(prefix="inferbench_")
            try:
                run_once(tmp)                              # warm-up
                _sync(device)
                t0 = time.perf_counter()
                run_once(tmp)
                _sync(device)
                wall = time.perf_counter() - t0
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            append_row(ds, name, wall, n_prefixes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--thp", action="store_true")
    ap.add_argument("--baselines", action="store_true")
    args = ap.parse_args()
    if not (args.thp or args.baselines):
        ap.error("provide --thp and/or --baselines")
    if args.thp:
        bench_thp()
    if args.baselines:
        bench_baselines()
