#!/usr/bin/env python3
"""Rebuild the paper's result tables from the stored per-method artifacts.

Table 3 (next event)     Act% and MAE in days, per method:
    THP-B, THP-M         paper_results/per_dataset/<ds>/thp_results.json
    SuTraN, ED-LSTM      paper_results/per_dataset/<ds>/ppm_results.json
    H-uni, H-mk          paper_results/per_dataset/<ds>/hawkes_results.json
Table 4 (remaining time) RT-MAE in days, and MA / MPIW / AURG for THP-M:
    THP-B                paper_results/thp_baseline_testline.csv
    THP-M                paper_results/per_dataset/<ds>/rt_uncertainty_results.json
    SuTraN, ED-LSTM      paper_results/per_dataset/<ds>/ppm_results.json
    UQ*                  UQ4PPM_RT, the per-metric best over the eight methods
                         reported by Amiri Elyasi et al., inlined below
Table 5 (cost)           paper_results/{time_consumption_comparison,inference_timings}.csv

Writes paper_results/{nextevent,remaining_time,cost}_table.csv and prints the
tables. With --check, every cell is compared column by column against
paper_results/paper_tables.csv, the frozen record of the numbers printed in the
paper; mismatches, and cells with no source in this repo, are reported
(exit code 1 on mismatch).

Usage:
    python evaluation/make_tables.py [--check]
"""

import argparse
import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
PER_DATASET = os.path.join(REPO_ROOT, "paper_results", "per_dataset")
OUT_DIR = os.path.join(REPO_ROOT, "paper_results")

# Group A, Group B, outlier: the order used in the paper's tables.
DATASETS = [
    ("UQ_Sepsis", "Sepsis"), ("UQ_BPIC13I", "BPIC13I"), ("UQ_BPIC12", "BPIC12"),
    ("UQ_HelpDesk", "HelpDesk"), ("UQ_BPIC20DD", "BPIC20DD"),
    ("UQ_BPIC20RFP", "BPIC20RFP"), ("UQ_BPIC20PTC", "BPIC20PTC"),
    ("UQ_BPIC20ID", "BPIC20ID"), ("UQ_BPIC20TPD", "BPIC20TPD"),
    ("UQ_BPIC15_1", "BPIC15-1"),
]

# Remaining time MAE (days) of the eight uncertainty-aware methods of
# Amiri Elyasi et al., under their own split; UQ* is the per-metric best.
UQ4PPM_RT = {
    "UQ_Sepsis":     15.32, "UQ_BPIC13I":   3.06, "UQ_BPIC12":     5.86,
    "UQ_HelpDesk":    9.45, "UQ_BPIC20DD":  4.12, "UQ_BPIC20RFP":  5.09,
    "UQ_BPIC20PTC":   7.73, "UQ_BPIC20ID": 14.08, "UQ_BPIC20TPD": 25.02,
    "UQ_BPIC15_1":   24.71,
}


def _load(ds, name):
    path = os.path.join(PER_DATASET, ds, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _thp_baseline_testline():
    """THP-B TEST line per dataset, archived from its training run.

    No evaluation script recomputes the THP-B remaining time: it is the
    cumulative sum of one-step expectations reported by main.py at the end of
    training, so the run's TEST line is its only source and is shipped as
    paper_results/thp_baseline_testline.csv.
    """
    path = os.path.join(OUT_DIR, "thp_baseline_testline.csv")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["dataset"]: {k: float(v) for k, v in r.items() if k != "dataset"}
                for r in csv.DictReader(f)}


def next_event_rows():
    rows = []
    for ds, label in DATASETS:
        thp = (_load(ds, "thp_results.json") or {}).get("methods", {})
        ppm = _load(ds, "ppm_results.json") or {}
        hawk = (_load(ds, "hawkes_results.json") or {}).get("methods", {})
        row = {"dataset": label}
        for key, src in [("THP-B", thp.get("THP Baseline")),
                         ("THP-M", thp.get("THP Mixture")),
                         ("H-uni", hawk.get("Hawkes (uni)")),
                         ("H-mk",  hawk.get("Hawkes (marked)"))]:
            row[f"act_{key}"] = None if src is None else src.get("activity_accuracy_teacher")
            row[f"mae_{key}"] = None if src is None else src.get("time_mae_teacher")
        for key in ("SuTraN", "ED-LSTM"):
            src = ppm.get(key)
            row[f"act_{key}"] = None if src is None else src.get("act_accuracy")
            row[f"mae_{key}"] = None if src is None else src.get("time_mae_days")
        if row.get("act_H-uni") == 0.0:          # time-only model, no activity
            row["act_H-uni"] = None
        rows.append(row)
    return rows


def remaining_time_rows():
    rows = []
    baseline = _thp_baseline_testline()
    for ds, label in DATASETS:
        ppm = _load(ds, "ppm_results.json") or {}
        rt = (_load(ds, "rt_uncertainty_results.json") or {}).get("models", {})
        mix = rt.get("THP-Mixture (MC rollout)", {})
        bl = baseline.get(ds)
        rows.append({
            "dataset":     label,
            "rtmae_THP-B": None if bl is None else bl.get("mae_rt"),
            "rtmae_THP-M": mix.get("mae"),
            "rtmae_SuTraN":  (ppm.get("SuTraN") or {}).get("rrt_mae_days"),
            "rtmae_ED-LSTM": (ppm.get("ED-LSTM") or {}).get("rrt_mae_days"),
            "rtmae_UQ*":   UQ4PPM_RT.get(ds),
            "ma_THP-M":    mix.get("ma"),
            "mpiw_THP-M":  mix.get("mpiw_rms_sigma"),
            "aurg_THP-M":  mix.get("aurg"),
        })
    return rows


def cost_rows():
    """Training minutes and inference ms/prefix, from the two timing CSVs.

    time_consumption_comparison.csv is wide (one <model>_wall_min column per
    model), inference_timings.csv is long (one row per dataset and model).
    """
    train, infer = {}, {}
    p = os.path.join(OUT_DIR, "time_consumption_comparison.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            train[r["dataset"]] = {k[: -len("_wall_min")]: v
                                   for k, v in r.items() if k.endswith("_wall_min")}
    p = os.path.join(OUT_DIR, "inference_timings.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            infer.setdefault(r["dataset"], {})[r["model"]] = r["ms_per_prefix"]
    rows = []
    for ds, label in DATASETS:
        row = {"dataset": label}
        for m, v in sorted(train.get(ds, {}).items()):
            row[f"train_min_{m}"] = v
        for m, v in sorted(infer.get(ds, {}).items()):
            row[f"infer_ms_{m}"] = v
        rows.append(row)
    return rows


def write_csv(rows, name):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.relpath(path, REPO_ROOT)}")


def show(rows, title, cols, digits=2, headers=None):
    print(f"\n{title}")
    heads = headers or [c.split("_", 1)[-1] for c in cols]
    print("  " + "".join(f"{h:>12}" for h in ["dataset"] + heads))
    for r in rows:
        cells = [f"{r['dataset']:>12}"]
        for c in cols:
            v = r.get(c)
            cells.append(f"{v:>12.{digits}f}" if isinstance(v, float) else f"{'--':>12}")
        print("  " + "".join(cells))


# Rounding the paper prints each cell at, and the tolerance the check applies.
# Timing is a wall-clock measurement with run-to-run spread, so it is compared
# with a relative tolerance rather than at a fixed rounding.
DIGITS = {"act": 1, "mae": 2, "rtmae": 2, "ma": 3, "mpiw": 2, "aurg": 3,
          "cases": 0, "events": 0, "avg_len": 1, "tie_pct": 1, "avg_dur_d": 1}
RTOL = {"train_min": 0.02, "infer_ms": 0.02, "med_dt_s": 0.01}


def paper_cells():
    """Every number printed in the paper's tables, keyed (table, dataset, column).

    paper_tables.csv is the frozen record of the typeset tables; the check
    compares column by column against it, so a value landing in the wrong
    column is caught.
    """
    path = os.path.join(OUT_DIR, "paper_tables.csv")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {(r["table"], r["dataset"], r["column"]): float(r["value"])
                for r in csv.DictReader(f)}


def _lacr():
    """LA-CR* per-metric best over the two calibration variants LA+I / LA+S."""
    path = os.path.join(OUT_DIR, "baselines", "lacr_summary.csv")
    out = {}
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path)):
        out[r["dataset"]] = {
            "ma_LA-CR*":   min(float(r["LA+I_ma"]), float(r["LA+S_ma"])),
            "mpiw_LA-CR*": min(float(r["LA+I_mpiw_rms_sigma"]),
                               float(r["LA+S_mpiw_rms_sigma"])),
            "aurg_LA-CR*": max(float(r["LA+I_aurg"]), float(r["LA+S_aurg"])),
        }
    return out


def recomputed_cells(ne_rows, rt_rows):
    """The same cells, recomputed from the shipped artifacts."""
    by_label = {label: ds for ds, label in DATASETS}
    out = {}

    for rows, table in ((ne_rows, "nextevent"), (rt_rows, "rtuq")):
        for row in rows:
            ds = by_label[row["dataset"]]
            for col, v in row.items():
                if col != "dataset" and v is not None:
                    out[(table, ds, col)] = v

    for ds, cols in _lacr().items():
        for col, v in cols.items():
            out[("rtuq", ds, col)] = v

    path = os.path.join(OUT_DIR, "dataset_stats.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            if r["stage"] != "raw":          # Table 1 counts the logs as published
                continue
            for col in ("cases", "events", "act", "avg_len",
                        "med_dt_s", "tie_pct", "avg_dur_d"):
                out[("datasets", r["dataset"], col)] = float(r[col])

    # The paper's THP-B is the baseline_act run, so its inference row is the
    # THP-B-act model of inference_timings.csv; the plain baseline is not in
    # the paper. Repeated measurements are appended, and the last one stands.
    path = os.path.join(OUT_DIR, "time_consumption_comparison.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            for src, col in (("THP-baseline", "THP-B"), ("THP-mixture", "THP-M"),
                             ("SuTraN", "SuTraN"), ("ED-LSTM", "ED-LSTM"),
                             ("Hawkes", "Hawkes")):
                v = r.get(f"{src}_wall_min")
                if v not in (None, "", "NA"):
                    out[("timing", r["dataset"], f"train_min_{col}")] = float(v)
    path = os.path.join(OUT_DIR, "inference_timings.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            col = {"THP-B-act": "THP-B", "THP-M": "THP-M",
                   "SuTraN": "SuTraN", "ED-LSTM": "ED-LSTM"}.get(r["model"])
            if col:
                out[("timing", r["dataset"], f"infer_ms_{col}")] = float(r["ms_per_prefix"])
    path = os.path.join(OUT_DIR, "baselines", "lacr_inference_timing.csv")
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            out[("timing", r["dataset"], "infer_ms_LA-CR")] = float(r["ms_per_prefix"])
    return out


def check_against_paper(ne_rows, rt_rows):
    """Compare every recomputed cell with the number the paper prints."""
    paper = paper_cells()
    if not paper:
        print("[skip] paper_tables.csv not present, nothing to check against")
        return 0
    mine = recomputed_cells(ne_rows, rt_rows)

    bad = checked = 0
    for key in sorted(paper):
        table, ds, col = key
        if key not in mine:
            continue
        want, got = paper[key], mine[key]
        checked += 1
        base = ("train_min" if col.startswith("train_min") else
                "infer_ms" if col.startswith("infer_ms") else col.split("_")[0])
        if base in RTOL or col in RTOL:
            tol = RTOL.get(col, RTOL.get(base))
            ok = abs(got - want) <= max(6e-4, tol * want)
        else:
            d = DIGITS.get(col.split("_")[0], DIGITS.get(col, 2))
            ok = f"{got:.{d}f}" == f"{want:.{d}f}"
        if not ok:
            bad += 1
            print(f"[MISMATCH] {table} {ds} {col}: paper {want}, recomputed {got}")

    missing = sorted(k for k in paper if k not in mine)
    print(f"\n{checked} of {len(paper)} printed cells recomputed and compared")
    if missing:
        print(f"{len(missing)} printed cells have no source in this repo, "
              f"so they cannot be checked:")
        for table, ds, col in missing:
            print(f"    {table:10s} {ds:14s} {col}")
    print("\nevery recomputable cell matches the paper" if not bad
          else f"\n{bad} mismatches")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify the values against the paper's printed numbers")
    args = ap.parse_args()

    ne, rt, cost = next_event_rows(), remaining_time_rows(), cost_rows()

    show(ne, "Table 3  next activity accuracy (%)",
         ["act_THP-B", "act_THP-M", "act_SuTraN", "act_ED-LSTM", "act_H-uni", "act_H-mk"], 1)
    show(ne, "Table 3  next event time MAE (days)",
         ["mae_THP-B", "mae_THP-M", "mae_SuTraN", "mae_ED-LSTM", "mae_H-uni", "mae_H-mk"])
    show(rt, "Table 4  remaining time MAE (days)",
         ["rtmae_THP-B", "rtmae_THP-M", "rtmae_SuTraN", "rtmae_ED-LSTM", "rtmae_UQ*"])
    show(rt, "Table 4  uncertainty metrics, THP-M",
         ["ma_THP-M", "mpiw_THP-M", "aurg_THP-M"], 3, headers=["MA", "MPIW", "AURG"])

    write_csv(ne, "nextevent_table.csv")
    write_csv(rt, "remaining_time_table.csv")
    write_csv(cost, "cost_table.csv")

    if args.check:
        raise SystemExit(1 if check_against_paper(ne, rt) else 0)


if __name__ == "__main__":
    main()
