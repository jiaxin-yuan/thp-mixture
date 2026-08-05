"""Prepare the UQ4PPM datasets for the THP_for_PPM pipeline.

Strict temporal split following Weytjens' predictive-process-monitoring-benchmarks
(create_benchmarks.py), so that no training case completes after a test case
starts and the dataset end is not right-censoring biased.

  1. Drop duplicates, optionally trim chronological outliers by date
     (disabled unless START_MONTH / END_MONTH are set).
  2. Debias the dataset end: latest_start = max_ts - max_duration, keep only
     cases starting on or before it.
  3. Test set = the last 20% of cases by start time, which defines T_sep.
  4. Train set = the cases completing before T_sep.
  5. Debias the test set start: for cases straddling T_sep, keep only prefixes
     whose prediction origin e_k lies in [T_sep, latest_start]. The model has
     no NaN-target mechanism, so such prefixes are dropped, not masked.
  6. Drop up to the 5% longest cases, at the duration cutoff maximizing the
     training set size (percentiles 95..100).

For a trace of length n, prefixes k = 2..n-1 are written with their target
event (k+1 events) under CaseID "<id>_p{k}"; CaseEndTime carries the original
trace end used for remaining time.

Writes to data/: {train,val,test,full_train}_fold0_variation0_{dataset}.csv
"""

import argparse
import os
import csv

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_DIR = Path(os.environ.get("UQ4PPM_RAW", Path(__file__).parent / "data" / "uq4ppm_csv"))
OUTPUT_DIR = Path(__file__).parent / "data"

DATASETS = {
    "BPIC12":    "UQ_BPIC12",
    "BPIC13I":   "UQ_BPIC13I",
    "BPIC15_1":  "UQ_BPIC15_1",
    "BPIC20DD":  "UQ_BPIC20DD",
    "BPIC20ID":  "UQ_BPIC20ID",
    "BPIC20PTC": "UQ_BPIC20PTC",
    "BPIC20RFP": "UQ_BPIC20RFP",
    "BPIC20TPD": "UQ_BPIC20TPD",
    "HelpDesk":  "UQ_HelpDesk",
    "Sepsis":    "UQ_Sepsis",
}

TEST_LEN        = 0.20          # last 20% of cases (by start time) -> test
VAL_LEN         = 0.20          # last 20% of TRAIN cases (by start time) -> val
MIN_TRACE_LEN   = 3             # need >=3 events: prefix(2) + target
STEP6_PCTL_LO   = 95.0          # step-6 sweep: drop up to top 5% longest cases
STEP6_PCTL_HI   = 100.0
STEP6_PCTL_STEP = 0.5
MIN_TRAIN_CASES = 1             # collapse guard for the step-6 sweep
MIN_TRAIN_FALLBACK = 50         # below this, retry without the step-2 end debias,
                                # which otherwise empties back-loaded logs
TIME_FMT        = "%Y/%m/%d %H:%M:%S.%f"

# Step-1 date trim, disabled unless set to period strings such as "2012-01".
START_MONTH = None
END_MONTH   = None


def load_and_standardize(csv_name: str) -> pd.DataFrame:
    """Load a raw CSV into long-form events with standardized names and tz-aware UTC timestamps."""
    csv_path = INPUT_DIR / f"{csv_name}.csv"
    df = pd.read_csv(csv_path)

    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ('caseid', 'case:concept:name'):
            col_map[col] = 'CaseID'
        elif cl in ('timestamp', 'time:timestamp'):
            col_map[col] = 'Timestamp'
        elif cl in ('activity', 'concept:name'):
            col_map[col] = 'Activity'
    df = df.rename(columns=col_map)

    # drop the duplicate Activity columns some logs carry (e.g. HelpDesk)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[['CaseID', 'Activity', 'Timestamp']].copy()

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', utc=True)
    df = df.sort_values(['CaseID', 'Timestamp']).reset_index(drop=True)
    return df


def start_from_date(df: pd.DataFrame, start_month) -> pd.DataFrame:
    """Drop cases whose start month is before *start_month* (a 'YYYY-MM' string)."""
    if start_month is None:
        return df
    case_start_month = df.groupby('CaseID')['Timestamp'].min().dt.to_period('M')
    keep = case_start_month[case_start_month.astype(str) >= start_month].index
    return df[df['CaseID'].isin(keep)]


def end_before_date(df: pd.DataFrame, end_month) -> pd.DataFrame:
    """Drop cases whose end month is after *end_month* (a 'YYYY-MM' string)."""
    if end_month is None:
        return df
    case_end_month = df.groupby('CaseID')['Timestamp'].max().dt.to_period('M')
    keep = case_end_month[case_end_month.astype(str) <= end_month].index
    return df[df['CaseID'].isin(keep)]


def apply_step1(df: pd.DataFrame) -> pd.DataFrame:
    """Step 1: drop duplicates, optionally trim chronological outliers by date."""
    df = start_from_date(df, START_MONTH)
    df = end_before_date(df, END_MONTH)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def limited_duration(df: pd.DataFrame, max_duration_days: float,
                     end_debias: bool = True):
    """Step 6 then step 2; returns (df, latest_start), the max timestamp when end_debias is off."""
    span = df.groupby('CaseID')['Timestamp'].agg(['min', 'max'])
    dur_days = (span['max'] - span['min']).dt.total_seconds() / 86400.0

    # epsilon as in the reference implementation
    keep_long = dur_days[dur_days <= max_duration_days * 1.00000000001].index
    df = df[df['CaseID'].isin(keep_long)].reset_index(drop=True)

    max_ts = df['Timestamp'].max()
    if not end_debias:
        return df, max_ts

    latest_start = max_ts - pd.Timedelta(days=max_duration_days)
    starts = df.groupby('CaseID')['Timestamp'].min()
    keep_start = starts[starts <= latest_start].index
    df = df[df['CaseID'].isin(keep_start)].reset_index(drop=True)
    return df, latest_start


def compute_Tsep(df: pd.DataFrame, test_len: float):
    """T_sep = start time of the first test case; sorting the Series keeps the UTC tz that .values would strip."""
    case_starts = df.groupby('CaseID')['Timestamp'].min().sort_values()
    n = len(case_starts)
    first_test_case_nr = int(n * (1 - test_len))
    return case_starts.iloc[first_test_case_nr]


def strict_train_test_split(df: pd.DataFrame, T_sep):
    """train = cases with max_ts < T_sep; test = cases with max_ts >= T_sep."""
    case_max = df.groupby('CaseID')['Timestamp'].max()
    train_ids = case_max[case_max < T_sep].index
    test_ids = case_max[case_max >= T_sep].index
    return train_ids, test_ids


def carve_val(df: pd.DataFrame, train_ids, val_len: float):
    """Val = the last `val_len` of the train cases, by start time."""
    starts = (
        df[df['CaseID'].isin(train_ids)]
        .groupby('CaseID')['Timestamp'].min()
        .sort_values()
    )
    ids = list(starts.index)
    n_tr = int(len(ids) * (1 - val_len))
    return set(ids[:n_tr]), set(ids[n_tr:])


def count_train_prefix_instances(df: pd.DataFrame, train_ids) -> int:
    """#prefixes for train cases = sum over cases of (n-2) for n >= MIN_TRACE_LEN."""
    sizes = df[df['CaseID'].isin(train_ids)].groupby('CaseID').size()
    sizes = sizes[sizes >= MIN_TRACE_LEN]
    return int((sizes - 2).clip(lower=0).sum())


def choose_max_duration(df1: pd.DataFrame, end_debias: bool = True) -> float:
    """Duration cutoff maximizing the training prefixes, swept over the percentiles [95, 100]."""
    span = df1.groupby('CaseID')['Timestamp'].agg(['min', 'max'])
    dur = (span['max'] - span['min']).dt.total_seconds() / 86400.0

    pctls = np.arange(STEP6_PCTL_LO, STEP6_PCTL_HI + 1e-9, STEP6_PCTL_STEP)
    cands = sorted(set(np.percentile(dur.values, pctls)))

    best_key = None
    best_md = None
    for md in cands:
        df2, _ = limited_duration(df1, md, end_debias=end_debias)
        if df2['CaseID'].nunique() < MIN_TRAIN_CASES:
            continue
        T_sep = compute_Tsep(df2, TEST_LEN)
        train_ids, _ = strict_train_test_split(df2, T_sep)
        cnt = count_train_prefix_instances(df2, train_ids)
        key = (cnt, -md)  # maximize train count, tie -> smallest md
        if best_key is None or key > best_key:
            best_key, best_md = key, md
    return best_md


def extract_prefixes(df: pd.DataFrame, gate: str = None,
                     T_sep=None, latest_start=None) -> pd.DataFrame:
    """Prefixes k = 2..n-1 as [e1, ..., e_k, e_{k+1}]; gate='test' keeps only those whose origin e_k lies in [T_sep, latest_start] (steps 5 and 2)."""
    rows = []
    for case_id, grp in df.groupby('CaseID', sort=False):
        events = grp.reset_index(drop=True)
        n = len(events)
        if n < MIN_TRACE_LEN:
            continue
        case_end_time = events['Timestamp'].iloc[-1]
        for k in range(2, n):
            if gate == 'test':
                origin_ts = events['Timestamp'].iloc[k - 1]  # e_k, last context event
                if origin_ts < T_sep or origin_ts > latest_start:
                    continue
            prefix_events = events.iloc[:k + 1].copy()       # k+1 events
            prefix_events['CaseID'] = f"{case_id}_p{k}"
            prefix_events['CaseEndTime'] = case_end_time
            rows.append(prefix_events)

    if not rows:
        return pd.DataFrame(columns=['CaseID', 'Activity', 'Timestamp', 'CaseEndTime'])

    result = pd.concat(rows, ignore_index=True)
    result['Timestamp'] = result['Timestamp'].dt.strftime(TIME_FMT)
    result['CaseEndTime'] = result['CaseEndTime'].dt.strftime(TIME_FMT)
    return result


def _split_once(df1: pd.DataFrame, end_debias: bool):
    """Run steps 6, 2, 3, 4 and the val carve for one end_debias mode."""
    md = choose_max_duration(df1, end_debias=end_debias)
    df2, latest_start = limited_duration(df1, md, end_debias=end_debias)
    T_sep = compute_Tsep(df2, TEST_LEN)
    train_ids, test_ids = strict_train_test_split(df2, T_sep)
    train_cases, val_cases = carve_val(df2, train_ids, VAL_LEN)
    return df2, latest_start, T_sep, train_cases, val_cases, test_ids, md


def dataset_stats(df: pd.DataFrame, out_name: str, stage: str) -> dict:
    """Whole-trace statistics at one stage of the pipeline; the raw stage reproduces Table 1."""
    g = df.groupby('CaseID')['Timestamp']
    lengths = g.size()
    span = g.agg(['min', 'max'])
    gaps = df.sort_values(['CaseID', 'Timestamp']).groupby('CaseID')['Timestamp'].diff()
    gaps = gaps.dropna().dt.total_seconds()
    return {
        'dataset':   out_name,
        'stage':     stage,
        'cases':     int(df['CaseID'].nunique()),
        'events':    int(len(df)),
        'act':       int(df['Activity'].nunique()),
        'avg_len':   round(float(lengths.mean()), 1),
        'med_dt_s':  round(float(gaps.median()), 1),
        'tie_pct':   round(float((gaps == 0).mean() * 100), 1),
        'avg_dur_d': round(float((span['max'] - span['min']).dt.total_seconds().mean() / 86400.0), 1),
    }


def prepare_dataset(csv_name: str, out_name: str, stats_only: bool = False):
    """Prepare one dataset with the strict temporal split + prefix extraction."""
    df0 = load_and_standardize(csv_name)
    n_cases0 = df0['CaseID'].nunique()

    df1 = apply_step1(df0)

    end_debias = True
    df2, latest_start, T_sep, train_cases, val_cases, test_ids, md = \
        _split_once(df1, end_debias=True)

    if len(train_cases) < MIN_TRAIN_FALLBACK:
        print(f"  [Warning] {out_name}: strict split left only {len(train_cases)} "
              f"train cases, retrying without the step-2 end debias")
        end_debias = False
        df2, latest_start, T_sep, train_cases, val_cases, test_ids, md = \
            _split_once(df1, end_debias=False)

    if len(train_cases) == 0 or len(test_ids) == 0:
        print(f"  [Error] {out_name}: split collapsed "
              f"(train={len(train_cases)}, test={len(test_ids)}), nothing written")
        return None

    if stats_only:
        return [dataset_stats(df0, out_name, "raw"),
                dataset_stats(df1, out_name, "dedup"),
                dataset_stats(df2, out_name, "retained")]

    train_df = extract_prefixes(df2[df2['CaseID'].isin(train_cases)])
    val_df = extract_prefixes(df2[df2['CaseID'].isin(val_cases)])
    test_df = extract_prefixes(df2[df2['CaseID'].isin(test_ids)],
                               gate='test', T_sep=T_sep, latest_start=latest_start)
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    if len(test_df) == 0:
        print(f"  [Warning] {out_name}: test set empty after prefix gating")

    fold_name = f"fold0_variation0_{out_name}"
    train_df.to_csv(OUTPUT_DIR / f"train_{fold_name}.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / f"val_{fold_name}.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / f"test_{fold_name}.csv", index=False)
    full_train_df.to_csv(OUTPUT_DIR / f"full_train_{fold_name}.csv", index=False)

    mode = "strict (steps 1-6)" if end_debias else "strict, step-2 end debias skipped"
    print(f"  {out_name}: {n_cases0} raw cases -> {df2['CaseID'].nunique()} retained   [{mode}]")
    print(f"    step6 max_duration = {md:.2f} days   "
          f"T_sep = {pd.Timestamp(T_sep):%Y-%m-%d %H:%M}   "
          f"latest_start = {pd.Timestamp(latest_start):%Y-%m-%d %H:%M}")
    print(f"    cases:    train={len(train_cases)}  val={len(val_cases)}  "
          f"test={len(test_ids)}")
    print(f"    prefixes: train={train_df['CaseID'].nunique()}  "
          f"val={val_df['CaseID'].nunique()}  test={test_df['CaseID'].nunique()}")
    print(f"    events:   train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    if len(train_cases) < 50:
        print(f"    [Warning] very small training set ({len(train_cases)} cases); "
              f"the vocabulary is built from train, so expect UNK at test")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", action="store_true",
                    help="only report per-log statistics at the raw, dedup and "
                         "retained stages, writing paper_results/dataset_stats.csv")
    args = ap.parse_args()

    if not args.stats:
        print("Preparing UQ4PPM datasets (strict temporal split, prefix extraction)\n")
        for csv_name, out_name in sorted(DATASETS.items()):
            prepare_dataset(csv_name, out_name)
        print("\nDone!")
        return

    rows = []
    for csv_name, out_name in sorted(DATASETS.items()):
        r = prepare_dataset(csv_name, out_name, stats_only=True)
        if r:
            rows.extend(r)

    def fmt_dt(sec):
        return (f"{sec:.0f}s" if sec < 90 else
                f"{sec/60:.1f}min" if sec < 5400 else
                f"{sec/3600:.1f}h" if sec < 86400 else
                f"{sec/86400:.2f}d")

    dup = sum(a['events'] - b['events']
              for a, b in zip((r for r in rows if r['stage'] == 'raw'),
                              (r for r in rows if r['stage'] == 'dedup')))
    for stage, title in [("raw", "raw logs (the paper's Table 1)"),
                         ("dedup", f"after duplicate removal (step 1, -{dup} events)"),
                         ("retained", "after the full strict split (steps 1-6)")]:
        print(f"\n{title}")
        print(f"{'dataset':<12}{'cases':>8}{'events':>9}{'act':>6}{'avg.len':>9}"
              f"{'med.dt':>12}{'%tie':>7}{'avg.dur(d)':>12}")
        for r in (x for x in rows if x['stage'] == stage):
            print(f"{r['dataset']:<12}{r['cases']:>8}{r['events']:>9}{r['act']:>6}"
                  f"{r['avg_len']:>9.1f}{fmt_dt(r['med_dt_s']):>12}"
                  f"{r['tie_pct']:>7.1f}{r['avg_dur_d']:>12.1f}")

    out = Path(__file__).parent / "paper_results" / "dataset_stats.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out.relative_to(Path(__file__).parent)}")


if __name__ == '__main__':
    main()
