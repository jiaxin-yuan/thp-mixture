# THP-Mixture

**Work in Progress:** This repository is currently being cleaned up and documented to accompany the submission of our paper. The complete and reproducible code will be finalized soon.

Reference implementation of "THP-Mixture: Generative, Uncertainty-Aware
Predictive Process Monitoring with Temporal Point Processes". This file covers running the code. All times in days.

| `--model_type` | Paper |
|----------------|-------|
| `mixture`      | THP-M | 
| `baseline_act` | THP-B |

`baseline_act` is the code name of THP-B in the checkpoints and the artifacts.

## Install

```bash
conda env create -f environment.yml && conda activate thppm
```

Pins python 3.7.16, torch 1.13.1, pandas 1.3.5, scikit-learn 1.0.2.

## Data

The pipeline reads CSV only, never XES. Put the ten logs as CSV (`CaseID,
Activity, Timestamp`, one row per event) under `data/uq4ppm_csv/`, named
`BPIC12.csv`, `BPIC13I.csv`, `BPIC15_1.csv`, `BPIC20DD.csv`, `BPIC20ID.csv`,
`BPIC20PTC.csv`, `BPIC20RFP.csv`, `BPIC20TPD.csv`, `HelpDesk.csv`,
`Sepsis.csv`. The corresponding original logs are XES, kept under `data/uq4ppm/`; convert
them once with pm4py, mapping `case:concept:name`, `concept:name` and
`time:timestamp` to the three columns.

```bash
python prepare_uq4ppm.py            # strict temporal split, prefix extraction
python prepare_uq4ppm.py --stats    # per-log statistics
```

Writes `data/{train,val,test,full_train}_fold0_variation0_UQ_<name>.csv`, each
case expanded into prefixes `<id>_p<k>`, `CaseEndTime` carrying the remaining
time target.

`--stats` writes no splits. It prints one table per stage and writes
`paper_results/dataset_stats.csv`, columns `dataset, stage, cases, events, act,
avg_len, med_dt_s, tie_pct, avg_dur_d`: `raw` reproduces Table 1, `dedup` is
after duplicate removal, `retained` after case-duration trimming and
end-of-dataset debiasing, the cases the models train on.

The vocabulary is built from the train split alone, so activities first seen in
val or test map to one shared UNK type. Six logs need it: BPIC15-1 by far the
most (train sees 259 of 361 activities, 91 more appear in test), HelpDesk
(`DUPLICATE`, `Require upgrade`), and BPIC13I, BPIC20ID, BPIC20RFP, BPIC20TPD
with one or two each. BPIC12, BPIC20DD, BPIC20PTC and Sepsis have none.

## Train

```bash
./run_thp.sh mixture 0        # variant, GPU id — all 10 logs
./run_thp.sh baseline_act 0
```

Both variants are needed. `run_thp.sh` applies the paper's settings (at most 300
epochs, patience 24, batch 32, lr 0.002, K = 5, gradient clipping 1.0, seed 42)
and writes three outputs, all gitignored:

- `saved_models/<fold>_<variant>_best_model_best_{mae,mae_rt,ll}.pth`, one
  checkpoint per validation metric. The evaluation scripts and `main.py --test`
  load the `_best_mae` one; the other two are kept but unused.
- `logs_<variant>_v2_<log>.log` in the repo root, which Table 5 reads for
  training cost.
- `results/fold0_variation0_<log>_<variant>.txt`, the testline of Table 4,
  copied to `results_<variant>_v2/` as each log finishes.

## Reproduce the paper

One subsection per table. `<ds>` is a dataset name such as `UQ_BPIC20TPD`, and
`per_dataset/` abbreviates `paper_results/per_dataset/<ds>/`. Table 2 and
Figures 1 to 3 need no code.

### Table 1 — dataset statistics

`python prepare_uq4ppm.py --stats` writes `dataset_stats.csv`, of which the
table quotes the `stage=raw` rows; `dedup` and `retained` record the two
trimming steps. `python evaluation/make_regime_clustering_fig.py` runs the Ward
clustering on (Act, Med. Δt) behind the Group A / Group B / outlier blocks.

### Table 3 — next event predictions

| Columns | Command | Artifact |
|---------|---------|----------|
| THP-B, THP-M | `python evaluation/run_evaluation_uq.py --thp --all` | `per_dataset/thp_results.json` |
| H-uni, H-mk | `python evaluation/hawkes_baseline.py --all` | `per_dataset/hawkes_results.json` |
| SuTraN, ED-LSTM | `SUTRAN_DIR=<checkout> python evaluation/run_evaluation_uq.py --ppm --all` | `per_dataset/ppm_results.json` |

Each JSON gives `activity_accuracy_teacher` and `time_mae_teacher`. H-uni models
no activity, hence the dash.

### Table 4 — remaining time prediction

| Columns | Command | Artifact |
|---------|---------|----------|
| THP-M RT-MAE, MA, MPIW, AURG | `python evaluation/run_rt_uncertainty.py --all --samples 50` | `per_dataset/rt_uncertainty_results.json` |
| THP-B RT-MAE | `./run_thp.sh baseline_act 0`, the `--test` line | `results/fold0_variation0_<ds>_baseline_act.txt` |
| SuTraN, ED-LSTM RT-MAE | `SUTRAN_DIR=<checkout> python evaluation/run_evaluation_uq.py --ppm --all` | `per_dataset/ppm_results.json` |
| LA-CR⋆ | not vendored, shipped as measured | `baselines/lacr_summary.csv`, columns `lacr_*` |
| UQ⋆ | published values, no code | `uq4ppm_rt_mae_reference.csv` |

The four THP-M columns are `mae`, `ma`, `mpiw_rms_sigma` and `aurg`, all from
the same S = 50 rollouts. `lacr_*` is the per-metric better of LA+I and LA+S,
the ⋆ of the caption. The THP-B column is `mae_rt`, shipped for all ten logs in
`thp_baseline_testline.csv`.

### Table 5 — training and inference cost

| Columns | Command | Artifact |
|---------|---------|----------|
| training minutes | `python evaluation/collect_timings.py` | `time_consumption_comparison.csv` |
| inference, THP-B and THP-M | `python evaluation/measure_inference.py --thp` | `inference_timings.csv` |
| inference, SuTraN and ED-LSTM | `SUTRAN_DIR=<checkout> python evaluation/measure_inference.py --baselines` | `inference_timings.csv`, appended |
| inference, LA-CR | not vendored, shipped as measured | `baselines/lacr_inference_timing.csv` |

`collect_timings.py` sums the `t=<x>min` lines of `logs_*_v2_<ds>.log` in the
repo root and reads SuTraN and ED-LSTM from `baselines/timings_sutran.csv`.
`hawkes_baseline.py` writes its logs under `evaluation/`, so copy them up first
or the Hawkes column comes out empty:

```bash
cp evaluation/logs_hawkes_v2_*.log .
python evaluation/collect_timings.py
```

The MC rollout of Section 4.2 is excluded from the table.

## Estimators

- Next event time (THP-M): the median of the predictive density, the MAE-optimal
  point estimate, from the default `--decode median`.
- Remaining time (THP-M): the mean of the S = 50 rollouts, the same samples that
  give MA, MPIW and AURG.
- The closed-form E[Δt] of Equation 7 and its teacher-forced accumulation of
  Equation 8 are what `main.py --test` prints; both are shipped for all ten logs
  in `paper_results/thp_mixture_testline.csv`, alongside
  `thp_baseline_testline.csv` for THP-B, whose remaining time comes from its
  regression head.
- Of the two testlines only THP-B feeds a table. THP-M is reported from the
  median decode (Table 3) and the rollout mean (Table 4), not from the closed
  form: on BPIC20TPD, 5.15 d and 22.56 d against the testline's 5.56 d and
  28.21 d.

## Layout

```
main.py              train / test one log
run_thp.sh           one variant over all 10 logs
prepare_uq4ppm.py    strict temporal split, prefixes, statistics
Utils.py             log-likelihood, mixture NLL, E[Δt], CE
preprocess/          CSV loading, Dataset and DataLoader
transformer/         THP backbone and prediction heads
trainer/             epochs, checkpointing, early stopping
utils/               seeding
evaluation/          next event, remaining time, uncertainty, cost
paper_results/       the numbers behind the paper
```

