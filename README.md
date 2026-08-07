# THP-Mixture


Reference implementation of "THP-Mixture: Generative, Uncertainty-Aware
Predictive Process Monitoring with Temporal Point Processes". This file covers running the code. All times in days.

| `--model_type` | Paper | Time head |
|----------------|-------|-----------|
| `mixture`      | THP-M (Mixture)  | zero-inflated LogNormal mixture |
| `baseline_act` | THP-B (Baseline) | point regression |

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

Writes `data/{train,val,test,full_train}_fold0_variation0_<ds>.csv`, where `<ds>`
is `UQ_` plus the input name, such as `UQ_HelpDesk`. Each case is expanded into
prefixes `<id>_p<k>`, `CaseEndTime` carrying the remaining time target.

`--stats` reports only, leaving those CSVs unwritten. It prints one table per
stage — `raw`, the log as loaded; `dedup`, after duplicate removal; `retained`,
after case-duration trimming and end-of-dataset debiasing, the cases the models
train on — and writes the same rows to `paper_results/dataset_stats.csv`, columns
`dataset, stage, cases, events, act, avg_len, med_dt_s, tie_pct, avg_dur_d`.

The CSVs keep the original activity names. The vocabulary is built from the train
split alone, so activities first seen in val or test share one UNK id at load
time — 91 of BPIC15-1's 361 activities, one or two in BPIC13I, BPIC20ID,
BPIC20RFP, BPIC20TPD and HelpDesk, none in the rest.

## Train

```bash
./run_thp.sh mixture 0        # variant, GPU id — all 10 logs
./run_thp.sh baseline_act 0
```

Both variants are needed. `run_thp.sh` applies the paper's settings (at most 300
epochs, patience 24, batch 32, lr 0.002, K = 5, gradient clipping 1.0, seed 42)
and writes three outputs per log, none of them committed, so a fresh clone has
none until you train:

- `saved_models/fold0_variation0_<ds>_<variant>_best_model_best_{mae,mae_rt,ll}.pth`,
  one checkpoint per validation metric. The evaluation scripts and
  `main.py --test` load the `_best_mae` one; the other two are kept but unused.
- `logs_<variant>_v2_<ds>.log` in the repo root, which Table 5 reads for
  training cost.
- `results/fold0_variation0_<ds>_<variant>.txt`, one
  `TEST ll=... mae=... mae_rt=... acc=...` line, also copied to
  `results_<variant>_v2/`. Of these, only `baseline_act`'s `mae_rt` feeds a
  table, as the THP-B column of Table 4.

Pretrained checkpoints are not distributed. The CSVs and JSONs under
`paper_results/` are the record of the original runs; reproducing Tables 3 to 5
means training both variants first.

## Reproduce the paper

One subsection per table. `per_dataset/` abbreviates
`paper_results/per_dataset/<ds>/`. Table 2 and
Figures 1 to 3 need no code.

Every printed cell of Tables 1, 3, 4 and 5 is recorded in
`paper_results/paper_tables.csv`, one `table, dataset, column, value` row each,
under the keys `datasets`, `nextevent`, `rtuq` and `timing` — the reference for
what a rerun should reproduce. No script in the repo writes it; it was
assembled by hand from the artifacts below.

The SuTraN and ED-LSTM columns are not retrained here. The commands below read
them from a local clone of https://github.com/BrechtWts/SuffixTransformerNetwork,
whose location is passed in the environment variable `SUTRAN_DIR`:

```bash
export SUTRAN_DIR=/path/to/SuffixTransformerNetwork
```


### Table 1 — dataset statistics

`python prepare_uq4ppm.py --stats` writes `dataset_stats.csv`; the table quotes
its `stage=raw` rows, the logs as loaded — see the mismatch noted below.
`python evaluation/make_regime_clustering_fig.py` runs the Ward clustering on the
cluster-defining statistics (Act, Med. Δt) behind the Group A, Group B and
Outlier blocks.

Act counts distinct `concept:name`. UQ4PPM and the LA-CR paper [1] count
`concept:name` × `lifecycle:transition`, which differs on the two logs whose
lifecycle attribute varies:

| Log | `concept:name` | with `lifecycle:transition` |
|-----|---------------------------|-----------------------------|
| BPIC12  | 24 | 36 |
| BPIC13I | 4  | 13 |

The other eight logs carry one lifecycle value, so both conventions agree there,
and the models are trained on the `concept:name` vocabulary. Substituting 36
and 13 leaves the Ward k = 2 partition and the k = 3 blocks unchanged, moving
only the silhouette (0.529 to 0.436 excluding the outlier, 0.580 to 0.608 over
all ten).

### Table 3 — next event predictions

| Columns | Command | Artifact |
|---------|---------|----------|
| THP-B, THP-M | `python evaluation/run_evaluation_uq.py --thp --all` | `per_dataset/thp_results.json` |
| H-uni, H-mk | `python evaluation/hawkes_baseline.py --all` | `per_dataset/hawkes_results.json` |
| SuTraN, ED-LSTM | `python evaluation/run_evaluation_uq.py --ppm --all` | `per_dataset/ppm_results.json` |

| Paper metric | `thp_results.json`, `hawkes_results.json` | `ppm_results.json` |
|--------------|-------------------------------------------|--------------------|
| Next activity accuracy (%) | `activity_accuracy_teacher` under `methods` | `act_accuracy` |
| Next event time MAE (days) | `time_mae_teacher` under `methods` | `time_mae_days` |

H-uni models inter-event times without marks, so its
`activity_accuracy_teacher` is a placeholder 0.0, printed as the "–" of the
table.

### Table 4 — remaining time prediction

| Columns | Command | Artifact |
|---------|---------|----------|
| THP-M RT-MAE, MA, MPIW, AURG | `python evaluation/run_rt_uncertainty.py --all --samples 50` | `per_dataset/rt_uncertainty_results.json`, `mae`, `ma`, `mpiw_rms_sigma` and `aurg` under `models` |
| THP-B RT-MAE | `python main.py --fold_dataset data/fold0_variation0_<ds>.csv --test --model_type baseline_act` | `thp_baseline_testline.csv`, column `mae_rt` |
| SuTraN, ED-LSTM RT-MAE | `python evaluation/run_evaluation_uq.py --ppm --all` | `per_dataset/ppm_results.json` |
| LA-CR⋆ | LA-CR's own implementation [1], retrained under our strict temporal split, run outside this repo | `baselines/lacr_summary.csv`, columns `lacr_*`, the ⋆ over its `LA+I_*` and `LA+S_*`; the THP-M side of that file is `thp_full_*` |
| UQ⋆ | quoted from Table 3 of [1], under that paper's own split; not rerun | — |

### Table 5 — training and inference cost

| Columns | Command | Artifact |
|---------|---------|----------|
| training minutes | `python evaluation/collect_timings.py` | `time_consumption_comparison.csv` |
| inference, THP-B and THP-M | `python evaluation/measure_inference.py --thp` | `inference_timings.csv` |
| inference, SuTraN and ED-LSTM | `python evaluation/measure_inference.py --baselines` | `inference_timings.csv` |
| inference, LA-CR | timed with LA-CR's own implementation [1], run outside this repo | `baselines/lacr_inference_timing.csv` |

## Open points

Three details behind the printed numbers, recorded so each cell can be traced,
and open to tightening in a later revision.

**Decode of the Table 3 time column.** THP-M's next event time is decoded as the
median of the predictive distribution, the MAE-optimal point prediction and the
default `--decode median`, rather than the expectation of Equation 7. Both are
shipped for all ten logs: the expectation is `mae` in `thp_mixture_testline.csv`,
5.56 d on BPIC20TPD against the printed 5.15 d. Equation 8 appears there as
`mae_rt`; Table 4 reports instead the S = 50 rollout sample mean of Section 4.2,
22.56 d against 28.21 d.

**Provenance of the Table 5 training times.** THP-B, THP-M and Hawkes are summed
from their training logs. SuTraN and ED-LSTM are derived — checkpoint mtime span
on eight logs, epochs × mean rate on two — and LA-CR's figure comes from its own
run; the `*_src` columns of `time_consumption_comparison.csv` record which applies
per cell. Re-timing the three baselines would put the whole table on one footing.

**Stage quoted in Table 1.** The printed rows are `stage=raw` in
`dataset_stats.csv`, the logs as loaded; the `stage=retained` rows, 8.2% fewer
cases and 14.5% fewer events, are the ones the models train on.

## Layout

```
main.py              train / test one log
run_thp.sh           one variant over all 10 logs
prepare_uq4ppm.py    strict temporal split, prefixes, statistics
Utils.py             log-likelihood, mixture NLL, E[Δt], CE
environment.yml      the pinned conda environment
preprocess/          CSV loading, Dataset and DataLoader
transformer/         THP backbone and prediction heads
trainer/             epochs, checkpointing, early stopping
utils/               seeding
evaluation/          next event, remaining time, uncertainty, cost
paper_results/       the numbers behind the paper
```

## References

[1] Amiri Elyasi, K., van der Aa, H., Stuckenschmidt, H.: A simple and
calibrated approach for uncertainty-aware remaining time prediction. In: BPM,
pp. 217–234. Springer (2025). Code: https://github.com/keyvan-amiri/UQ4PPM

