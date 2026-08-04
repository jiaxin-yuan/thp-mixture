# THP-Mixture

Reference implementation of "THP-Mixture: Generative, Uncertainty-Aware
Predictive Process Monitoring with Temporal Point Processes". The paper defines
the method, the metrics and the split protocol; this file covers running the
code. All times in days.

| `--model_type` | Paper | Time head |
|----------------|-------|-----------|
| `mixture`      | THP-M | zero-inflated LogNormal mixture, K = 5 |
| `baseline_act` | THP-B | intensity MLE + time and remaining time regression |

`baseline_act` is the code name of THP-B in the checkpoints and the artifacts.

## Install

```bash
conda env create -f environment.yml && conda activate thppm
```

Pins python 3.7.16, torch 1.13.1, pandas 1.3.5, scikit-learn 1.0.2.

## Data

Put the ten logs as CSV (`CaseID, Activity, Timestamp`, one row per event) under
`data/uq4ppm_csv/`, named `BPIC12.csv`, `BPIC13I.csv`, `BPIC15_1.csv`,
`BPIC20DD.csv`, `BPIC20ID.csv`, `BPIC20PTC.csv`, `BPIC20RFP.csv`,
`BPIC20TPD.csv`, `HelpDesk.csv`, `Sepsis.csv`.

```bash
python prepare_uq4ppm.py            # strict temporal split, prefix extraction
python prepare_uq4ppm.py --stats    # per-log statistics
```

Writes `data/{train,val,test,full_train}_fold0_variation0_UQ_<name>.csv`, one
prefix per case, `CaseEndTime` carrying the remaining time target. Activities
first seen in val or test map to one UNK type.

## Train

```bash
./run_thp.sh mixture 0        # variant, GPU id — all 10 logs
./run_thp.sh baseline_act 0
```

Both variants are needed. `run_thp.sh` applies the paper's settings (at most 300
epochs, patience 24, batch 32, lr 0.002, K = 5, gradient clipping 1.0, seed 42)
and writes `saved_models/`, which the evaluation scripts load.

## Evaluate

```bash
python evaluation/run_evaluation_uq.py --thp --all           # next event, THP-B / THP-M
python evaluation/hawkes_baseline.py --all                   # next event, H-uni / H-mk
python evaluation/run_rt_uncertainty.py --all --samples 50   # remaining time, MA / MPIW / AURG
python evaluation/collect_timings.py                         # training cost
python evaluation/measure_inference.py --thp                 # inference cost
python evaluation/make_regime_clustering_fig.py              # regime grouping figure
```

Results land in `paper_results/`. SuTraN and ED-LSTM are not vendored;
`SUTRAN_DIR=<checkout>` lets `run_evaluation_uq.py --ppm` and
`measure_inference.py --baselines` remeasure them.

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

