# THP-Mixture

Reference implementation of "THP-Mixture: Generative, Uncertainty-Aware
Predictive Process Monitoring with Temporal Point Processes".

A Transformer Hawkes backbone over a case prefix, with a zero-inflated LogNormal
mixture time head and a categorical next activity head, so that equal timestamps
carry probability mass. Two model variants appear in the paper:

| `--model_type` | Paper | Time head | Trained with |
|----------------|-------|-----------|--------------|
| `mixture`      | THP-M | zero-inflated LogNormal mixture, K = 5 | mixture NLL + CE |
| `baseline_act` | THP-B | two point regression heads (Δt, remaining time) | intensity MLE + MAE + CE |

`baseline` is `baseline_act` without the activity head; it is not reported in the
paper. All times are in days.

## Install

```bash
conda env create -f environment.yml
conda activate thppm
```

Pins python 3.7.16, torch 1.13.1, pandas 1.3.5, scikit-learn 1.0.2 and
tensorboard 2.11.2.

## Data

Put the ten public event logs as CSV (`CaseID, Activity, Timestamp`, one row per
event) under `data/uq4ppm_csv/`, named `BPIC12.csv`, `BPIC13I.csv`,
`BPIC15_1.csv`, `BPIC20DD.csv`, `BPIC20ID.csv`, `BPIC20PTC.csv`,
`BPIC20RFP.csv`, `BPIC20TPD.csv`, `HelpDesk.csv`, `Sepsis.csv`. Then:

```bash
python prepare_uq4ppm.py            # strict temporal split + prefix extraction
python prepare_uq4ppm.py --stats    # Table 1 statistics of each log
```

The strict temporal protocol (no training case completes after a test case
starts, dataset end debiased, longest 5% of cases trimmed) is implemented in six
steps documented at the top of `prepare_uq4ppm.py`. It writes
`data/{train,val,test,full_train}_fold0_variation0_UQ_<name>.csv`, where a case
is one prefix, `CaseEndTime` carries the true case end for the remaining time
target, and `UQ4PPM_RAW` overrides the input directory.

`main.py` reads three CSVs per run, resolved from
`--fold_dataset <dir>/<stem>.csv` as `<dir>/{train,val,test}_<stem>.csv`. The
activity vocabulary is built from the train split only; activities first seen in
val or test map to a single UNK type.

## Train and test

```bash
./run_thp.sh mixture 0        # variant, GPU id — all 10 logs
./run_thp.sh baseline_act 0
```

Defaults, each overridable by an environment variable of the same name:
`EPOCHS=300 PATIENCE=24 BATCH_SIZE=32 LR=0.002 N_MIX=5 GRAD_CLIP=1.0`;
`DATASETS_OVERRIDE` restricts the dataset list. Per dataset the script writes
`logs_<variant>_v2_<dataset>.log` and copies the result file to
`results_<variant>_v2/`. A single run:

```bash
python main.py --fold_dataset data/fold0_variation0_UQ_Sepsis.csv \
               --model_type mixture --train --test \
               --epoch 300 --patience 24 --batch_size 32 --lr 0.002 --device cuda
```

`--train` and `--test` are independent; `--test` alone loads
`saved_models/<stem>_<model_type>_best_model_best_mae.pth`, or `--model_path`.
Outputs: `results/<stem>_<model_type>.txt` (test line), `results/raw_<stem>.txt`
(per-epoch validation), `saved_models/*_best_{mae,mae_rt,ll}.pth`, and
TensorBoard runs under `runs/` (`tensorboard --logdir=runs`).

The test line of a training run reports next event MAE decoded as E[Δt] and
remaining time as the cumulative sum of one-step expectations. The paper decodes
THP-M differently (conditional median for MAE, Monte Carlo rollout for the
remaining time), so its THP-M time columns come from the evaluation scripts
below, not from this line.

## Reproducing the paper's tables

| Table | Command | Artifact |
|-------|---------|----------|
| 1 dataset statistics | `python prepare_uq4ppm.py --stats` | `paper_results/dataset_stats.csv` |
| 1 regime grouping | `python evaluation/make_regime_clustering_fig.py` | `paper_results/fig_regime_clustering.pdf` |
| 3 next event, THP-B / THP-M | `python evaluation/run_evaluation_uq.py --thp --all` | `paper_results/per_dataset/<ds>/thp_results.json` |
| 3 next event, SuTraN / ED-LSTM | `SUTRAN_DIR=<checkout> python evaluation/run_evaluation_uq.py --ppm --all` | `paper_results/per_dataset/<ds>/ppm_results.json` |
| 3 next event, H-uni / H-mk | `python evaluation/hawkes_baseline.py` | `paper_results/per_dataset/<ds>/hawkes_results.json` |
| 4 remaining time, THP-B | training run, `run_thp.sh baseline_act` | `paper_results/thp_baseline_testline.csv` |
| 4 remaining time, THP-M and its MA / MPIW / AURG | `python evaluation/run_rt_uncertainty.py --all --samples 50` | `paper_results/per_dataset/<ds>/rt_uncertainty_results.json` |
| 5 training cost | `python evaluation/collect_timings.py` | `paper_results/time_consumption_comparison.csv` |
| 5 inference cost | `python evaluation/measure_inference.py --thp` | `paper_results/inference_timings.csv` |
| 3, 4, 5 assembled | `python evaluation/make_tables.py --check` | `paper_results/{nextevent,remaining_time,cost}_table.csv` |

`make_tables.py` prints Tables 3, 4 and 5 from the stored artifacts, and with
`--check` compares every cell, column by column, against
`paper_results/paper_tables.csv`, the frozen record of the 430 numbers printed in
Tables 1 and 3 to 5. It recomputes 390 of them from the artifacts and reports no
mismatch; the remaining 40 are listed by name as having no source in this
repository, and are the UQ$\star$ calibration columns (MA, MPIW, AURG, quoted
from the UQ4PPM paper, which reports no per-metric breakdown we could recompute)
and the LA-CR training minutes (measured in the upstream checkout).

The uncertainty metrics MA, MPIW and AURG follow the definitions of Amiri Elyasi
et al. (github.com/keyvan-amiri/UQ4PPM); `evaluation/uq_metrics.py` documents each
one and the port. `S = 50` Monte Carlo continuations are drawn per prefix, over
the oracle remaining length, as the model has no end-of-case symbol.

Baselines that are not ours are not vendored here. Their numbers are shipped
(`paper_results/per_dataset/<ds>/ppm_results.json`,
`paper_results/baselines/`), and the two scripts that measure them take a
checkout of the upstream repository through `SUTRAN_DIR`. `UQ*` is the per-metric
best over the eight methods reported by Amiri Elyasi et al. under their own
split, inlined in `evaluation/make_tables.py`; `LA-CR*` is the per-metric best of
their two calibration variants retrained on our split, in
`paper_results/baselines/lacr_*.csv`.

## What is shipped

```
paper_results/
  paper_tables.csv           every number printed in Tables 1 and 3-5
  per_dataset/<ds>/*.json    per-method metrics behind Tables 3 and 4
  thp_baseline_testline.csv  THP-B test line per log, source of its RT-MAE
  *.csv                      assembled tables and timing measurements
  baselines/                 SuTraN / ED-LSTM and LA-CR timing and summary files
  fig_regime_clustering.*    the Table 1 grouping
```

`paper_tables.csv` is long-format (`table, dataset, column, value`) and carries
no LaTeX: the typeset sources live with the paper, not with the code.

Every number in the paper comes from `paper_results/`. The per-run output
directories the scripts write (`results/`, `results_{mixture,baseline_act}_v2/`,
`results_hawkes{,_marked}_v2/`) are not shipped; rerunning a command recreates
them. `thp_baseline_testline.csv` is the one exception that had to be lifted out
of them: the THP-B remaining time is reported by the training run and by no
evaluation script, so its test line is archived here.

Event logs, checkpoints and TensorBoard runs are not shipped; regenerate them
with the commands above.

## Reproducibility

- `--seed` (default 42) seeds `random`, NumPy and torch and puts cuDNN in
  deterministic mode. Two runs of the same command on the same GPU and library
  versions give identical metrics; different GPU models or torch versions can
  shift the last digits.
- The reported numbers come from `run_thp.sh` at the defaults above, on the
  fold-0 strict temporal split, one run per log and variant. No seed variance is
  reported.
- Model selection uses the best validation MAE checkpoint. Two further
  checkpoints (best remaining time MAE, best log-likelihood) are kept but unused.
- Table 1 counts the logs as published. Step 1 then removes 712 fully duplicated
  events over the ten logs, and steps 2 and 6 drop further cases before the
  split; `prepare_uq4ppm.py --stats` prints all three stages.
- Training and inference were measured on one NVIDIA RTX A6000; the Hawkes
  baselines are fitted on CPU.

## Layout

```
main.py                     entry point, train / test
run_thp.sh                  all 10 logs for one variant
prepare_uq4ppm.py           strict temporal split, prefix extraction, Table 1 stats
preprocess/dataset.py       CSV loading, features, Dataset and DataLoader
transformer/                backbone (Layers.py) and heads (model.py)
trainer/train.py            epochs, checkpointing, early stopping
Utils.py                    intensity log-likelihood, mixture NLL, E[Δt], CE
utils/reproducibility.py    set_all_seeds
evaluation/                 the scripts behind Tables 1 and 3-5
paper_results/              the numbers behind the paper's tables
```
