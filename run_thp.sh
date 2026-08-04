#!/bin/bash
# Train and test THP_for_PPM on the 10 UQ4PPM datasets, one variant per call:
#   ./run_thp.sh baseline_act [GPU]  THP-B
#                                    
#   ./run_thp.sh mixture      [GPU]  THP-M
#            
MODEL="${1:-}"                    # baseline_act | mixture, required
GPU="${2:-0}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-24}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.002}"
N_MIX="${N_MIX:-5}"

if [[ "$MODEL" != "baseline_act" && "$MODEL" != "mixture" ]]; then
  echo "usage: $0 {baseline_act|mixture} [gpu]"; exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU"
RESULTS_DIR="results_${MODEL}_v2"
mkdir -p "$RESULTS_DIR"

EXTRA=()
[[ "$MODEL" == "mixture" ]] && EXTRA=(--n_mix "$N_MIX")

DATASETS=(
  UQ_HelpDesk UQ_Sepsis UQ_BPIC13I UQ_BPIC20DD UQ_BPIC20RFP
  UQ_BPIC20PTC UQ_BPIC20ID UQ_BPIC20TPD UQ_BPIC12 UQ_BPIC15_1
)
# Space-separated subset, e.g. DATASETS_OVERRIDE="UQ_BPIC12 UQ_BPIC15_1"
[[ -n "${DATASETS_OVERRIDE:-}" ]] && read -ra DATASETS <<< "$DATASETS_OVERRIDE"

echo "=========================================="
echo "THP_for_PPM [$MODEL] on ${#DATASETS[@]} UQ4PPM datasets"
echo "GPU=$GPU  epochs<=$EPOCHS  patience=$PATIENCE  lr=$LR  bs=$BATCH_SIZE"
echo "=========================================="

DONE=0; FAIL=0
for ds in "${DATASETS[@]}"; do
  echo ""; echo "[START] $ds ($(date))"
  GRAD_CLIP="${GRAD_CLIP:-1.0}" python main.py \
    --fold_dataset "data/fold0_variation0_${ds}.csv" \
    --train --test \
    --model_type "$MODEL" "${EXTRA[@]}" \
    --epoch "$EPOCHS" --patience "$PATIENCE" \
    --batch_size "$BATCH_SIZE" --lr "$LR" \
    --device cuda \
    2>&1 | tee "logs_${MODEL}_v2_${ds}.log"

  res="results/fold0_variation0_${ds}_${MODEL}.txt"
  [[ -f "$res" ]] && cp "$res" "$RESULTS_DIR/"
  if [[ -f "$res" ]]; then echo "[OK] $ds ($(date))"; DONE=$((DONE+1));
  else echo "[FAIL] $ds ($(date))"; FAIL=$((FAIL+1)); fi
done

echo ""; echo "=========================================="
echo "FINISHED [$MODEL]: $DONE done, $FAIL failed / ${#DATASETS[@]}"
echo "=========================================="
for ds in "${DATASETS[@]}"; do
  res="$RESULTS_DIR/fold0_variation0_${ds}_${MODEL}.txt"
  [[ -f "$res" ]] && echo "$ds: $(tail -1 "$res")"
done
