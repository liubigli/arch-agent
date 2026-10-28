#!/usr/bin/env bash

set -u

INPUT_DIR="${1:-/mnt/c/Users/Utente/Desktop/Lucrezia/Lu_test_project/laz_archdataset_segmented}"
OUTPUT_BASE="${2:-/mnt/c/Users/Utente/Desktop/Lucrezia/Lu_test_project/triple_validation_outputs}"
TODAY="$(date +%Y%m%d)"

TEST_N=1
while [[ -d "$OUTPUT_BASE/dataset_L1_${TODAY}_test_${TEST_N}" ]]; do
  TEST_N=$((TEST_N + 1))
done

RUN_DIR="$OUTPUT_BASE/dataset_L1_${TODAY}_test_${TEST_N}"
mkdir -p "$RUN_DIR"

mapfile -d '' SCENES < <(find "$INPUT_DIR" -maxdepth 1 -type f -iname '*.laz' -print0 | sort -z)
EXPECTED=${#SCENES[@]}
SUCCESS=0
FAILED=0

echo "Dataset: $INPUT_DIR"
echo "LAZ scenes found: $EXPECTED"
echo "Output: $RUN_DIR"
echo

for SCENE in "${SCENES[@]}"; do
  NAME="$(basename "${SCENE%.*}")"
  SCENE_OUT="$RUN_DIR/$NAME"
  mkdir -p "$SCENE_OUT"

  echo "============================================================"
  echo "Scene: $NAME"
  echo "============================================================"

  if pixi run python validate_triples.py \
    "$SCENE" \
    --graph spatial \
    --skip-annotations \
    --eps 0.5 \
    --min-samples 15 \
    --distance-threshold 2.0 \
    --sample-n 150000 \
    --output-dir "$SCENE_OUT" \
    2>&1 | tee "$SCENE_OUT/run.log"; then
    SUCCESS=$((SUCCESS + 1))
    echo "SUCCESS: $NAME"
  else
    FAILED=$((FAILED + 1))
    echo "FAILED: $NAME" | tee -a "$RUN_DIR/failures.txt"
  fi
  echo
done

pixi run python summarize_dataset_triples.py "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/dataset_summary.log"

{
  echo "Dataset L1 batch run"
  echo "Date: $TODAY"
  echo "Input scenes: $EXPECTED"
  echo "Successful: $SUCCESS"
  echo "Failed: $FAILED"
  echo "Output: $RUN_DIR"
} | tee "$RUN_DIR/batch_status.txt"

echo
echo "Completed. Results:"
echo "$RUN_DIR"
