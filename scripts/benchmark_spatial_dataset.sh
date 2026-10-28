#!/usr/bin/env bash
set -uo pipefail

# Run the same 60-question tool-routing benchmark on every LAZ file without
# loading CSV annotations. Usage:
#   bash scripts/benchmark_spatial_dataset.sh INPUT_DIR OUTPUT_DIR [MODEL ...]

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 INPUT_DIR OUTPUT_DIR [MODEL ...]" >&2
  exit 2
fi

INPUT_DIR=$1
OUTPUT_DIR=$2
shift 2
MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=("llama3.1")
fi

QUESTIONS_FILE="benchmark/domande_per_scene.txt"
EXPECTED_QUESTIONS=60

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory not found: $INPUT_DIR" >&2
  exit 2
fi
if [[ ! -f "$QUESTIONS_FILE" ]]; then
  echo "Questions file not found: $QUESTIONS_FILE" >&2
  exit 2
fi

QUESTION_COUNT=$(grep -cE '\?$' "$QUESTIONS_FILE")
echo "Loaded questions: $QUESTION_COUNT"
if [[ "$QUESTION_COUNT" -ne "$EXPECTED_QUESTIONS" ]]; then
  echo "Expected $EXPECTED_QUESTIONS questions, loaded $QUESTION_COUNT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
STATUS_FILE="$OUTPUT_DIR/batch_status.tsv"
printf 'scene\tstatus\n' > "$STATUS_FILE"

mapfile -d '' SCENES < <(find "$INPUT_DIR" -maxdepth 1 -type f \( -iname '*.laz' -o -iname '*.las' \) -print0 | sort -z)
if [[ ${#SCENES[@]} -eq 0 ]]; then
  echo "No LAZ/LAS files found in: $INPUT_DIR" >&2
  exit 1
fi

echo "Scenes found: ${#SCENES[@]}"
echo "Models: ${MODELS[*]}"
echo "Condition: graph (CSV annotations withheld)"

FAILURES=0
for SCENE in "${SCENES[@]}"; do
  SCENE_NAME=$(basename "${SCENE%.*}")
  SCENE_OUTPUT="$OUTPUT_DIR/$SCENE_NAME"
  mkdir -p "$SCENE_OUTPUT"
  echo
  echo "=== $SCENE_NAME ==="

  if pixi run python benchmark.py "$SCENE" \
      --models "${MODELS[@]}" \
      --condition graph \
      --no-reference \
      --think true \
      --distance-threshold 2.0 \
      --questions-file "$QUESTIONS_FILE" \
      --output-dir "$SCENE_OUTPUT" \
      2>&1 | tee "$SCENE_OUTPUT/benchmark.log"; then
    printf '%s\tok\n' "$SCENE_NAME" >> "$STATUS_FILE"
  else
    printf '%s\tfailed\n' "$SCENE_NAME" >> "$STATUS_FILE"
    FAILURES=$((FAILURES + 1))
  fi
done

echo
echo "Completed scenes: ${#SCENES[@]}"
echo "Failures: $FAILURES"
echo "Status file: $STATUS_FILE"
exit "$FAILURES"
