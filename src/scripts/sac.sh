set -euo pipefail
# ---- Python interpreter resolution ----
PY="${PY:-}"

if [[ -z "$PY" ]]; then
  if command -v python >/dev/null 2>&1; then
    PY="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  elif [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
  else
    echo "ERROR: No Python interpreter found. Try: source .venv/bin/activate"
    exit 1
  fi
fi

echo "Using Python: $PY"


ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

# ------------------------
# User-tunable defaults
# ------------------------
PHASE="${PHASE:-phase3}"
SEED0="${SEED0:-42}"
TESTING_SEEDS="${TESTING_SEEDS:-30}"
ACTION_REPEAT="${ACTION_REPEAT:-6}"

TRAIN_EPISODES="${TRAIN_EPISODES:-365}"
TRAIN_NET="${TRAIN_NET:-porto20_s600}"     # network id (expects configs/network_${TRAIN_NET}.yaml)
TRAIN_HOURS="${TRAIN_HOURS:-168}"          # training episode horizon (hours); recommended 168 for your weather/events story

# N=20 networks to evaluate (space-separated)
NETS_20="${NETS_20:-porto20_s600 porto20_s900 lisbon20_s700}"

# Evaluate on both horizons to support "1 day vs 7 days" comparison
HORIZONS=(24 168)

# Scenarios
SCENARIOS=(baseline hotspot_od hetero_lambda event_heavy)

# Output structure
OUT_ROOT="out/paper/${PHASE}"

TRAIN_OUT_DIR="${OUT_ROOT}/train/${TRAIN_NET}"
EVAL_OUT_DIR="${OUT_ROOT}/eval_sac20"   # will further split by net/hours/scenario

mkdir -p "$TRAIN_OUT_DIR" "$EVAL_OUT_DIR"

TRAIN_CONFIG="configs/network_${TRAIN_NET}.yaml"
if [[ ! -f "$TRAIN_CONFIG" ]]; then
  echo "ERROR: training config not found: $TRAIN_CONFIG"
  exit 1
fi

# Verify eval configs exist
for net in $NETS_20; do
  cfg="configs/network_${net}.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "ERROR: eval config not found: $cfg"
    exit 1
  fi
done

echo "== SAC: Train on N=20 baseline, reuse for all N=20 evals =="
echo "PHASE:          $PHASE"
echo "SEED0:           $SEED0"
echo "TESTING_SEEDS:   $TESTING_SEEDS"
echo "ACTION_REPEAT:   $ACTION_REPEAT"
echo
echo "TRAIN_NET:       $TRAIN_NET"
echo "TRAIN_CONFIG:    $TRAIN_CONFIG"
echo "TRAIN_HOURS:     $TRAIN_HOURS"
echo "TRAIN_EPISODES:  $TRAIN_EPISODES"
echo
echo "EVAL_NETS_20:     $NETS_20"
echo "HORIZONS:        ${HORIZONS[*]}"
echo "SCENARIOS:       ${SCENARIOS[*]}"
echo

# ------------------------
# 1) Train once (baseline)
# ------------------------
TRAIN_JSON="${TRAIN_OUT_DIR}/sac_train_baseline_${TRAIN_HOURS}h.json"
TRAIN_LOG="${TRAIN_OUT_DIR}/sac_train_baseline_${TRAIN_HOURS}h.log"

if [[ -f "$TRAIN_JSON" ]]; then
  echo "Training output already exists (will not retrain): $TRAIN_JSON"
  # Try to recover model path from JSON (fallback to log if present)
  MODEL_PATH_FROM_JSON="$(python - <<'PY'
import json, sys
p = sys.argv[1]
with open(p, "r", encoding="utf-8") as f:
    obj = json.load(f)
mp = obj.get("model_path", "")
print(mp)
PY
"$TRAIN_JSON")"
  MODEL_PATH="$MODEL_PATH_FROM_JSON"
else
  echo "== Training SAC (baseline) =="
  "$PY" -m ml.sac \
    --config "$TRAIN_CONFIG" \
    --hours "$TRAIN_HOURS" \
    --seed0 "$SEED0" \
    --training_episodes "$TRAIN_EPISODES" \
    --testing_seeds 0 \
    --action_repeat "$ACTION_REPEAT" \
    --scenario baseline \
    --out "$TRAIN_JSON" | tee "$TRAIN_LOG"

  # Extract saved model path from stdout log
  MODEL_PATH="$(grep -E "Model saved to:" "$TRAIN_LOG" | tail -n 1 | sed -E 's/.*Model saved to:\s*//')"
fi

# Normalize model path to a .zip file if needed
if [[ -n "${MODEL_PATH:-}" ]]; then
  if [[ -f "$MODEL_PATH" ]]; then
    true
  elif [[ -f "${MODEL_PATH}.zip" ]]; then
    MODEL_PATH="${MODEL_PATH}.zip"
  fi
fi

if [[ -z "${MODEL_PATH:-}" || ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: could not determine a valid model path after training."
  echo "Look in: $TRAIN_LOG"
  echo "Or set MODEL_PATH manually by exporting it before running."
  exit 1
fi

echo
echo "Using trained model: $MODEL_PATH"
echo

# ------------------------
# 2) Evaluate reuse model
# ------------------------
echo "== Evaluating trained N=20 model across networks/scenarios/horizons =="

for net in $NETS_20; do
  CFG="configs/network_${net}.yaml"

  for hours in "${HORIZONS[@]}"; do
    OUT_DIR="${EVAL_OUT_DIR}/${net}/${hours}h"
    mkdir -p "$OUT_DIR"

    for scenario in "${SCENARIOS[@]}"; do
      OUT_JSON="${OUT_DIR}/${scenario}.json"

      if [[ -f "$OUT_JSON" ]]; then
        echo "OK (exists): $OUT_JSON"
        continue
      fi

      echo "RUN: net=$net hours=${hours} scenario=$scenario"
      python -m ml.sac \
        --config "$CFG" \
        --hours "$hours" \
        --seed0 "$SEED0" \
        --testing_seeds "$TESTING_SEEDS" \
        --action_repeat "$ACTION_REPEAT" \
        --scenario "$scenario" \
        --load_model "$MODEL_PATH" \
        --out "$OUT_JSON"
    done
  done
done

echo
echo "Done."
echo "Train outputs: $TRAIN_OUT_DIR"
echo "Eval outputs:  $EVAL_OUT_DIR"
