#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-phaseA}"  # phaseA or phaseB

# pick a python executable robustly
PYBIN="${PYBIN:-}"
if [[ -z "$PYBIN" ]]; then
  if [[ -x ".venv/bin/python" ]]; then PYBIN=".venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then PYBIN="python3"
  else PYBIN="python"
  fi
fi

if [[ "$PHASE" == "phaseA" ]]; then
  HOURS=24
  SEEDS=10
  EPISODES=10
  OUTBASE="out/paper/phaseA_24h"
  SAC_TRAIN_HOURS=24
  SAC_TRAIN_EPISODES=365
else
  HOURS=168
  SEEDS=30
  EPISODES=30
  OUTBASE="out/paper/phaseB_168h"
  SAC_TRAIN_HOURS=168
  SAC_TRAIN_EPISODES=365
fi

SEED0=42
SCENARIOS=("baseline" "hotspot_od" "hetero_lambda" "event_heavy")

# Networks
declare -A NETS
NETS["porto10"]="configs/network_porto10.yaml"
NETS["porto20_s600"]="configs/network_porto20_s600.yaml"
NETS["lisbon20_s700"]="configs/network_lisbon20_s700.yaml"
NETS["porto20_s900"]="configs/network_porto20_s900.yaml"

# Bandit settings
BM=30
ALPHA=1.0
KM_BUDGETS=(0 10 20 40)
CHARGE_FRACS=(0.0 0.5 1.0)

mkdir -p "$OUTBASE"
mkdir -p "$OUTBASE/models"

SAC_TRAIN_NET="${SAC_TRAIN_NET:-porto20_s600}"
SAC_TRAIN_CFG="${NETS[$SAC_TRAIN_NET]}"

SAC_MODEL_PATH="$OUTBASE/models/sac_${SAC_TRAIN_NET}_${SAC_TRAIN_HOURS}h_seed${SEED0}.zip"
SAC_TRAIN_JSON="$OUTBASE/$SAC_TRAIN_NET/sac/baseline_train.json"

is_valid_zip () {
  "$PYBIN" - <<'PY'
import sys, zipfile
p = sys.argv[1]
try:
    with zipfile.ZipFile(p, 'r') as z:
        bad = z.testzip()
    sys.exit(0 if bad is None else 2)
except Exception:
    sys.exit(1)
PY
}

echo "== SAC: Train once per phase, then evaluate all nets/scenarios =="
echo "PHASE:        $PHASE"
echo "TRAIN_NET:    $SAC_TRAIN_NET"
echo "TRAIN_CONFIG: $SAC_TRAIN_CFG"
echo "MODEL_OUT:    $SAC_MODEL_PATH"
echo "TRAIN_JSON:   $SAC_TRAIN_JSON"
echo

if [[ ! -f "$SAC_MODEL_PATH" ]]; then
  echo "== Training SAC (baseline) =="
  mkdir -p "$(dirname "$SAC_TRAIN_JSON")"

  # Train SAC; IMPORTANT: --out is JSON, and --save_model triggers SB3 zip saving
  "$PYBIN" -m ml.sac \
    --config "$SAC_TRAIN_CFG" \
    --hours "$SAC_TRAIN_HOURS" \
    --scenario baseline \
    --seed0 "$SEED0" \
    --training_episodes "$SAC_TRAIN_EPISODES" \
    --testing_seeds "$SEEDS" \
    --save_model \
    --out "$SAC_TRAIN_JSON"

  # Pick the newest REAL zip from src/models/
  SAC_LATEST_MODEL="$(ls -t models/*.zip 2>/dev/null | head -n 1 || true)"
  if [[ -z "$SAC_LATEST_MODEL" ]]; then
    echo "ERROR: no *.zip model found under src/models/ after training." >&2
    exit 1
  fi

  # Validate it before copying
  if ! is_valid_zip "$SAC_LATEST_MODEL"; then
    echo "ERROR: newest model file is not a valid zip: $SAC_LATEST_MODEL" >&2
    echo "Tip: delete corrupt files in src/models/ and rerun training." >&2
    exit 1
  fi

  cp -f "$SAC_LATEST_MODEL" "$SAC_MODEL_PATH"

  # Validate the copied file too (catches copy/path issues)
  if ! is_valid_zip "$SAC_MODEL_PATH"; then
    echo "ERROR: copied model is not a valid zip: $SAC_MODEL_PATH" >&2
    exit 1
  fi

  echo "Copied trained SAC model:"
  echo "  from: $SAC_LATEST_MODEL"
  echo "  to:   $SAC_MODEL_PATH"
  echo
else
  echo "== SAC model already exists; skipping training =="
  echo
fi

# Main evaluation loops
for NET in "${!NETS[@]}"; do
  CFG="${NETS[$NET]}"

  for SCEN in "${SCENARIOS[@]}"; do
    echo "=== [$PHASE] NET=$NET SCEN=$SCEN ==="

    # Baselines (default pairs)
    "$PYBIN" -m scripts.eval_policy \
      --config "$CFG" --hours "$HOURS" --seeds "$SEEDS" --seed0 "$SEED0" \
      --scenario "$SCEN" \
      --out "$OUTBASE/$NET/baselines/$SCEN.json"

    # Baseline sweep (budget grid)
    "$PYBIN" -m scripts.eval_policy \
      --config "$CFG" --hours "$HOURS" --seeds "$SEEDS" --seed0 "$SEED0" \
      --scenario "$SCEN" \
      --sweep \
      --sweep_reloc_planner budgeted \
      --sweep_charge_planner slack \
      --reloc_km_budgets_json "[0,10,20,40]" \
      --charge_budget_fracs_json "[0.0,0.5,1.0]" \
      --out "$OUTBASE/$NET/sweeps/${SCEN}_budgetgrid.json"

    # Heuristic
    "$PYBIN" -m ml.heuristic \
      --config "$CFG" --hours "$HOURS" --seeds "$SEEDS" --seed0 "$SEED0" \
      --scenario "$SCEN" \
      --out "$OUTBASE/$NET/heuristic/$SCEN.json"

    # Bandit (LinUCB RH)
    "$PYBIN" -m ml.bandit_contextual_rh \
      --config "$CFG" --hours "$HOURS" --episodes "$EPISODES" --seed0 "$SEED0" \
      --scenario "$SCEN" \
      --reloc budgeted --charge slack \
      --block_minutes "$BM" --warmup_blocks 1 \
      --linucb_alpha "$ALPHA" --linucb_reg 1.0 \
      --km_budgets "${KM_BUDGETS[@]}" \
      --charge_fracs "${CHARGE_FRACS[@]}" \
      --out "$OUTBASE/$NET/bandit_linucb/${SCEN}_b${BM}_a${ALPHA}.json"

    # Bandit (UCB1 over arms)
    "$PYBIN" -m ml.bandit_param_arms \
      --config "$CFG" --hours "$HOURS" --episodes "$EPISODES" --seed0 "$SEED0" \
      --scenario "$SCEN" \
      --reloc budgeted --charge slack \
      --km_budgets "${KM_BUDGETS[@]}" \
      --charge_fracs "${CHARGE_FRACS[@]}" \
      --ucb_c 2.0 \
      --default_action 0.5 0.5 0.5 0.5 \
      --out "$OUTBASE/$NET/bandit_ucb1/${SCEN}_c2.json"

    # SAC (evaluate using trained model)
    "$PYBIN" -m ml.sac \
      --config "$CFG" \
      --hours "$HOURS" \
      --scenario "$SCEN" \
      --seed0 "$SEED0" \
      --testing_seeds "$SEEDS" \
      --load_model "$SAC_MODEL_PATH" \
      --out "$OUTBASE/$NET/sac/$SCEN.json"

    echo
  done
done

echo "Done. Results under: $OUTBASE"
echo "SAC model: $SAC_MODEL_PATH"
