#!/usr/bin/env bash
set -euo pipefail

PHASE="${1:-phaseA}"  # phaseA or phaseB

if [[ "$PHASE" == "phaseA" ]]; then
  HOURS=24
  SEEDS=10
  EPISODES=10
  OUTBASE="out/paper/phaseA_24h"
  SAC_TRAIN_HOURS=24
  SAC_TRAIN_TIMESTEPS=300000   
else
  HOURS=168
  SEEDS=30
  EPISODES=30
  OUTBASE="out/paper/phaseB_168h"
  SAC_TRAIN_HOURS=168
  SAC_TRAIN_TIMESTEPS=800000  
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

# SAC: Train once, eval everywhere
SAC_TRAIN_NET="${SAC_TRAIN_NET:-porto20_s600}"
SAC_TRAIN_CFG="${NETS[$SAC_TRAIN_NET]}"

# name includes phase + train net + hours + seed0 so reruns do not overwrite accidentally
SAC_MODEL_PATH="$OUTBASE/models/sac_${SAC_TRAIN_NET}_${SAC_TRAIN_HOURS}h_seed${SEED0}.zip"

echo "== SAC: Train-once per phase, then evaluate all nets/scenarios =="
echo "PHASE:        $PHASE"
echo "TRAIN_NET:    $SAC_TRAIN_NET"
echo "TRAIN_CONFIG: $SAC_TRAIN_CFG"
echo "MODEL_OUT:    $SAC_MODEL_PATH"
echo

if [[ ! -f "$SAC_MODEL_PATH" ]]; then
  echo "== Training SAC (baseline) =="
  python -m ml.sac \
    --config "$SAC_TRAIN_CFG" \
    --hours "$SAC_TRAIN_HOURS" \
    --scenario baseline \
    --seed0 "$SEED0" \
    --training_episodes "$EPISODES" \
    --save_model \
    --out "$SAC_MODEL_PATH"
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

    # 1) Baselines (default pairs)
    python -m scripts.eval_policy \
      --config "$CFG" --hours "$HOURS" --seeds "$SEEDS" --seed0 "$SEED0" \
      --scenario "$SCEN" \
      --out "$OUTBASE/$NET/baselines/$SCEN.json"

    # 2) Baseline sweep (budget grid) - optional but useful for Pareto plots
    python -m scripts.eval_policy \
      --config "$CFG" --hours "$HOURS" --seeds "$SEEDS" --seed0 "$SEED0" \
      --scenario "$SCEN" \
      --sweep \
      --sweep_reloc_planner budgeted \
      --sweep_charge_planner slack \
      --reloc_km_budgets_json "[0,10,20,40]" \
      --charge_budget_fracs_json "[0.0,0.5,1.0]" \
      --out "$OUTBASE/$NET/sweeps/${SCEN}_budgetgrid.json"

    # 3) Heuristic
    python -m ml.heuristic \
      --config "$CFG" --hours "$HOURS" --seeds "$SEEDS" --seed0 "$SEED0" \
      --scenario "$SCEN" \
      --out "$OUTBASE/$NET/heuristic/$SCEN.json"

    # 4) Bandit (LinUCB RH)
    python -m ml.bandit_contextual_rh \
      --config "$CFG" --hours "$HOURS" --episodes "$EPISODES" --seed0 "$SEED0" \
      --scenario "$SCEN" \
      --reloc budgeted --charge slack \
      --block_minutes "$BM" --warmup_blocks 1 \
      --linucb_alpha "$ALPHA" --linucb_reg 1.0 \
      --km_budgets "${KM_BUDGETS[@]}" \
      --charge_fracs "${CHARGE_FRACS[@]}" \
      --out "$OUTBASE/$NET/bandit_linucb/${SCEN}_b${BM}_a${ALPHA}.json"

    # 5) Bandit (UCB1 over arms)
    python -m ml.bandit_param_arms \
      --config "$CFG" --hours "$HOURS" --episodes "$EPISODES" --seed0 "$SEED0" \
      --scenario "$SCEN" \
      --reloc budgeted --charge slack \
      --km_budgets "${KM_BUDGETS[@]}" \
      --charge_fracs "${CHARGE_FRACS[@]}" \
      --ucb_c 2.0 \
      --default_action 0.5 0.5 0.5 0.5 \
      --out "$OUTBASE/$NET/bandit_ucb1/${SCEN}_c2.json"

    # 6) SAC (evaluate using trained model)
    python -m ml.sac \
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
