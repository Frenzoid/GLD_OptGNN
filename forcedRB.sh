#!/usr/bin/env bash
set -euo pipefail

# ---- CONFIGURATION ----
MODEL_TYPE="LiftMP"
NUM_LAYERS=8
RANK=16
PE_ENC="random_walk"
PE_DIM=8
STEPS=20000
VALID_FREQ=1000
DROPOUT=0.1
LR=0.001
BATCH_SIZE=32
SEED=42
INFINITE="True"
PARALLEL=0

# Function to run one experiment
run_experiment () {
  local NAME=$1
  local N_MIN=$2 N_MAX=$3
  local K_MIN=$4 K_MAX=$5

  echo "=== Running vertex_cover on ForcedRB ${NAME} (N=[${N_MIN},${N_MAX}], K=[${K_MIN},${K_MAX}]) ==="

  python train.py \
    --problem_type vertex_cover \
    --dataset ForcedRB \
    --gen_n ${N_MIN} ${N_MAX} \
    --gen_k ${K_MIN} ${K_MAX} \
    --prefix vertex_cover_${NAME} \
    --model_type ${MODEL_TYPE} \
    --num_layers ${NUM_LAYERS} \
    --rank ${RANK} \
    --positional_encoding ${PE_ENC} \
    --pe_dimension ${PE_DIM} \
    --steps ${STEPS} \
    --valid_freq ${VALID_FREQ} \
    --dropout ${DROPOUT} \
    --lr ${LR} \
    --batch_size ${BATCH_SIZE} \
    --seed ${SEED} \
    --infinite ${INFINITE} \
    --parallel ${PARALLEL}
}

# ---- RB200: N ~ U(6,15), K ~ U(12,21) ----
#run_experiment RB200 6 15 12 21

# ---- RB500: N ~ U(20,31), K ~ U(10,29) ----
run_experiment RB500 20 31 10 29

echo "All ForcedRB vertex_cover runs with OptGNN complete."