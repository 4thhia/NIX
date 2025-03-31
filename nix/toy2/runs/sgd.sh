#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
export CUDA_VISIBLE_DEVICES=0 #!


EXPERIMENT_NAME="1d_gap"
SUB_EXPERIMENT_NAME="test"
UNIXTIME=$(date +%s)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/params/${EXPERIMENT_NAME}.sh"

ALGORITHM="sgd"
MAX_ITER="100"

LR="0.01"

# Hydraのオーバーライドを行う
python3 algorithms/sgd/main.py \
    hydra.run.dir="algorithms/_configs/_logs/${EXPERIMENT_NAME}/${SUB_EXPERIMENT_NAME}" \
    experiment_name=${EXPERIMENT_NAME} \
    sub_experiment_name=${SUB_EXPERIMENT_NAME} \
    algorithm=${ALGORITHM} \
    training.max_iter=${MAX_ITER} \
    optimizers.lr_params=${LR} \
    params=${PARAMS} \
    main.mux=${MAIN_MUX} \
    main.muy=${MAIN_MUY} \
    main.stdx=${MAIN_STDX} \
    main.stdy=${MAIN_STDY} \
    main.rho=${MAIN_RHO} \
    main.flat=${MAIN_FLAT} \
    plot.x_lim=${X_LIM} \
    plot.weight_lim=${WEIGHT_LIM} \
    run_time=${UNIXTIME}
