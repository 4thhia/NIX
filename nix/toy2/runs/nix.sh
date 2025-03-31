#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
export CUDA_VISIBLE_DEVICES=1 #!


UNIXTIME=$(date +%s)
EXPERIMENT_NAME="1d_gap"
SUB_EXPERIMENT_NAME="test"


ALGORITHM="nix"
BETA="0.5"
LAMBDA_LR="0.001"

ACTIVATION="tanh" # "sigmoid" "tanh"
NORMALIZE="0.0" # "True" "False"
REGULARIZATION_COEF="0.05"
MAX_ITER="200"
X_LIM="1.5"
WEIGHT_LIM="1.0"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/params/${EXPERIMENT_NAME}.sh"

# override
python3 algorithms/nix/main.py \
    hydra.run.dir="algorithms/_configs/_logs/${EXPERIMENT_NAME}/${UNIXTIME}" \
    experiment_name=${EXPERIMENT_NAME} \
    sub_experiment_name=${SUB_EXPERIMENT_NAME} \
    algorithm=${ALGORITHM} \
    algorithm.beta=${BETA} \
    algorithm.lmb.lr=${LAMBDA_LR} \
    training.activation=${ACTIVATION} \
    training.normalize=${NORMALIZE} \
    training.regularization_coef=${REGULARIZATION_COEF} \
    training.max_iter=${MAX_ITER} \
    params=${PARAMS} \
    weights=${WEIGHTS} \
    main.mux=${MAIN_MUX} \
    main.muy=${MAIN_MUY} \
    main.stdx=${MAIN_STDX} \
    main.stdy=${MAIN_STDY} \
    main.rho=${MAIN_RHO} \
    main.flat=${MAIN_FLAT} \
    aux.mux=${AUX_MUX} \
    aux.muy=${AUX_MUY} \
    aux.stdx=${AUX_STDX} \
    aux.stdy=${AUX_STDY} \
    aux.rho=${AUX_RHO} \
    aux.flat=${AUX_FLAT} \
    plot.x_lim=${X_LIM} \
    plot.weight_lim=${WEIGHT_LIM} \
    run_time=${UNIXTIME}