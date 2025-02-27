#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
export CUDA_VISIBLE_DEVICES=0 #!

UNIXTIME=$(date +%s)
EXPERIMENT_NAME="better_local_minima"
SUB_EXPERIMENT_NAME="plus_beta__normalize_False__regu_coef_0_025"


ALGORITHM="nix"
BETA="0.5"
LAMBDA_LR="0.1"

ACTIVATION="None" # "sigmoid" "tanh"
NORMALIZE="False" # "True" "False"
REGULARIZATION_COEF="0.025"
MAX_ITER="500"

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
    plot.lim=${LIM} \
    run_time=${UNIXTIME}