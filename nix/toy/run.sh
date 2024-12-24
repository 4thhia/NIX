#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
export CUDA_VISIBLE_DEVICES=3 #!

EXPERIMENT_NAME="flat_minima"
SUB_EXPERIMENT_NAME="normalize_regularize"
REGULARIZER="squared_loss" # "squared_loss"
NORMALIZE="True" # "True"

PARAMS="[0.2, 0.2]"
WEIGHTS="[1.0, 1.0]"
LAMBDA="10"

MAX_ITER="500"
MAIN_MUX="[0.0]"
MAIN_MUY="[0.0]"
MAIN_STDX="[1.5]"
MAIN_STDY="[1.5]"
MAIN_RHO="[0.0]"
MAIN_FLAT="[True]"
AUX_MUX="[-0.75, 0.5]"
AUX_MUY="[0.5, -0.5]"
AUX_STDX="[0.50, 0.80]"
AUX_STDY="[0.50, 0.80]"
AUX_RHO="[0.0, 0.0]"
AUX_FLAT="[False, False]"
LIM="2.0"

# Hydraのオーバーライドを行う
python3 src/main2.py \
    hydra.run.dir="configs/_logs/${EXPERIMENT_NAME}" \
    experiment_name=${EXPERIMENT_NAME} \
    sub_experiment_name=${SUB_EXPERIMENT_NAME} \
    training.max_iter=${MAX_ITER} \
    training.regularizer=${REGULARIZER} \
    training.normalize=${NORMALIZE} \
    params="${PARAMS}" \
    weights="${WEIGHTS}" \
    lmb=${LAMBDA} \
    main.mux="${MAIN_MUX}" \
    main.muy="${MAIN_MUY}" \
    main.stdx="${MAIN_STDX}" \
    main.stdy="${MAIN_STDY}" \
    main.rho="${MAIN_RHO}" \
    main.flat="${MAIN_FLAT}" \
    aux.mux="${AUX_MUX}" \
    aux.muy="${AUX_MUY}" \
    aux.stdx="${AUX_STDX}" \
    aux.stdy="${AUX_STDY}" \
    aux.rho="${AUX_RHO}" \
    aux.flat="${AUX_FLAT}" \
    plot.lim=${LIM}