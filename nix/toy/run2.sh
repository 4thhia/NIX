#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.50
export CUDA_VISIBLE_DEVICES=2 #!

EXPERIMENT_NAME="multi_aux"
SUB_EXPERIMENT_NAME="tanh/normalize"
MAX_ITER="500"
REGULARIZER="False" # "squared_loss" "False"
NORMALIZE="True" # "True" "False"
ACTIVATION="tanh"



# Hydraのオーバーライドを行う
python3 src/main2.py \
    hydra.run.dir="configs/_logs/${EXPERIMENT_NAME}/${SUB_EXPERIMENT_NAME}" \
    experiment_name=${EXPERIMENT_NAME} \
    sub_experiment_name=${SUB_EXPERIMENT_NAME} \
    training.max_iter=${MAX_ITER} \
    training.regularizer=${REGULARIZER} \
    training.normalize=${NORMALIZE} \
    training.activation="${ACTIVATION}"
