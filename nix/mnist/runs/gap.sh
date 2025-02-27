#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export CUDA_VISIBLE_DEVICES=1 #,1,2


UNIXTIME=$(date +%s)
PROJECT="NIX2"
EXPERIMENT_NAME="_"
DATASET="mnifar"
INTENSITY="0.5"
ZDIM="16"
NUM_EPOCHS="40"


ALGO="gap"
ACTIVATION="sigmoid"
LEGURALIZER="negative_square" # negative_square or offset or smooth
LEGURALIZATION_COEF="0.0"

BETA="-0.3"
LR_LMB="100"


python3 algos/${ALGO}/main.py \
    hydra.run.dir="algos/configs/logs/${ALGO}/${DATASET}/${LEGURALIZER}/${UNIXTIME}" \
    project=${PROJECT} \
    dataset=${DATASET} \
    experiment_name=${EXPERIMENT_NAME} \
    dataset.intensity=${INTENSITY} \
    algo=${ALGO} \
    algo.beta=${BETA} \
    algo.lmb.lr=${LR_LMB} \
    training.zdim=${ZDIM} \
    training.num_epochs=${NUM_EPOCHS} \
    weight.activation=${ACTIVATION} \
    weight.regularization_type=${LEGURALIZER} \
    weight.regularization_coef=${LEGURALIZATION_COEF} \
    run_time=${UNIXTIME}