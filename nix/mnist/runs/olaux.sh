#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export CUDA_VISIBLE_DEVICES=0


UNIXTIME=$(date +%s)
PROJECT="NIX2"
EXPERIMENT_NAME="_"
DATASET="mnifar"
INTENSITY="0.5"
ZDIM="16"
NUM_EPOCHS="40"


ALGO="olaux"
ACTIVATION="sigmoid"
LEGURALIZER="negative_square" # negative_square or offset or smooth
LEGURALIZATION_COEF="0.0000000000001"


python3 algos/${ALGO}/main.py \
    hydra.run.dir="algos/configs/logs/${ALGO}/${DATASET}/${LEGURALIZER}/${UNIXTIME}" \
    project=${PROJECT} \
    experiment_name=${EXPERIMENT_NAME} \
    dataset=${DATASET} \
    dataset.intensity=${INTENSITY} \
    algo=${ALGO} \
    training.zdim=${ZDIM} \
    training.num_epochs=${NUM_EPOCHS} \
    weight.activation=${ACTIVATION} \
    weight.regularization_type=${LEGURALIZER} \
    weight.regularization_coef=${LEGURALIZATION_COEF} \
    run_time=${UNIXTIME}
