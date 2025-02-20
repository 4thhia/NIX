#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export CUDA_VISIBLE_DEVICES=3 #,1,2


UNIXTIME=$(date +%s)
PROJECT="NIX2"
EXPERIMENT_NAME="_"
DATASET="mnifar"
INTENSITY="0.5"
ZDIM="16"
NUM_EPOCHS="40"


ALGO="baseline"


python3 algos/${ALGO}/main.py \
    hydra.run.dir="algos/configs/logs/${ALGO}/${DATASET}/${LEGURALIZER}/${UNIXTIME}" \
    project=${PROJECT} \
    experiment_name=${EXPERIMENT_NAME} \
    dataset=${DATASET} \
    dataset.intensity=${INTENSITY} \
    algo=${ALGO} \
    training.zdim=${ZDIM} \
    training.num_epochs=${NUM_EPOCHS} \
    run_time=${UNIXTIME}