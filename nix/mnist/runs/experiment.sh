#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
export CUDA_VISIBLE_DEVICES=1 #,1,2


UNIXTIME=$(date +%s)
ALGO="classification"
EXPERIMENT_NAME="_"
DATASET="mnifar"
INTENSITY="0.5"
ZDIM="16"
NUM_EPOCHS=20


python3 experiments/${ALGO}.py \
        hydra.run.dir="algos/configs/logs/var" \
        experiment_name=${EXPERIMENT_NAME} \
        dataset=${DATASET} \
        dataset.intensity=${INTENSITY} \
        algo=experiment \
        algo.name=${ALGO} \
        training.zdim=${ZDIM} \
        training.num_epochs=${NUM_EPOCHS} \
        run_time=${UNIXTIME}