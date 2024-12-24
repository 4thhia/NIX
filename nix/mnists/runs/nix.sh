#!/bin/bash
# tuning parameters: lmb.lr / regularization coef / weightunet.lr

PROJECT="NIX"
EXPERIMENT_NAME="emnifar"
SUB_EXPERIMENT_NAME="L2"
DATASET="emnifar"
INTENSITY="0.5"

ALGO="nix"
BETA="0.3"
LMB="10"
MAX_LR_LMB="300"
MAX_LR_LMB=$(printf "%.5f" "${MAX_LR_LMB}")
MIN_LR_LMB="0.1"
MIN_LR_LMB=$(printf "%.5f" "${MIN_LR_LMB}")

ZDIM="16"
VALID_LABELS="False"
NUM_EPOCHS="40"

ACTIVATION="sigmoid"
LEGURALIZER="L2"
MAX_LEGURALIZATION_COEF="10"
MAX_LEGURALIZATION_COEF=$(printf "%.5f" "${MAX_LEGURALIZATION_COEF}")
MIN_LEGURALIZATION_COEF="0.1"
MIN_LEGURALIZATION_COEF=$(printf "%.5f" "${MIN_LEGURALIZATION_COEF}")


MAX_LR_WEIGHTUNET="0.01"
MAX_LR_WEIGHTUNET=$(printf "%.5f" "${MAX_LR_WEIGHTUNET}")
MIN_LR_WEIGHTUNET="0.0001"
MIN_LR_WEIGHTUNET=$(printf "%.5f" "${MIN_LR_WEIGHTUNET}")

python3 algos/${ALGO}/main.py --multirun \
    hydra.run.dir="configs/logs/${ALGO}/${EXPERIMENT_NAME}/${SUB_EXPERIMENT_NAME}/${UNIXTIME}" \
    project=${PROJECT} \
    experiment_name=${EXPERIMENT_NAME} \
    sub_experiment_name=${SUB_EXPERIMENT_NAME} \
    dataset=${DATASET} \
    dataset.intensity=${INTENSITY} \
    algo=${ALGO} \
    algo.beta=${BETA} \
    algo.lmb.initial_value=${LMB} \
    algo.lmb.lr="range(${MIN_LR_LMB}, ${MAX_LR_LMB})" \
    training.zdim=${ZDIM} \
    training.valid_labels=${VALID_LABELS} \
    training.num_epochs=${NUM_EPOCHS} \
    weight.activation=${ACTIVATION} \
    weight.regularization_type=${LEGURALIZER} \
    weight.regularization_coef="range(${MIN_LEGURALIZATION_COEF}, ${MAX_LEGURALIZATION_COEF})" \
    optimizers.weightunet.lr="range(${MIN_LR_WEIGHTUNET}, ${MAX_LR_WEIGHTUNET})" \
    run_time=${UNIXTIME}