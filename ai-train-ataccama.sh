#!/usr/bin/env bash
JOB_NAME="lstm_hierarchical_base_"$(date +%s)
JOB_DIR="./"${JOB_NAME}
DATA_FILE="./data/s3+cvut_data.csv"


python -m trainer.lstm_hierarchical_base.task  --job-dir ${JOB_DIR} \
                                --data-file ${DATA_FILE} \

