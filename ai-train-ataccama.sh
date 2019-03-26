#!/usr/bin/env bash
JOB_NAME="gpt2"$(date +%s)
JOB_DIR="./"${JOB_NAME}
DATA_FILE="./data/s3+cvut_data.csv"


python -m trainer.gpt2.task  --job-dir ${JOB_DIR} \
                                          --data-file ${DATA_FILE} \
