#!/usr/bin/env bash
JOB_NAME="cnn_tcn"$(date +%s)
JOB_DIR="./"${JOB_NAME}
DATA_FILE="./data/cvutProfiles_gnumbers.csv"


python -m trainer.seq2seq.task  --job-dir ${JOB_DIR} \
                                --data-file ${DATA_FILE} \

