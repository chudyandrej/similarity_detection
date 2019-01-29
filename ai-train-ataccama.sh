#!/usr/bin/env bash
JOB_NAME="seq2seq_embedding"$(date +%s)
JOB_DIR="./"${JOB_NAME}
DATA_FILE="./data/s3+cvut_data.csv"


python -m trainer.seq2seq_embedding.task  --job-dir ${JOB_DIR} \
                                --data-file ${DATA_FILE} \

