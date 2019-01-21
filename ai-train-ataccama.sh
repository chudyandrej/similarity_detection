#!/usr/bin/env bash
JOB_NAME="seq2seq_embedding_l_2_"$(date +%s)
JOB_DIR="./"${JOB_NAME}
DATA_FILE="./data/s3+cvut_data.csv"


python -m trainer.seq2seq_embedding_l_2.task  --job-dir ${JOB_DIR} \
                                --data-file ${DATA_FILE} \

