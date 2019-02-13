#!/usr/bin/env bash
JOB_NAME="gru_seq2seq"$(date +%s)
JOB_DIR="./"${JOB_NAME}
DATA_FILE="./data/s3+cvut_data.csv"


gcloud ml-engine local train --package-path trainer.gru_seq2seq \
                             --module-name trainer.gru_seq2seq.task \
                             --job-dir ${JOB_DIR} \
                             -- \
                             --data-file ${DATA_FILE} \

