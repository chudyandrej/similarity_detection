#!/usr/bin/env bash
JOB_NAME="seq2seq_embedding_l"$(date +%s)
JOB_DIR="./"${JOB_NAME}
DATA_FILE="./data/s3+cvut_data.csv"


gcloud ml-engine local train --package-path trainer.seq2seq_embedding_l \
                             --module-name trainer.seq2seq_embedding_l.task \
                             --job-dir ${JOB_DIR} \
                             -- \
                             --data-file ${DATA_FILE} \

