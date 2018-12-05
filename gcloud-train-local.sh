#!/usr/bin/env bash
JOB_NAME="seq2seq"$(date +%s)
JOB_DIR="./"${JOB_NAME}
DATA_FILE="./data/cvutProfiles_gnumbers.csv"


gcloud ml-engine local train --package-path trainer.seq2seq \
                             --module-name trainer.seq2seq.task \
                             --job-dir ${JOB_DIR} \
                             -- \
                             --data-file ${DATA_FILE} \

