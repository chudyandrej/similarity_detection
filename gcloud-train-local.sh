#!/usr/bin/env bash
JOB_NAME="seq2_siamese"$(date +%s)
JOB_DIR="./"${JOB_NAME}
DATA_FILE="./data/cvutProfiles_gnumbers.csv"


gcloud ml-engine local train --package-path trainer.seq2_siamese \
                             --module-name trainer.seq2_siamese.task \
                             --job-dir ${JOB_DIR} \
                             -- \
                             --data-file ${DATA_FILE} \

