#!/usr/bin/env bash
JOB_NAME="cnn_tcn"$(date +%s)
JOB_DIR="./"${JOB_NAME}
DATA_FILE="./data/cvutProfiles_gnumbers.csv"


gcloud ml-engine local train --package-path trainer.cnn_tcn \
                             --module-name trainer.cnn_tcn.task \
                             --job-dir ${JOB_DIR} \
                             -- \
                             --data-file ${DATA_FILE} \

