#!/usr/bin/env bash

JOB_NAME="cnn_zhang"$(date +%s)
JOB_DIR="gs://seq2seq_europe/"${JOB_NAME}
DATA_FILE="gs://seq2seq_europe/data/cvutProfiles_gnumbers.csv"
GPU_CONFIG="./trainer/cloudml-gpu.yaml"

gcloud ml-engine jobs submit training ${JOB_NAME} \
                                    --job-dir ${JOB_DIR} \
                                    --runtime-version 1.8 \
                                    --package-path trainer \
                                    --module-name trainer.cnn_zhang.task \
                                    --region europe-west4 \
                                    --python-version 3.5 \
                                    --config ${GPU_CONFIG} \
                                    -- \
                                    --data-file ${DATA_FILE}

