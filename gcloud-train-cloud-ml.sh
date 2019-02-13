#!/usr/bin/env bash

JOB_NAME="lstm_seq2seq_onehot"$(date +%s)
JOB_DIR="gs://similarity-detection/"${JOB_NAME}
DATA_FILE="gs://similarity-detection/data/s3+cvut_data.csv"
GPU_CONFIG="./trainer/cloudml-gpu.yaml"

gcloud ml-engine jobs submit training ${JOB_NAME} \
                                    --job-dir ${JOB_DIR} \
                                    --runtime-version 1.8 \
                                    --package-path trainer \
                                    --module-name trainer.lstm_seq2seq.task \
                                    --region europe-west4 \
                                    --python-version 3.6 \
                                    --config ${GPU_CONFIG} \
                                    -- \
                                    --data-file ${DATA_FILE}

