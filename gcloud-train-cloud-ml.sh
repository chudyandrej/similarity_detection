#!/usr/bin/env bash

JOB_NAME="seq2seq"$(date +%s)
JOB_DIR="gs://seq2seq_europe/"${JOB_NAME}
DATA_FILE="gs://seq2seq_europe/data/cvutProfiles_gnumbers.csv"
GPU_CONFIG="./trainer/cloudml-gpu.yaml"

gcloud ml-engine jobs submit training ${JOB_NAME} \
                                    --job-dir ${JOB_DIR} \
                                    --runtime-version 1.8 \
                                    --package-path trainer \
                                    --module-name trainer.seq2seq.task \
                                    --region europe-west1 \
                                    --python-version 3.5 \
                                    --config ${GPU_CONFIG} \
                                    -- \
                                    --data-file ${DATA_FILE}

