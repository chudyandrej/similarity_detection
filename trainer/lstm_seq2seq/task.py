# -*- coding: utf-8 -*-
"""seq2seq_siamese.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ge0znrxoOb0ej4jFmX3zAAVZfLMhvq9S
"""

from keras.callbacks import EarlyStopping, TensorBoard

import trainer.lstm_seq2seq.model as model
import trainer.custom_components as cc
from trainer.modelCheckpoint import ModelCheckpointMLEngine
from sklearn.model_selection import train_test_split
import argparse
import os

CHECKPOINT_FILE_PATH = 'best_model.h5'


def main(data_file, job_dir):
    os.makedirs(job_dir)

    tokenized_data, count_chars = model.load_and_preprocess_data(data_file)
    print("Index contains " + str(count_chars) + " chars!")
    # model_seq2seq = model.create_model_fullunicode(count_chars + 2)
    model_seq2seq = model.create_model_onehot_layer(2100)
    train_data, valid_data = train_test_split(list(zip(tokenized_data[0], tokenized_data[1], tokenized_data[2])),
                                              train_size=0.9, random_state=18)
    print("Training set has " + str(len(train_data)) + "values!")
    print("Validation set has " + str(len(valid_data)) + "values!")
    valid_data = next(model.generate_batches(valid_data))
    model_seq2seq.compile(optimizer='adam', loss=cc.zero_loss)
    model_seq2seq.summary()
    model_seq2seq.fit(model.generate_batches(train_data),
                                steps_per_epoch=len(train_data) // model.BATCH_SIZE,
                                epochs=model.EPOCHS,
                                workers=0,
                                validation_data=valid_data,
                                callbacks=[
                                    ModelCheckpointMLEngine(job_dir + '/model.h5', monitor='val_loss', verbose=1,
                                                            save_best_only=True, mode='min'),
                                    EarlyStopping(monitor='val_loss', patience=15, verbose=1),
                                    TensorBoard(log_dir=job_dir + '/log', write_graph=True, embeddings_freq=0)
                                ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file',
                        required=False,
                        type=str,
                        help='Data file local or GCS')
    parser.add_argument('--job-dir',
                        required=False,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')
    parse_args, _ = parser.parse_known_args()

    main(**parse_args.__dict__)
