# -*- coding: utf-8 -*-
"""seq2seq_siamese.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ge0znrxoOb0ej4jFmX3zAAVZfLMhvq9S
"""

# !wget https://www.dropbox.com/s/kxbmga5p8j8amjt/cvutProfiles_gnumbers.zip?dl=0
# !mv cvutProfiles_gnumbers.zip?dl=0 cvutProfiles_gnumbers.zip
# !unzip cvutProfiles_gnumbers.zip
# !rm cvutProfiles_gnumbers.zip
#
# !pip install unidecode

from keras.callbacks import EarlyStopping, TensorBoard
import trainer.cnn_tcn.model as model
from trainer.modelCheckpoint import ModelCheckpointMLEngine
import argparse

CHECKPOINT_FILE_PATH = 'best_model.h5'


def main(data_file, job_dir):

    input_texts, types = model.load_data(data_file)

    model_cnn_kim = model.create_model()
    model_cnn_kim.compile(optimizer="adam", loss="binary_crossentropy")
    val_data = next(model.generate_random_fit(input_texts, types))
    model_cnn_kim.fit_generator(model.generate_random_fit(input_texts, types),
                                steps_per_epoch=model.DATA_SIZE // model.BATCH_SIZE,
                                epochs=model.EPOCHS,
                                validation_data=val_data,
                                callbacks=[
                                    ModelCheckpointMLEngine(job_dir + '/model.h5', monitor='loss', verbose=1,
                                                    save_best_only=True, mode='min'),
                                    EarlyStopping(monitor='loss', patience=10, verbose=1),
                                    TensorBoard(log_dir=job_dir + '/log', write_graph=True,
                                                embeddings_freq=0)
                                ])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file',
                        required=True,
                        type=str,
                        help='Data file local or GCS')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to write checkpoints and export model')
    parse_args, _ = parser.parse_known_args()

    main(**parse_args.__dict__)