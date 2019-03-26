
import os
import numpy as np
import evaluater.embedder as em

import trainer.custom_components as cc
from keras.models import Model, load_model
from typing import List, Optional, Tuple

from sdep import AuthorityEvaluator, Profile   # Needed
from preprocessor.preprocessor import DataPreprocessorSeq2seq
from preprocessor.encoder.bpe import BytePairEncoding
from .config import Config


def exp1():
    def load_h5(path):
        model = load_model(path, custom_objects={
            "mean_squared_error_from_pred": cc.mean_squared_error_from_pred,
            "zero_loss": cc.zero_loss,
            "EmbeddingRet": cc.EmbeddingRet
        })

        encoder = Model(model.inputs[0], model.layers[4].output[1])
        return encoder
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    model_name = "V2"

    print("Experiment FIX GPT2 encoder running ...")
    ev = AuthorityEvaluator(username='andrej', neighbors=20, train_size=0.50,
                            results_file=Config.OUTPUT_SPACE+"/"+model_name)
    test_profiles: List[Profile] = ev.get_test_dataset("s3")

    # -------------- COMPUTING EXPERIMENT BODY --------------------

    encoder_model = load_h5(Config.OUTPUT_SPACE+"/"+model_name+"/model.h5")
    encoder_model.summary()
    uid_values: List[Tuple[Tuple, str]] = [(profile.uid, value) for profile in test_profiles
                                           for value in profile.quantiles]
    nn_input: np.array = DataPreprocessorSeq2seq.preprocess_input(map(lambda x: x[1], uid_values),
                                                                  64, embedder=BytePairEncoding())

    nn_output = encoder_model.predict(nn_input)

    print("Clustering value vectors to column representation")
    uids = list(map(lambda x: x[0], uid_values))
    uid_embedding = em.create_column_embedding_by(list(zip(uids, nn_output)), ag_method="mean")
    uid_profile_index = dict(map(lambda profile: (profile.uid, profile), test_profiles))
    profile_embedding = [(uid_profile_index[uid], embedding) for uid, embedding in uid_embedding]

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(uid_embedding)))
    ev.evaluate_embeddings(profile_embedding)


if __name__ == '__main__':
    exp1()
