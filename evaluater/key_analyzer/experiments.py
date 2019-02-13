import os

from typing import List, Optional, Tuple
from sdep import Profile, S3Profile, AuthorityEvaluator
from keras.models import Model, load_model
from evaluater.preprocessing import preprocess_values_standard

import trainer.custom_components as cc
import evaluater.embedder as em


def computing_body_by_interface(model, test_profiles: List[Profile], ev: AuthorityEvaluator, ag_method='mean'):
    print("Predicting vectors")
    uid_values = [(profile.uid, value) for profile in test_profiles for value in profile.quantiles]
    tokenized_data = preprocess_values_standard(map(lambda x: x[1], uid_values), 64)
    embeddings = model.predict(tokenized_data)
    uids = list(map(lambda x: x[0], uid_values))

    print("Clustering value vectors to column representation")
    uid_embedding = em.create_column_embedding_by(list(zip(uids, embeddings)), ag_method)
    uid_profile_index = dict(map(lambda profile: (profile.uid, profile), test_profiles))
    profile_embedding = [(uid_profile_index[uid], embedding) for uid, embedding in uid_embedding]

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(profile_embedding)))
    ev.evaluate_foreign_keys(profile_embedding)


# ====================================================================
#                           EXPERIMENT 1.0
# Experiment with GRU seq2seq with one-hot layer.

# RESULT:{'total': 63585, 0: 25282, None: 14353, 1: 5569, 2: 2545,
# 3: 1824, 4: 1271, 5: 1059, 6: 913, 7: 703, 8: 589, 9: 561, 10: 510}
# Percentage of found labels on first 3 index : 52%
# ====================================================================
def key_discovery_seq2seq_gru():
    def load_h5(path):
        model = load_model(path, custom_objects={
            "CustomRegularization": cc.CustomRegularization,
            "zero_loss": cc.zero_loss
        })
        encoder = Model(model.inputs[0], model.layers[4].output[1])
        return encoder
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = key_discovery_seq2seq_gru.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/gru_seq2seq-hot1549903432/model.h5"

    # -------------- LOAD DATA AND MODEL --------------------
    print("Experiment " + experiment_name + " running ...")
    ev_object = AuthorityEvaluator(username='andrej', radius=0.8, results_file="./results/"+experiment_name+".txt")
    test_profiles: List[Profile] = ev_object.cvut_profiles
    [profile.reduce_quantiles(range(0, 101, 10)) for profile in test_profiles]
    assert len(test_profiles) != 11, "Quantiles have invalid count"
    encoder_model = load_h5(model_path)
    encoder_model.summary()
    print("Model successfully loaded. ")
    computing_body_by_interface(encoder_model, test_profiles, ev_object)


if __name__ == "__main__":
    key_discovery_seq2seq_gru()
