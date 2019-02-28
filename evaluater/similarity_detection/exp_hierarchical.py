import os
import pickle
import numpy as np
from keras.layers import Concatenate, Input, TimeDistributed, LSTM, Bidirectional, Embedding, GRU

from sdep import Profile, AuthorityEvaluator
from keras.models import Model, load_model


from evaluater.preprocessing import preprocess_values_standard
import trainer.custom_components as cc

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# ====================================================================
#                           EXPERIMENT 2
# [PRE-TRAIN GRU value encoder] - [Lambda mean] - [2x Dense]
# Experiment testing a model with pre-train GRU value encoder witch output is mean over quantile and next connected
# to two dense layer. That model was train as siamese model with optimization criterio euclidean distance
# RESULT:{'total': 73766, 0: 23246, None: 21204, 1: 8098, 2: 4725, 3: 3345, 4: 2616, 5: 2169, 6: 1707, 7: 1467,
# 8: 1213, 9: 1017, 10: 849, 11: 693, 12: 511, 13: 408, 14: 248, 15: 144, 16: 76, 17: 23, 18: 7}
# Percentage of found labels on first 3 index : 48%
# Conclusion: Output over mean layer was similar as mean of value vectors calculated out of model. [GOOD] But, dense
# layer connected after mean absolutely destroy accuracy of model.
# ====================================================================
def experiment2():
    def load_h5(model_src):
        model = load_model(model_src, custom_objects={
            "contrastive_loss": cc.contrastive_loss,
            "euclidean_distance": cc.euclidean_distance
        })
        encoder = Model(model.inputs[0], model.layers[7].get_output_at(0))
        encoder.summary()
        return encoder
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment2.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + '/data/models/GRUv_mean_DENSE_21.2.h5'

    print("Experiment " + experiment_name + " running ...")
    ev = AuthorityEvaluator(username='andrej', neighbors=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    print("Model successfully loaded. ")
    test_profiles = ev.get_test_dataset(data_src="s3")
    quantiles_data = np.array(list(map(lambda x: preprocess_values_standard(x.quantiles, 64), test_profiles)))
    print("Model is invoking ... ")
    embedding_vectors = encoder_model.predict(quantiles_data)
    print("Processed " + str(len(embedding_vectors)) + " value embeddings")

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(set(test_profiles))))
    ev.evaluate_embeddings(list(zip(test_profiles, embedding_vectors)))


# ====================================================================
#                           EXPERIMENT 3
# This experiment is about try connect pre-train GRU value encoder before LSTM network and then train in siamese
# architecture. [TRAINING Fail trainable on TIME distrebuted layer]
# RESULT: TODO
# ====================================================================
def experiment3():
    def load_h5(model_src):
        model = load_model(model_src, custom_objects={
            "contrastive_loss": cc.contrastive_loss,
            "l2_similarity": cc.l2_similarity
        })
        encoder = Model(model.inputs[0], model.layers[4].get_output_at(0))
        encoder.summary()
        return encoder
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment3.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + '/data/models/GRUv_LSTM.h5'

    print("Experiment " + experiment_name + " running ...")
    ev = AuthorityEvaluator(username='andrej', neighbors=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    print("Model successfully loaded. ")
    test_profiles = ev.get_test_dataset(data_src="s3")
    quantiles_data = np.array(list(map(lambda x: preprocess_values_standard(x.quantiles, 64), test_profiles)))
    print("Model is invoking ... ")
    embedding_vectors = encoder_model.predict(quantiles_data)
    print("Processed " + str(len(embedding_vectors)) + " value embeddings")

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(set(test_profiles))))
    ev.evaluate_embeddings(list(zip(test_profiles, embedding_vectors)))


# ====================================================================
#                           EXPERIMENT 4
# [PRE-TRAIN GRU value encoder] - [2x LSTM]
# RESULT: {'total': 73766, 0: 28391, None: 19123, 1: 8344, 2: 4568, 3: 2966, 4: 2230, 5: 1776,
# 6: 1374, 7: 1129, 8: 926, 9: 816, 10: 663, 11: 526, 12: 350, 13: 246, 14: 182, 15: 95, 16: 40, 17: 15, 18: 6}
# Percentage of found labels on first 3 index : 55%
# ====================================================================
def experiment4():
    def load_h5(model_src):
        model = load_model(model_src, custom_objects={
            "contrastive_loss": cc.contrastive_loss,
            "euclidean_distance": cc.euclidean_distance
        })
        encoder = Model(model.inputs[0], model.layers[5].get_output_at(0))
        encoder.summary()
        return encoder

    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment4.__name__

    print("Experiment " + experiment_name + " running ...")
    ev = AuthorityEvaluator(username='andrej', neighbors=20, metric="euclidean")
    model_path = os.environ['PYTHONPATH'].split(":")[0] + '/hierarchival_vGRU_CONV1550827207/model.h5'
    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    encoder_model.summary()
    print("Model successfully loaded. ")
    test_profiles = ev.get_test_dataset(data_src="s3")
    quantiles_data = np.array(list(map(lambda x: preprocess_values_standard(x.quantiles, 64), test_profiles)))
    print("Model is invoking ... ")
    embedding_vectors = encoder_model.predict(quantiles_data)
    print("Processed " + str(len(embedding_vectors)) + " value embeddings")

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(set(test_profiles))))
    ev.evaluate_embeddings(list(zip(test_profiles, embedding_vectors)))


# ====================================================================
#                           EXPERIMENT 5
# [Embedding]-[GRU]-[Bidirectional[GRU]]
# RESULT:{'total': 73766, 0: 31329, None: 16836, 1: 8759, 2: 4333, 3: 2930, 4: 2187, 5: 1657, 6: 1266, 7: 1081,
# 8: 883, 9: 691, 10: 572, 11: 458, 12: 334, 13: 217, 14: 130, 15: 63, 16: 27, 17: 13}
# Percentage of found labels on first 3 index : 60%
# ====================================================================
def exp5():
    def load_h5(model_src):
        model = load_model(model_src, custom_objects={
            "contrastive_loss": cc.contrastive_loss,
            "euclidean_distance": cc.euclidean_distance
        })

        encoder = Model(model.inputs[0], model.layers[6].get_output_at(0))
        encoder.summary()
        return encoder

    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = exp5.__name__

    print("Experiment " + experiment_name + " running ...")
    ev = AuthorityEvaluator(username='andrej', neighbors=20, metric="euclidean")
    model_path = os.environ['PYTHONPATH'].split(":")[0] + '/gru_hierarchical1550841088/model.h5'
    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    encoder_model.summary()
    print("Model successfully loaded. ")
    test_profiles = ev.get_test_dataset(data_src="s3")
    quantiles_data = np.array(list(map(lambda x: preprocess_values_standard(x.quantiles, 64), test_profiles)))
    print("Model is invoking ... ")
    embedding_vectors = encoder_model.predict(quantiles_data)
    print("Processed " + str(len(embedding_vectors)) + " value embeddings")

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(set(test_profiles))))
    ev.evaluate_embeddings(list(zip(test_profiles, embedding_vectors)))


################################################################################
#                   EXPERIMENT WITH GOOD RESULTS
################################################################################

# ====================================================================
#                           EXPERIMENT 1.0
# This experiment is base line of my results. Try Build hierarchical model with LSTM layer as value level and next
# use second LSTM as quantile level. This model is use with random init weights (without training)
# RESULT:{'total': 73766, 0: 33513, None: 16424, 1: 8330, 2: 3949, 3: 2550, 4: 1861, 5: 1491, 6: 1133, 7: 949,
# 8: 775, 9: 638, 10: 551, 11: 449, 12: 401, 13: 303, 14: 210, 15: 132, 16: 67, 17: 34, 18: 6}
# Percentage of found labels on first 3 index : 62%
# Conclusion: This model work fine. It has surprisingly great results.
# ====================================================================
def exp1():
    def load_h5():
        net_input = Input(shape=(11, 64), name='left_input')
        value_embedder = Embedding(input_dim=67000, output_dim=128, name='value_embedder')
        embedded = TimeDistributed(value_embedder)(net_input)

        value_encoder = LSTM(128, dropout=0.50, recurrent_dropout=0.50, name='value_encoder')
        value_encoded = TimeDistributed(value_encoder)(embedded)
        quantile_encoder = Bidirectional(LSTM(128, dropout=0.50, recurrent_dropout=0.50), name='quantile_encoder')

        encoded = quantile_encoder(value_encoded)
        return Model(inputs=net_input, outputs=encoded)
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = exp1.__name__

    print("Experiment " + experiment_name + " running ...")
    ev = AuthorityEvaluator(username='andrej', neighbors=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5()
    encoder_model.summary()
    print("Model successfully loaded. ")
    test_profiles = ev.get_test_dataset(data_src="s3")
    quantiles_data = np.array(list(map(lambda x: preprocess_values_standard(x.quantiles, 64), test_profiles)))
    print("Model is invoking ... ")
    embedding_vectors = encoder_model.predict(quantiles_data)
    print("Processed " + str(len(embedding_vectors)) + " value embeddings")

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(set(test_profiles))))
    ev.evaluate_embeddings(list(zip(test_profiles, embedding_vectors)))


# ====================================================================
#                           EXPERIMENT 6
# [Embedding]-[TimeDistributed[GRU-fix]]-[AttentionWithContext]-[Bidirectional[GRU-128]] (model3)
# RESULT: {'total': 73766, 0: 37854, None: 12854, 1: 8440, 2: 3983, 3: 2587, 4: 1826, 5: 1416, 6: 1137, 7: 917,
# 8: 708, 9: 585, 10: 479, 11: 342, 12: 259, 13: 178, 14: 115, 15: 52, 16: 19, 17: 13, 18: 2}
# Percentage of found labels on first 3 index : 68%
# ====================================================================
def exp6():
    def load_h5(model_src):
        model = load_model(model_src, custom_objects={
            "AttentionWithContext": cc.AttentionWithContext,
            "euclidean_distance": cc.euclidean_distance,
            "contrastive_loss": cc.contrastive_loss

        })

        encoder = Model(model.inputs[0], model.layers[6].get_output_at(0))
        encoder.summary()
        return encoder

    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = exp6.__name__

    print("Experiment " + experiment_name + " running ...")
    ev = AuthorityEvaluator(username='andrej', neighbors=20, metric="euclidean")
    model_path = os.environ['PYTHONPATH'].split(":")[0] + '/data/models/[Emb]-[TD[GRU-fix]]-[Attention]-[Bid[GRU]]'
    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    print("Model successfully loaded. ")
    test_profiles = ev.get_test_dataset(data_src="s3")
    quantiles_data = np.array(list(map(lambda x: preprocess_values_standard(x.quantiles, 64), test_profiles)))
    print("Model is invoking ... ")
    embedding_vectors = encoder_model.predict(quantiles_data)
    print("Processed " + str(len(embedding_vectors)) + " value embeddings")

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(set(test_profiles))))
    ev.evaluate_embeddings(list(zip(test_profiles, embedding_vectors)))


if __name__ == '__main__':
    exp6()
