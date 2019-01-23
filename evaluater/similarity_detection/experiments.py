import os
import sdep
import pickle

import numpy as np
from unidecode import unidecode

from evaluater.computing import compute_neighbors
from evaluater.similarity_detection.evaluation import evaluate_similarity_index
from keras.preprocessing.sequence import pad_sequences

import data_preparation.cvut as dt
import evaluater.load_models as lm
import evaluater.embedder as em

CHECKPOINT_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/evaluater/similarity_detection/pickles/"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ====================================================================
#                           HELPER FUNCTIONS
# ====================================================================
def print_similar_domain(similarity_index, dataclass):
    for column_name_base, similar in similarity_index.items():
        tmp1 = list(filter(lambda x: x[1] == column_name_base, dataclass.df.values))[:5]

        max_len = max(list(map(lambda x: len(str(x[0])), tmp1)))
        if max_len > 30 or len(similar) == 0:
            continue
        print("======================================================")
        print(str(list(map(lambda x: x[0], tmp1))))
        print("----->")
        for column_name in similar[:5]:
            if len(similar) == 0:
                continue
            tmp2 = list(filter(lambda x: x[1] == column_name, dataclass.df.values))[:5]
            print(list(map(lambda x: x[0], tmp2)))




# ====================================================================
#                           EXPERIMENT 1
# ====================================================================
def experiment_seq2seq_siamese():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_siamese.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_siamese1543504392-model.h5"
    print("Experiment " + experiment_name + " running ...")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        dataclass = dt.CvutDataset(dt.SelectData.profile_similarity_basic)
        encoder_model = lm.load_seq2_siamese(model_path)
        print("Checkpoint not found. Calculating...")
        dataclass.df.value = dataclass.df.value.map(lambda x: unidecode(str(x))[:63])
        value_embedding = em.convert_to_vec_onehot(encoder_model, dataclass.df["value"].values, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(dataclass.df["type"].values, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes), n_neighbors=250, radius=0.15,
                                         mode="radius+kneighbors")

    # print_similar_domain(similarity_index, dt.CvutDataset(dt.SelectData.profile_similarity_basic))
    index = evaluate_similarity_index(similarity_index)
    return index


# ====================================================================
#                           EXPERIMENT 2
# ====================================================================
def experiment_seq2_siamese():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2_siamese.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2_siamese1543913096-model.h5"
    print("Experiment " + experiment_name + " running ...")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        dataclass = dt.CvutDataset(dt.SelectData.profile_similarity_basic)
        encoder_model = lm.load_seq2_siamese(model_path)
        print("Checkpoint not found. Calculating...")
        dataclass.df.value = dataclass.df.value.map(lambda x: unidecode(str(x))[:63])
        value_embedding = em.convert_to_vec_onehot(encoder_model, dataclass.df["value"].values, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(dataclass.df["type"].values, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes), n_neighbors=100, radius=0.1,
                                         mode="radius+kneighbors")

    # print_similar_domain(similarity_index, dt.CvutDataset(dt.SelectData.profile_similarity_basic))
    index = evaluate_similarity_index(similarity_index)
    return index


# ====================================================================
#                           EXPERIMENT 3
# ====================================================================
def experiment_seq2seq():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq1544020916-model.h5"
    print("Experiment " + experiment_name + " running ...")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        dataclass = dt.CvutDataset(dt.SelectData.profile_similarity_basic)
        encoder_model = lm.load_seq2seq(model_path)
        print("Checkpoint not found. Calculating...")
        dataclass.df.value = dataclass.df.value.map(lambda x: unidecode(str(x))[:63])
        value_embedding = em.convert_to_vec_onehot(encoder_model, dataclass.df["value"].values, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(dataclass.df["type"].values, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes), n_neighbors=100, radius=0.3,
                                         mode="radius+kneighbors")

    # print_similar_domain(similarity_index, dt.CvutDataset(dt.SelectData.profile_similarity_basic))
    index = evaluate_similarity_index(similarity_index)
    return index


# ====================================================================
#                           EXPERIMENT 4
# ====================================================================
def experiment_cnn_kim():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_cnn_kim.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/cnn_kim1544535197-model.h5"
    print("Experiment " + experiment_name + " running ...")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        dataclass = dt.CvutDataset(dt.SelectData.profile_similarity_basic)
        encoder_model = lm.load_cnn_kim(model_path)
        print("Checkpoint not found. Calculating...")
        dataclass.df.value = dataclass.df.value.map(lambda x: unidecode(str(x))[:63])
        value_embedding = em.convert_to_vec_tok(encoder_model, dataclass.df["value"].values, 64)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(dataclass.df["type"].values, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes), n_neighbors=100, radius=0.3,
                                         mode="radius+kneighbors")

    # print_similar_domain(similarity_index, dt.CvutDataset(dt.SelectData.profile_similarity_basic))
    index = evaluate_similarity_index(similarity_index)
    return index


# ====================================================================
#                           EXPERIMENT 5
# ====================================================================
def experiment_cnn_tck():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_cnn_tck.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/cnn_tcn1544609078-model.h5"
    print("Experiment " + experiment_name + " running ...")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        dataclass = dt.CvutDataset(dt.SelectData.profile_similarity_basic)
        encoder_model = lm.load_cnn_tcn(model_path)
        print("Checkpoint not found. Calculating...")
        dataclass.df.value = dataclass.df.value.map(lambda x: unidecode(str(x))[:63])
        value_embedding = em.convert_to_vec_tok(encoder_model, dataclass.df["value"].values, 64)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(dataclass.df["type"].values, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes), n_neighbors=100, radius=0.3,
                                         mode="radius+kneighbors")

    # print_similar_domain(similarity_index, dt.CvutDataset(dt.SelectData.profile_similarity_basic))
    index = evaluate_similarity_index(similarity_index)
    return index


# ====================================================================
#                           EXPERIMENT 6
# ====================================================================
def experiment_cnn_tck2():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_cnn_tck.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/cnn_tcn1544609078-model.h5"
    print("Experiment " + experiment_name + " running ...")
    class_embeddings = {}
    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embeddings = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        dataclass = dt.CvutDataset(dt.SelectData.profile_similarity_basic)
        encoder_model = lm.load_cnn_tcn(model_path)
        print("Checkpoint not found. Calculating...")
        dataclass.df.value = dataclass.df.value.map(lambda x: unidecode(str(x))[:63])
        value_embedding = em.convert_to_vec_tok(encoder_model, dataclass.df["value"].values, 64)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embeddings = em.create_column_embedding_by_mrc(list(zip(dataclass.df["type"].values, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embeddings, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes = []
    embedding_vectors = []
    for key, embeddings in class_embeddings.items():
        for embedding in embeddings:
            classes.append(key)
            embedding_vectors.append(embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes), n_neighbors=100, radius=0.3,
                                         mode="radius+kneighbors")
    index = evaluate_similarity_index(similarity_index)
    return index


# ====================================================================
#                           EXPERIMENT 7
# ====================================================================
def experiment_seq2seq_siamese_sdep():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_siamese_sdep.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_siamese1543504392-model.h5"
    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)
    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        test_data = ev.get_test_dataset()
        quantiles = [data.quantiles for data in test_data]
        quantiles = [unidecode(str(q))[:63] for q in quantiles]
        encoder_model = lm.load_seq2_siamese(model_path)
        print("Checkpoint not found. Calculating...")

        value_embedding = em.convert_to_vec_onehot(encoder_model, quantiles, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(test_data, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 8
# ====================================================================
def experiment_seq2_siamese_sdep():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2_siamese_sdep.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2_siamese1543913096-model.h5"
    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)
    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found :-)")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        test_data = ev.get_test_dataset()
        # test_data = list(filter(lambda x: x.dtype == "object", test_data))
        quantiles = [data.quantiles for data in test_data]
        quantiles = [unidecode(str(q))[:63] for q in quantiles]
        encoder_model = lm.load_seq2_siamese(model_path)
        print("Checkpoint not found. Calculating...")
        value_embedding = em.convert_to_vec_onehot(encoder_model, quantiles, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(test_data, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 9
# ====================================================================
def experiment_seq2seq_sdep():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_sdep.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq1544020916-model.h5"
    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        test_data = ev.get_test_dataset()
        # test_data = list(filter(lambda x: x.dtype == "object", test_data))
        quantiles = [data.quantiles for data in test_data]
        quantiles = [unidecode(str(q))[:63] for q in quantiles]
        encoder_model = lm.load_seq2seq(model_path)
        print("Checkpoint not found. Calculating...")
        value_embedding = em.convert_to_vec_onehot(encoder_model, quantiles, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(test_data, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 10
# ====================================================================
def experiment_cnn_kim_sdep():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_cnn_kim_sdep.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/cnn_kim1544535197-model.h5"
    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        test_data = ev.get_test_dataset()
        quantiles = [data.quantiles for data in test_data]
        quantiles = [unidecode(str(q))[:63] for q in quantiles]
        encoder_model = lm.load_cnn_kim(model_path)
        print("Checkpoint not found. Calculating...")
        value_embedding = em.convert_to_vec_tok(encoder_model, quantiles, 64)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(test_data, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 11
# ====================================================================
def experiment_cnn_tck_sdeq():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_cnn_tck_sdeq.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/cnn_tcn1544609078-model.h5"
    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        test_data = ev.get_test_dataset()
        quantiles = [data.quantiles for data in test_data]
        quantiles = [unidecode(str(q))[:63] for q in quantiles]
        encoder_model = lm.load_cnn_tcn(model_path)
        print("Checkpoint not found. Calculating...")
        value_embedding = em.convert_to_vec_tok(encoder_model, quantiles, 64)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(test_data, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 12
# ====================================================================
def experiment_seq2seq_sdep_2():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_sdep_2.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq1544020916-model.h5"
    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        test_data = ev.get_test_dataset()
        # test_data = list(filter(lambda x: x.dtype == "object", test_data))
        quantiles = [data.quantiles for data in test_data]
        quantiles = [unidecode(str(q))[:63] for q in quantiles]
        encoder_model = lm.load_seq2seq(model_path)
        print("Checkpoint not found. Calculating...")
        value_embedding = em.convert_to_vec_onehot(encoder_model, quantiles, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_mrc(list(zip(test_data, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 13
# ====================================================================
def experiment_seq2seq_siamese_sdep_2():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_siamese_sdep_2.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_siamese1543504392-model.h5"
    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)
    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        test_data = ev.get_test_dataset()
        quantiles = [data.quantiles for data in test_data]
        quantiles = [unidecode(str(q))[:63] for q in quantiles]
        encoder_model = lm.load_seq2_siamese(model_path)
        print("Checkpoint not found. Calculating...")

        value_embedding = em.convert_to_vec_onehot(encoder_model, quantiles, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_mrc(list(zip(test_data, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 14
# ====================================================================
def experiment_seq2seq_sdep_3():
    max_text_seqence_len = 100
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_sdep_3.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq1547040526-model.h5"
    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        test_data = ev.get_test_dataset()

        values = []
        classes = []
        for data in test_data:
            values += [unidecode(str(q)[:max_text_seqence_len]) for q in data.quantiles]
            classes += [data]*len(data.quantiles)

        encoder_model = lm.load_seq2seq(model_path)
        print("Checkpoint not found. Calculating...")
        value_embedding = em.convert_to_vec_onehot(encoder_model, values, max_text_seqence_len, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(classes, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 15
# ====================================================================
def experiment_seq2seq_embedder():
    def preprocess_values(values, pad_maxlen):
        values = map(str, values)
        values = map(str.strip, values)
        values = (x[::-1] for x in values)
        values = list(map(lambda x: [ord(y) for y in x], values))
        values = pad_sequences(values, maxlen=pad_maxlen, truncating='pre', padding='pre')
        return values

    max_text_seqence_len = 64
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_embedder.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_embedding_2/model.h5"
    emb_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_embedding_2/embedding_model.h5"

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20, train_size=0.7)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        encoder_model = lm.load_seq2seq_embedder(model_path, emb_path)
        print("Model successfully loaded. ")
        test_profiles = ev.get_test_dataset()
        print(str(len(test_profiles)) + " classes!")
        class_values = [(profile, value) for profile in test_profiles for value in profile.quantiles]
        class_values = list(class_values)
        tokened_data = preprocess_values(map(lambda x: x[1], class_values), max_text_seqence_len)
        value_embeddings = encoder_model.predict(tokened_data)
        class_embeddings = list(map(lambda x: x[0], class_values))
        print(str(len(value_embeddings)) + " values for activation.")
        print(str(len(class_embeddings)) + " classes for data.")

        class_embedding = em.create_column_embedding_by_avg(list(zip(class_embeddings, value_embeddings)))
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 16
# ====================================================================
def experiment_seq2seq_hierarchy_lstm():
    def preprocess_quantiles(quantiles, pad_maxlen):
        quantiles = map(str, quantiles)
        quantiles = map(str.strip, quantiles)
        quantiles = (x[::-1] for x in quantiles)
        quantiles = list(map(lambda x: [ord(y) for y in x], quantiles))
        quantiles = pad_sequences(quantiles, maxlen=pad_maxlen, truncating='pre', padding='pre')
        return quantiles

    max_text_seqence_len = 64
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_hierarchy_lstm.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_embedding_2/model.h5"
    emb_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_embedding_2/embedding_model.h5"

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20, train_size=0.7)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        encoder_model = lm.load_hierarchy_lstm_model(model_path, emb_path)
        print("Checkpoint not found. Calculating...")
        test_profiles = ev.get_test_dataset()
        print(str(len(test_profiles)) + " classes!")

        quantiles = np.array(list(map(lambda x: preprocess_quantiles(x.quantiles, max_text_seqence_len), test_profiles)))
        embedding_vectors = encoder_model.predict(quantiles)

        print("Processed " + str(len(embedding_vectors)) + " value embeddings")
        class_embedding = list(zip(test_profiles, embedding_vectors))
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 17
# ====================================================================
def experiment_seq2seq_hierarchy_lstm_base():
    def preprocess_quantiles(quantiles, pad_maxlen):
        quantiles = map(str, quantiles)
        quantiles = map(str.strip, quantiles)
        quantiles = (x[::-1] for x in quantiles)
        quantiles = list(map(lambda x: [ord(y) for y in x], quantiles))
        quantiles = pad_sequences(quantiles, maxlen=pad_maxlen, truncating='pre', padding='pre')
        return quantiles

    max_text_seqence_len = 64
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_hierarchy_lstm_base.__name__

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        encoder_model = lm.load_hierarchy_lstm_base_model()
        print("Checkpoint not found. Calculating...")

        test_data = ev.get_test_dataset()

        quantiles = np.array(list(map(lambda x: preprocess_quantiles(x.quantiles, max_text_seqence_len), test_data)))
        embedding_vectors = encoder_model.predict(quantiles)

        print("Processed " + str(len(embedding_vectors)) + " value embeddings")
        class_embedding = list(zip(test_data, embedding_vectors))
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


