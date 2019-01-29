import os
import sdep
import pickle

from keras.preprocessing.sequence import pad_sequences

import argparse
from evaluater.embedder import tokenizer_0_96
from unidecode import unidecode
import evaluater.load_models as lm
import evaluater.embedder as em

CHECKPOINT_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/evaluater/similarity_detection/pickles/"

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def preprocess_values(values, pad_maxlen, full_unicode=True):
    values = map(lambda x: str(x), values)
    values = map(str.strip, values)
    values = (x[::-1] for x in values)
    if full_unicode:
        values = list(map(lambda x: [ord(y) for y in x], values))
    else:
        values = map(lambda x: unidecode(x), values)
        values = map(lambda x: [tokenizer_0_96(y) for y in x], values)
    # print(list(values)[:100])
    values = pad_sequences(list(values), maxlen=pad_maxlen, truncating='pre', padding='pre')

    return values


# ====================================================================
#                           EXPERIMENT 1
# ====================================================================
def experiment_seq2seq_embedder_jointly(recompute):

    max_text_seqence_len = 64
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_embedder_jointly.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/seq2seq_embedding1548769086/model.h5"

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20, train_size=0.5)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name) and not recompute:
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        encoder_model = lm.seq2seq_embedder_jointly(model_path)
        print("Model successfully loaded. ")
        test_profiles = ev.get_test_dataset()
        print(str(len(test_profiles)) + " classes!")
        class_values = [(profile, value) for profile in test_profiles for value in profile.quantiles]
        tokened_data = preprocess_values(map(lambda x: x[1], class_values), max_text_seqence_len, full_unicode=False)
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
#                           EXPERIMENT 2
# Value embedding of seq2seq with embedder layer over 75k characters.
#        Percentage of found labels on first 3 index : 50%
# ====================================================================
def experiment_seq2seq_embedder(recompute):

    max_text_seqence_len = 64
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_embedder.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_embedding_2/model.h5"
    emb_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_embedding_2/embedding_model.h5"

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20, train_size=0.5)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH + experiment_name) and not recompute:
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH + experiment_name, "rb"))
    else:
        encoder_model = lm.load_seq2seq_embedder(model_path, emb_path)
        print("Model successfully loaded. ")
        test_profiles = ev.get_test_dataset()
        print(str(len(test_profiles)) + " classes!")
        class_values = [(profile, value) for profile in test_profiles for value in profile.quantiles]
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',
                        required=True,
                        type=int,
                        help='Experiment selection')

    parser.add_argument('--recompute', action='store_true', default=False)

    parse_args, _ = parser.parse_known_args()
    states = {}

    if parse_args.exp == 1:
        experiment_seq2seq_embedder_jointly(parse_args.recompute)
    elif parse_args.exp == 2:
        experiment_seq2seq_embedder(parse_args.recompute)


