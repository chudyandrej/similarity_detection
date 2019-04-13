import os

from .seq2seq import *

from .siamese.models import *
from preprocessor.encoder import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def run_value_experiments(code):
    if code == 0:
        model_execution = LstmSeq2seqWithEmbedder(lstm_dim=128, dropout=0.2, embedding_dim=128, max_seq_len=64, version="AE_v1", encoder=AsciiEncoding())
    elif code == 1:
        model_execution = LstmSeq2seqWithEmbedder(lstm_dim=128, dropout=0.2, embedding_dim=128, max_seq_len=64, version="BPE_v1", encoder=BytePairEncoding())
    elif code == 2:
        model_execution = LstmSeq2seqWithGpt2Encoder(lstm_dim=128, dropout=0.2, max_seq_len=64, version="v2", encoder=BytePairEncoding())
    elif code == 3:
        model_execution = GruSeq2seqWithEmbedder(gru_dim=128, dropout=0.2, embedding_dim=128, max_seq_len=64, version="AE_v1", encoder=AsciiEncoding())
    elif code == 4:
        model_execution = GruSeq2seqWithEmbedder(gru_dim=128, dropout=0.2, embedding_dim=128, max_seq_len=64, version="BPE_v1", encoder=BytePairEncoding())
    elif code == 5:
        model_execution = GruSeq2seqWithGpt2Encoder(gru_dim=128, dropout=0.2, max_seq_len=64, version="v1", encoder=BytePairEncoding())
    elif code == 6:
        model_execution = CuDNNGRUSeq2seqWithGpt2Encoder(gru_dim=128, dropout=0.2, max_seq_len=64, version="v1", encoder=BytePairEncoding())
    else:
        return

    # model_execution.train_model()
    model_execution.evaluate_model()
    # model_execution.print_training_stats()


def run_hierarchy_experiments(code):
    if code == 0:
        model_execution = HierSiamJointly(rnn_type="Lstm", attention=False, encoder=AsciiEncoding(), enc_out_dim=128,
                                          max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1")
    elif code == 1:
        model_execution = HierSiamJointly(rnn_type="Gru", attention=False, encoder=AsciiEncoding(), enc_out_dim=128,
                                          max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1")
    elif code == 2:
        model_execution = HierSiamJointly(rnn_type="Lstm", attention=True, encoder=AsciiEncoding(), enc_out_dim=128,
                                          max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1")
    elif code == 3:
        model_execution = HierSiamJointly(rnn_type="Gru", attention=True, encoder=AsciiEncoding(), enc_out_dim=128,
                                          max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1")
    elif code == 4:
        model_execution = MeanHierSiamJointly(rnn_type="Lstm", attention=False, encoder=AsciiEncoding(), enc_out_dim=128,
                                              max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1")
    elif code == 5:
        model_execution = MeanHierSiamJointly(rnn_type="Gru", attention=False, encoder=AsciiEncoding(), enc_out_dim=128,
                                              max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1")
    elif code == 6:
        model_execution = MeanHierSiamJointly(rnn_type="Lstm", attention=True, encoder=AsciiEncoding(), enc_out_dim=128,
                                              max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1")
    elif code == 7:
        model_execution = MeanHierSiamJointly(rnn_type="Gru", attention=True, encoder=AsciiEncoding(), enc_out_dim=128,
                                              max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1")
    elif code == 8:
        model_execution = MeanHierSiamJointlyWithGpt2Encoder(rnn_type="Gru", attention=False, encoder=BytePairEncoding(),
                                                             enc_out_dim=128, max_seq_len=64, rnn_dim=128, dropout=0.2,
                                                             version="v1")
    elif code == 9:
        model_execution = MeanHierSiamJointlyWithGpt2Encoder(rnn_type="Gru", attention=True, encoder=BytePairEncoding(),
                                                             enc_out_dim=128, max_seq_len=64, rnn_dim=128, dropout=0.2,
                                                             version="v1")
    elif code == 10:
        model_execution = MeanHierSiamJointlyWithGpt2Encoder(rnn_type="Lstm", attention=True, encoder=BytePairEncoding(),
                                                             enc_out_dim=128, max_seq_len=64, rnn_dim=128, dropout=0.2,
                                                             version="v1")
    else:
        return

    # model_execution.train_model()
    # model_execution.evaluate_model()
    model_execution.print_training_stats()


def run_hybrid_model(code):
    if code == 0:
        model_execution = HierSiamJointlyWithSeq2Encoder(rnn_type="Lstm", attention=False, encoder=BytePairEncoding(),
                                                             enc_out_dim=128, max_seq_len=64, rnn_dim=128, dropout=0.2,
                                                             version="v1")
    elif code == 1:
        model_execution = HierSiamJointlyWithSeq2Encoder(rnn_type="Gru", attention=False,
                                                             encoder=BytePairEncoding(),
                                                             enc_out_dim=128, max_seq_len=64, rnn_dim=128, dropout=0.2,
                                                             version="v1")
    else:
        return

    # model_execution.train_model()
    # model_execution.evaluate_model()
    model_execution.print_training_stats()



if __name__ == '__main__':
    for i in range(0, 10):
        print("##"*10)
        print(i)
        print("##" * 10)
        run_hierarchy_experiments(i)

    # run_value_experiments(6)
    # run_hierarchy_experiments(5)
    # run_hierarchy_experiments(4)
