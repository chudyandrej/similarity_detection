import os

from .seq2seq import *
from preprocessor.encoder import *
from sdep import Profile

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def run(code):
    if code == 0:
        model_execution = LstmSeq2seqWithEmbedder(lstm_dim=128, dropout=0.4, embedding_dim=128, max_seq_len=64, version="AE_v1", encoder=AsciiEncoding())
        model_execution.train_model()

    elif code == 1:
        model_execution = LstmSeq2seqWithEmbedder(lstm_dim=128, dropout=0.4, embedding_dim=128, max_seq_len=64, version="BPE_v1", encoder=BytePairEncoding())
        model_execution.train_model()

    elif code == 2:
        model_execution = LstmSeq2seqWithGpt2Encoder(lstm_dim=128, dropout=0.4, max_seq_len=64, version="v1", encoder=BytePairEncoding())
        model_execution.train_model()

    elif code == 3:
        model_execution = GruSeq2seqWithEmbedder(gru_dim=128, dropout=0.4, embedding_dim=128, max_seq_len=64, version="AE_v1", encoder=AsciiEncoding())
        model_execution.train_model()

    elif code == 4:
        model_execution = GruSeq2seqWithEmbedder(gru_dim=128, dropout=0.4, embedding_dim=128, max_seq_len=64, version="BPE_v1", encoder=BytePairEncoding())
        model_execution.train_model()

    elif code == 5:
        model_execution = GruSeq2seqWithGpt2Encoder(gru_dim=128, dropout=0.4, max_seq_len=64, version="v1", encoder=BytePairEncoding())
        model_execution.train_model()


if __name__ == '__main__':
    run(4)