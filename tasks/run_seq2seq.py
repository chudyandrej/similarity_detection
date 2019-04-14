
import os
from .seq2seq import *
from preprocessor.encoder import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def run(code):
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


if __name__ == '__main__':
    run(0)
