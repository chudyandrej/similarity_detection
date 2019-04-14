import os
from .triplet.models import *
from preprocessor.encoder import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def run(code):
    if code == 0:
        model_execution = HierTripletJointly(rnn_type="Lstm", attention=False, encoder=AsciiEncoding(), enc_out_dim=128, max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1")
    elif code == 1:
        model_execution = HierTripletJointly(rnn_type="Gru", attention=False, encoder=AsciiEncoding(), enc_out_dim=128, max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1")
    elif code == 2:
        model_execution = MeanTriplet(rnn_type="Lstm", attention=False, encoder=AsciiEncoding(), enc_out_dim=128, max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1")
    elif code == 3:
        model_execution = MeanTriplet(rnn_type="Gru", attention=False, encoder=AsciiEncoding(), enc_out_dim=128, max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1")
    elif code == 4:
        model_execution = HierTripletWithSeq2Encoder(rnn_type="Gru", attention=False, encoder=AsciiEncoding(), enc_out_dim=128, max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1")
    else:
        return

    model_execution.train_model()


if __name__ == '__main__':
    # for i in range(0, 2):
    #     print(f"{'##'*10}\n{i}\n{'##'*10}")
    #     run(i)

    for i in range(2, 5):
        print(f"{'##'*10}\n{i}\n{'##'*10}")
        run(i)


