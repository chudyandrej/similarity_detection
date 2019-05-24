
import os
from .siamese.models import *
from preprocessor.encoder import *

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def run(code):
    if code == 0:
        model_execution = HierSiamJointly(rnn_type="Lstm", attention=False, encoder=AsciiEncoding(), enc_out_dim=128,
                                          max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1", name="ULL")
    elif code == 1:
        model_execution = HierSiamJointly(rnn_type="Gru", attention=False, encoder=AsciiEncoding(), enc_out_dim=128,
                                          max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1", name="UCC")
    elif code == 2:
        model_execution = HierSiamJointly(rnn_type="Lstm", attention=True, encoder=AsciiEncoding(), enc_out_dim=128,
                                          max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1", name="ULLWA")
    elif code == 3:
        model_execution = HierSiamJointly(rnn_type="Gru", attention=True, encoder=AsciiEncoding(), enc_out_dim=128,
                                          max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1", name="UCCWA")
    elif code == 4:
        model_execution = MeanHierSiamJointly(rnn_type="Lstm", attention=False, encoder=AsciiEncoding(), enc_out_dim=128,
                                              max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1", name="ULM")
    elif code == 5:
        model_execution = MeanHierSiamJointly(rnn_type="Gru", attention=False, encoder=AsciiEncoding(), enc_out_dim=128,
                                              max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1", name="UCM")
    elif code == 6:
        model_execution = MeanHierSiamJointly(rnn_type="Lstm", attention=True, encoder=AsciiEncoding(), enc_out_dim=128,
                                              max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1", name="ULMWA")
    elif code == 7:
        model_execution = MeanHierSiamJointly(rnn_type="Gru", attention=True, encoder=AsciiEncoding(), enc_out_dim=128,
                                              max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1", name="UCMWA")
    elif code == 8:
        model_execution = MeanHierSiamJointlyWithGpt2Encoder(rnn_type="Gru", attention=False, encoder=BytePairEncoding(),
                                                             enc_out_dim=128, max_seq_len=64, rnn_dim=128, dropout=0.2,
                                                             version="v1", name="GCM")
    elif code == 9:
        model_execution = MeanHierSiamJointlyWithGpt2Encoder(rnn_type="Gru", attention=True, encoder=BytePairEncoding(),
                                                             enc_out_dim=128, max_seq_len=64, rnn_dim=128, dropout=0.2,
                                                             version="v1", name="GCMWA")
    elif code == 10:
        model_execution = MeanHierSiamJointlyWithGpt2Encoder(rnn_type="Lstm", attention=True, encoder=BytePairEncoding(),
                                                             enc_out_dim=128, max_seq_len=64, rnn_dim=128, dropout=0.2,
                                                             version="v1", name="GLM")
    elif code == 11:
        model_execution = HierSiamJointlyWithGpt2Encoder(rnn_type="Lstm", attention=False, encoder=BytePairEncoding(),
                                                         enc_out_dim=128, max_seq_len=64, rnn_dim=128, dropout=0.2,
                                                         version="v1", name="GLL")
    elif code == 12:
        model_execution = HierSiamJointlyWithGpt2Encoder(rnn_type="Gru", attention=False, encoder=BytePairEncoding(),
                                                         enc_out_dim=128, max_seq_len=64, rnn_dim=128, dropout=0.2,
                                                         version="v1", name="GCC")

    else:
        return

    model_execution.train_model()
    # model_execution.evaluate_model()
    # model_execution.print_training_stats()
    # model_execution.make_plots()




if __name__ == '__main__':
    # TO train
    run(4)

    # run(12)






