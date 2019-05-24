import os
from .triplet.models import *
from preprocessor.encoder import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def run(code):
    if code == 0:
        model_execution = HierTripletJointly(rnn_type="Lstm", attention=False, encoder=AsciiEncoding(), enc_out_dim=128,
                                             max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1", name="ULL")
    elif code == 1:
        model_execution = HierTripletJointly(rnn_type="Gru", attention=False, encoder=AsciiEncoding(), enc_out_dim=128,
                                             max_seq_len=64, rnn_dim=300, dropout=0.2, version="v1", name="UCC")
        model_execution.save_embedder()
    elif code == 2:
        model_execution = MeanTriplet(rnn_type="Lstm", attention=False, encoder=AsciiEncoding(), enc_out_dim=128,
                                      max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1", name="ULM")
    elif code == 3:
        model_execution = MeanTriplet(rnn_type="Gru", attention=False, encoder=AsciiEncoding(), enc_out_dim=128,
                                      max_seq_len=64, rnn_dim=300, dropout=0.2, version="v2", name="UCM")
    elif code == 5:
        model_execution = HierTripletJointly(rnn_type="Gru", attention=True, encoder=AsciiEncoding(), enc_out_dim=128,
                                             max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1", name="UCCWA")
    elif code == 6:
        model_execution = MeanTriplet(rnn_type="Gru", attention=True, encoder=AsciiEncoding(), enc_out_dim=128,
                                      max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1", name="UCMWA")
    elif code == 7:
        model_execution = HierTripletJointly(rnn_type="Lstm", attention=True, encoder=AsciiEncoding(), enc_out_dim=128,
                                             max_seq_len=64, rnn_dim=128, dropout=0.2, version="v1", name="ULLWA")
    elif code == 8:
        model_execution = HierTripletJointlyWithGpt2Encoder(rnn_type="Gru", attention=False, encoder=BytePairEncoding(),
                                                            enc_out_dim=128, max_seq_len=64, rnn_dim=128, dropout=0.2,
                                                            version="v1", name="GCC")
    elif code == 9:
        model_execution = MeanTripletWithGpt2Encoder(rnn_type="Gru", attention=False, encoder=BytePairEncoding(),
                                                     enc_out_dim=128, max_seq_len=64, rnn_dim=128, dropout=0.2,
                                                     version="v1", name="GCM")
    else:
        return

    # model_execution.train_model()
    # model_execution.evaluate_model()
    # model_execution.make_plots()
    # model_execution.print_training_stats()


if __name__ == '__main__':
    # ToDO
    run(1)
    # run(7)



