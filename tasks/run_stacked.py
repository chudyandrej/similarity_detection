
import os
from .seq2seq import *
from .triplet.models import *
from .siamese.models import *


from preprocessor.encoder import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def run(code):
    if code == 0:
        value_obj = CuDNNGRUSeq2seqWithGpt2Encoder(gru_dim=128, dropout=0.2, max_seq_len=64, version="v1", encoder=BytePairEncoding())
        model_execution = HierTripletWithSeq2Encoder(rnn_type="Gru", attention=False, value_compute_obj=value_obj,
                                                     value_model_name="Bgcu", enc_out_dim=128, max_seq_len=64,
                                                     rnn_dim=128, dropout=0.2, version="v1")
    elif code == 1:
        value_obj = CuDNNGRUSeq2seqWithGpt2Encoder(gru_dim=128, dropout=0.2, max_seq_len=64, version="v1", encoder=BytePairEncoding())
        model_execution = HierTripletWithSeq2Encoder(rnn_type="Gru", attention=True, value_compute_obj=value_obj,
                                                     value_model_name="Bgcu", enc_out_dim=128, max_seq_len=64,
                                                     rnn_dim=128, dropout=0.2, version="v1")
    elif code == 2:
        value_obj= GruSeq2seqWithEmbedder(gru_dim=128, dropout=0.2, embedding_dim=128, max_seq_len=64, version="AE_v1", encoder=AsciiEncoding())
        model_execution = HierTripletWithSeq2Encoder(rnn_type="Gru", attention=False, value_compute_obj=value_obj,
                                                     value_model_name="Aeg", enc_out_dim=128, max_seq_len=64,
                                                     rnn_dim=128, dropout=0.2, version="v1")
    elif code == 3:
        value_obj = GruSeq2seqWithEmbedder(gru_dim=128, dropout=0.2, embedding_dim=128, max_seq_len=64, version="AE_v1", encoder=AsciiEncoding())
        model_execution = HierTripletWithSeq2Encoder(rnn_type="Lstm", attention=False, value_compute_obj=value_obj,
                                                     value_model_name="Aeg", enc_out_dim=128, max_seq_len=64,
                                                     rnn_dim=128, dropout=0.2, version="v1")
    elif code == 4:
        value_obj = CuDNNGRUSeq2seqWithGpt2Encoder(gru_dim=128, dropout=0.2, max_seq_len=64, version="v1", encoder=BytePairEncoding())
        model_execution = HierSiamJointlyWithSeq2Encoder(rnn_type="Gru", attention=False, value_compute_obj=value_obj,
                                                         value_model_name="Bgcu", enc_out_dim=128, max_seq_len=64,
                                                         rnn_dim=128, dropout=0.2, version="v3")
    elif code == 5:
        value_obj = GruSeq2seqWithEmbedder(gru_dim=128, dropout=0.2, embedding_dim=256, max_seq_len=64, version="AE_v1", encoder=AsciiEncoding())
        model_execution = HierSiamJointlyWithSeq2Encoder(rnn_type="Gru", attention=False, value_compute_obj=value_obj,
                                                         value_model_name="Aeg", enc_out_dim=128, max_seq_len=64,
                                                         rnn_dim=128, dropout=0.2, version="v1")
    else:
        return

    model_execution.train_model()
    model_execution.evaluate_model()
    # model_execution.print_training_stats()


if __name__ == '__main__':
    # for i in range(3, 4):
    #     print(f"{'##'*10}\n{i}\n{'##'*10}")
    #     run(i)
    # run(3)
    run(4)
    # run(5)
