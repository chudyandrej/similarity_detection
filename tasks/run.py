import os

from .seq2seq import *

from .siamese.models import *
from preprocessor.encoder import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def run_value_experiments(code):





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
