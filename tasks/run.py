import os

from .seq2seq.gru.gruSeq2seqWithGpt2Encoder import GruSeq2seqWithGpt2Encoder
from .seq2seq.lstm.lstmSeq2seqWithGpt2Encoder import LstmSeq2seqWithGpt2Encoder
from preprocessor.encoder.bpe import BytePairEncoding
from sdep import AuthorityEvaluator, Profile   # Needed


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


if __name__ == '__main__':
    model_exec = LstmSeq2seqWithGpt2Encoder(256, 64, "V1", BytePairEncoding())
    model_exec.evaluate_model()
