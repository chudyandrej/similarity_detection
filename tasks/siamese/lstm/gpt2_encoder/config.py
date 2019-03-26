import os

# RUNNING config
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class Config:
    # Experiment config
    NAME = "gpt2_encoder"
    OUTPUT_SPACE = os.environ['PYTHONPATH'].split(":")[0] + "/outcome/seq2seq/lstm/"+NAME
    os.makedirs(OUTPUT_SPACE, exist_ok=True)

    # Data config
    DATA_PATH = os.environ['PYTHONPATH'].split(":")[0]+"/data/s3+cvut_data.csv"
    MAX_TEXT_SEQUENCE_LEN = 64

    # Options for load GPT2
    model_folder = os.environ['PYTHONPATH'].split(":")[0] + '/data/models/117M'
    GPT2_CONFIG_PATH = os.path.join(model_folder, 'hparams.json')
    GPT2_CHECKPOINT_PATH = os.path.join(model_folder, 'model.ckpt')
    GPT2_ENCODER_PATH = os.path.join(model_folder, 'encoder.json')
    GPT2_VOCAB_PATH = os.path.join(model_folder, 'vocab.bpe')

