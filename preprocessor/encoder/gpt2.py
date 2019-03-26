import os
import pickle
import numpy as np
import regex as re
from sdep.profiler import Profile
from keras_gpt_2 import load_trained_model_from_checkpoint

from .bpe import BytePairEncoding


# Options for load GPT2
model_folder = os.environ['PYTHONPATH'].split(":")[0] + '/data/models/117M'
config_path = os.path.join(model_folder, 'hparams.json')
checkpoint_path = os.path.join(model_folder, 'model.ckpt')
encoder_path = os.path.join(model_folder, 'encoder.json')
vocab_path = os.path.join(model_folder, 'vocab.bpe')


class GPT2ProfileEncoding(object):
    def __init__(self, all_profiles):
        self.model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
        self.bpe = BytePairEncoding()
        self.embeddings = []
        self.batch_size = 10000
        self.profiles = all_profiles
        self.profiles_batches = [all_profiles[i:i + self.batch_size] for i in range(0, len(all_profiles), self.batch_size)]
        self.profile_index = self.create_profile_index(checkpoint_path+"/gpt2_embedding_gpt.pck")

    def create_profile_index(self, checkpoint_path):

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                self.embeddings = pickle.load(f)
            return dict(zip(self.profiles, self.embeddings))

        for page_index, batch_of_profiles in enumerate(self.profiles_batches):
            print(f"List {len(batch_of_profiles) * page_index}  {page_index}/{len(self.profiles)} #####")
            quantiles_texts = list(map(self.convert_quantile_to_text, batch_of_profiles))
            encodes = [self.bpe.encode(text) for text in quantiles_texts]
            text_lens = [len(encode) for encode in encodes]
            max_len = max(text_lens)
            input_data = [encode + [0] * (max_len - len(encode)) for encode in encodes]
            output_data = self.model.predict(np.array(input_data))
            for i in range(len(output_data)):
                self.embeddings.append(output_data[i][text_lens[i] - 1].copy())
            output_data = None
            if checkpoint_path is not None:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(self.embeddings, f)

        self.profile_index = dict(zip(self.profiles, self.embeddings))

    def encode(self, prof_obj: Profile):
        return self.profile_index[prof_obj]

    @classmethod
    def convert_quantile_to_text(prof_obj: Profile):
        text = ""
        for value in prof_obj.quantiles:
            value = str(value).replace("$", " ")
            text += value + " $ "
        return text


