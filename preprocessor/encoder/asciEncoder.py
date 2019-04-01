from .encoder import Encoder

from unidecode import unidecode


class AsciiEncoding(Encoder):
    def __init__(self, use_unidecode=False):
        self.unicode_size = 65536
        self.use_unidecode = use_unidecode

    def decode(self, tokens):
        return list(map(lambda t: list(map(lambda char: chr(char), t)), tokens))

    def encode(self, value):
        if self.use_unidecode:
            value = list(map(lambda char: unidecode(char), value))
        return list(map(lambda char: ord(char), value))

    def get_vocab_size(self):
        return self.unicode_size



