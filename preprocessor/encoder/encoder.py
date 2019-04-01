from abc import abstractmethod


class Encoder(object):

    @abstractmethod
    def encode(self, tokens):
        pass

    @abstractmethod
    def decode(self, tokens):
        pass

    @abstractmethod
    def get_vocab_size(self):
        pass
