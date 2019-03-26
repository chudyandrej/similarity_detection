from unidecode import unidecode


class AsciiEncoding(object):
    @staticmethod
    def encode(value, use_unidecode=False):
        if use_unidecode:
            value = list(map(lambda char: unidecode(char), value))
        return list(map(lambda v: list(map(lambda char: ord(char), v)), value))

    @staticmethod
    def decode(tokens):
        return list(map(lambda t: list(map(lambda char: chr(char), t)), tokens))


