
import re

from shorttext.generators.charbase.char2concatvec import SpellingToConcatCharVecEncoder

def hasnum(word):
    return len(re.findall('\\d', word)) > 0

class SCRNNBinarizer:
    def __init__(self, alpha, signalchar_dict):
        self.signalchar_dict = signalchar_dict
        self.concatchar_encoder = SpellingToConcatCharVecEncoder(alpha)

    def noise_char(self, word, opt):
        pass

    def jumble_char(self, word, opt):
        pass
