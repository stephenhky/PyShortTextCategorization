
# Reference: https://github.com/keisks/robsut-wrod-reocginiton
# Article: http://cs.jhu.edu/~kevinduh/papers/sakaguchi17robsut.pdf

from . import SpellCorrector
from shorttext.generators import SentenceToCharVecEncoder, initSentenceToCharVecEncoder
from .binarize import SpellingToConcatCharVecEncoder, SCRNNBinarizer

class SCRNNSpellCorrector(SpellCorrector):
    def __init__(self, concatcharvec_encoder=None, charvec_encoder=None):
        self.concatcharvec_encoder = SpellingToConcatCharVecEncoder() if concatcharvec_encoder==None else concatcharvec_encoder

    def preprocess_vec(self, text):
        pass

    def train(self, text):
        pass

    def correct(self, word):
        pass
