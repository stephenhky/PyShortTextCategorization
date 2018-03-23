
from .char2vec import initSentenceToCharVecEncoder

class SpellingToConcatCharVecEncoder:
    def __init__(self, alph):
        self.charevec_encoder = initSentenceToCharVecEncoder(alph)

    def encode_spelling(self, spelling):
        spmat = self.charevec_encoder.encode_sentence(spelling, len(spelling))
        return spmat.sum(axis=0)