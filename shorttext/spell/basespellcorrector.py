
import shorttext.utils.classification_exceptions as ce

class SpellCorrector:
    def train(self, text):
        raise ce.NotImplementedException()

    def correct(self, word):
        return word