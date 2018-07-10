
from shorttext.utils.classification_exceptions import NotImplementedException

class SpellCorrector:
    """ Base class for all spell corrector.

    This class is not implemented; this can be seen as an "abstract class."

    """
    def train(self, text):
        """ Train the spell corrector with the given corpus.

        :param text: training corpus
        :type text: str
        """
        raise NotImplementedException()

    def correct(self, word):
        """ Recommend a spell correction to given the word.

        :param word: word to be checked
        :return: recommended correction
        :type word: str
        :rtype: str
        """
        return word