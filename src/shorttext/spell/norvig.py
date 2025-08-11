
# reference: https://norvig.com/spell-correct.html

import re
from collections import Counter

from . import SpellCorrector
from .editor import compute_set_edits1, compute_set_edits2


class NorvigSpellCorrector(SpellCorrector):
    """ Spell corrector described by Peter Norvig in his blog. (https://norvig.com/spell-correct.html)

    """
    def __init__(self):
        """ Instantiate the class

        """
        self.train('')

    def train(self, text):
        """ Given the text, train the spell corrector.

        :param text: training corpus
        :type text: str
        """
        self.words = re.findall('\\w+', text.lower())
        self.WORDS = Counter(self.words)
        self.N = sum(self.WORDS.values())

    def P(self, word):
        """ Compute the probability of the words randomly sampled from the training corpus.

        :param word: a word
        :return: probability of the word sampled randomly in the corpus
        :type word: str
        :rtype: float
        """
        return self.WORDS[word] / float(self.N)

    def correct(self, word):
        """ Recommend a spelling correction to the given word

        :param word: a word
        :return: recommended correction
        :type word: str
        :rtype: str
        """
        return max(self.candidates(word), key=self.P)

    def known(self, words):
        """ Filter away the words that are not found in the training corpus.

        :param words: list of words
        :return: list of words that can be found in the training corpus
        :type words: list
        :rtype: list
        """
        return set(w for w in words if w in self.WORDS)

    def candidates(self, word):
        """ List potential candidates for corrected spelling to the given words.

        :param word: a word
        :return: list of recommended corrections
        :type word: str
        :rtype: list
        """
        return (self.known([word]) or self.known(compute_set_edits1(word)) or self.known(compute_set_edits2(word)) or [word])

