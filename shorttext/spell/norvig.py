
# reference: https://norvig.com/spell-correct.html

import re
from collections import Counter

from . import SpellCorrector
from .editor import compute_set_edits1, compute_set_edits2

class NorvigSpellCorrector(SpellCorrector):
    def __init__(self):
        self.train('')

    def train(self, text):
        self.words = re.findall(r'\w+', text.lower())
        self.WORDS = Counter(self.words)
        self.N = sum(self.WORDS.values())

    def P(self, word):
        return self.WORDS[word] / float(self.N)

    def correct(self, word):
        return max(self.candidates(word), key=self.P)

    def known(self, words):
        return set(w for w in words if w in self.WORDS)

    def candidates(self, word):
        return (self.known([word]) or self.known(compute_set_edits1(word)) or self.known(compute_set_edits2(word)) or [word])

