
# reference: https://norvig.com/spell-correct.html

import re
from collections import Counter
from numba import jit


@jit(cache=True)
def compute_set_edits1(word):
    "All edits that are one edit away from `word`."
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


@jit(cache=True)
def compute_set_edits2(word):
    return (e2 for e1 in compute_set_edits1(word) for e2 in compute_set_edits1(e1))



class NorvigSpellCorrector:
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



# WORDS = Counter(words(open('big.txt').read()))


