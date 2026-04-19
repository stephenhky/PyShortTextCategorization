
# reference: https://norvig.com/spell-correct.html

import re
from collections import Counter
from typing import Generator

from . import SpellCorrector
from .editor import compute_set_edits1, compute_set_edits2


class NorvigSpellCorrector(SpellCorrector):
    """Spell corrector based on Peter Norvig's algorithm.

    Uses word frequency counts to suggest corrections for misspelled
    words by finding edits that exist in the vocabulary.

    Reference:
        https://norvig.com/spell-correct.html
    """

    def __init__(self):
        """Initialize the spell corrector."""
        self.train('')

    def train(self, text: str) -> None:
        """Train on a text corpus.

        Builds a word frequency dictionary from the input text.

        Args:
            text: Training text corpus.
        """
        self.words = re.findall('\\w+', text.lower())
        self.WORDS = Counter(self.words)
        self.N = sum(self.WORDS.values())

    def P(self, word: str) -> float:
        """Compute word probability from the training corpus.

        Args:
            word: Word to get probability for.

        Returns:
            Probability of the word appearing in the corpus.
        """
        return self.WORDS[word] / float(self.N)

    def correct(self, word: str) -> str:
        """Recommend spelling correction for a word.

        Args:
            word: Word to correct.

        Returns:
            Most likely correction, or the original word if no better option.
        """
        return max(self.candidates(word), key=self.P)

    def known(self, words: list[str]) -> set[str]:
        """Filter words found in the training vocabulary.

        Args:
            words: List of words to check.

        Returns:
            Subset of words that appear in the training corpus.
        """
        return set(w for w in words if w in self.WORDS)

    def candidates(self, word: str) -> Generator[str, None, None]:
        """Generate spelling correction candidates.

        Checks exact match, then edits of distance 1 and 2.

        Args:
            word: Word to find candidates for.

        Yields:
            Viable correction candidates.
        """
        return (self.known([word]) or self.known(compute_set_edits1(word)) or self.known(compute_set_edits2(word)) or [word])

