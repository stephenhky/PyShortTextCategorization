
from abc import ABC, abstractmethod


class SpellCorrector(ABC):
    """Abstract base class for spell correctors.

    Defines the interface for spelling correction algorithms.
    """

    @abstractmethod
    def train(self, text: str) -> None:
        """Train the spell corrector on a corpus.

        Args:
            text: Training text corpus.
        """
        raise NotImplemented()

    @abstractmethod
    def correct(self, word: str) -> str:
        """Recommend a spelling correction for a word.

        Args:
            word: Word to correct.

        Returns:
            The corrected word.
        """
        return word
