
from abc import ABC, abstractmethod


class AbstractScorer(ABC):
    """Abstract base class for scoring classifiers."""

    @abstractmethod
    def score(self, shorttext: str) -> dict[str, float]:
        """Calculate classification scores.

        Args:
            shorttext: Input text to classify.

        Returns:
            Dictionary mapping class labels to scores.
        """
        raise NotImplementedError()
