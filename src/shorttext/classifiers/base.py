
from abc import ABC, abstractmethod


class AbstractScorer(ABC):
    @abstractmethod
    def score(self, shorttext: str) -> dict[str, float]:
        raise NotImplemented()
