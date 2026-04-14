
from abc import ABC, abstractmethod


class Scorable(ABC):
    @abstractmethod
    def score(self, shorttexts: str | list[str]) -> dict[str, float] | list[dict[str, float]]:
        raise NotImplemented()
