
from os import PathLike
from pathlib import Path

from deprecation import deprecated
import numpy as np


class ModelNotTrainedException(Exception):
    """Exception raised when attempting to use an untrained model."""
    def __init__(self):
        self.message = 'Model not trained.'


class AlgorithmNotExistException(Exception):
    """Exception raised when a requested algorithm is not available."""
    def __init__(self, algoname: str):
        self.message = f"Algorithm {algoname} not exist."


class WordEmbeddingModelNotExistException(Exception):
    """Exception raised when the word embedding model file is not found."""
    def __init__(self, path: str | PathLike):
        self.message = f"Given path of the word-embedding model not exist: {path.as_posix() if isinstance(path, Path) else path}"


class UnequalArrayLengthsException(Exception):
    """Exception raised when two arrays have unequal lengths."""
    def __init__(self, arr1: np.ndarray | list, arr2: np.ndarray | list):
        self.message = f"Unequal lengths: {len(arr1)} and {len(arr2)}"


@deprecated(deprecated_in="4.0.0", removed_in="5.0.0")
class NotImplementedException(Exception):
    """Exception raised when a method is not implemented."""
    def __init__(self):
        self.message = 'Method not implemented.'


class IncorrectClassificationModelFileException(Exception):
    """Exception raised when model file doesn't match expected type."""
    def __init__(self, expectedname: str, actualname: str):
        self.message = f"Incorrect model (expected: {expectedname} ; actual: {actualname})"


class OperationNotDefinedException(Exception):
    """Exception raised when an operation is not defined."""
    def __init__(self, opname: str):
        self.message = f"Operation {opname} not defined"
