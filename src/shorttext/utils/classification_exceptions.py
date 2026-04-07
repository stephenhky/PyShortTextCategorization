
from deprecation import deprecated


class ModelNotTrainedException(Exception):
    def __init__(self):
        self.message = 'Model not trained.'


class AlgorithmNotExistException(Exception):
    def __init__(self, algoname):
        self.message = f"Algorithm {algoname} not exist."


class WordEmbeddingModelNotExistException(Exception):
    def __init__(self, path):
        self.message = f"Given path of the word-embedding model not exist: {path}"


class UnequalArrayLengthsException(Exception):
    def __init__(self, arr1, arr2):
        self.message = f"Unequal lengths: {len(arr1)} and {len(arr2)}"


@deprecated(deprecated_in="4.0.0", removed_in="5.0.0")
class NotImplementedException(Exception):
    def __init__(self):
        self.message = 'Method not implemented.'


class IncorrectClassificationModelFileException(Exception):
    def __init__(self, expectedname, actualname):
        self.message = f"Incorrect model (expected: {expectedname} ; actual: {actualname})"


class OperationNotDefinedException(Exception):
    def __init__(self, opname):
        self.message = f"Operation {opname} not defined"
