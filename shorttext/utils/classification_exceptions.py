
class ModelNotTrainedException(Exception):
    def __init__(self):
        self.message = 'Model not trained.'

class AlgorithmNotExistException(Exception):
    def __init__(self, algoname):
        self.message = 'Algorithm '+algoname+' not exist.'

class Word2VecModelNotExistException(Exception):
    def __init__(self, path):
        self.message = 'Given path of Word2Vec not exist: '+path

class UnequalArrayLengthsException(Exception):
    def __init__(self, arr1, arr2):
        self.message = 'Unequal lengths: '+str(len(arr1))+" and "+str(len(arr2))

class NotImplementedException(Exception):
    def __init__(self):
        self.message = 'Method not implemented.'

class IncorrectClassificationModelFileException(Exception):
    def __init__(self, expectedname, actualname):
        self.message = 'Incorrect model (expected: '+expectedname+' ; actual: '+actualname+')'