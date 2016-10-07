
class ModelNotTrainedException(Exception):
    def __init__(self):
        self.message = 'Model not trained.'

class AlgorithmNotExistException(Exception):
    def __init__(self, algoname):
        self.message = 'Algorithm '+algoname+' not exist.'

class Word2VecModelNotExistException(Exception):
    def __init__(self, path):
        self.message = 'Given path of Word2Vec not exist: '+path