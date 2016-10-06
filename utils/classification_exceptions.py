
class ModelNotTrainedException(Exception):
    def __init__(self):
        self.message = 'Model not trained.'