import pickle
from collections import defaultdict

import numpy as np
from nltk import word_tokenize
from scipy.spatial.distance import cosine

from utils import ModelNotTrainedException


class SumEmbeddedVecClassifier:
    """
    This is a supervised classification algorithm for short text categorization.
    Each class label has a few short sentences, where each token is converted
    to an embedded vector, given by a pre-trained word-embedding model (e.g., Google Word2Vec model).
    They are then summed up and normalized to a unit vector for that particular class labels.
    To perform prediction, the input short sentences is converted to a unit vector
    in the same way. The similarity score is calculated by the cosine similarity.

    A pre-trained Google Word2Vec model can be downloaded

    Examples:
    >>> from gensim.models import Word2Vec
    >>> wvmodel = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    >>> import shorttext.embed.sumvec.SumWord2VecClassification as sumwv
    >>> classifier = sumwv.SumEmbeddedVecClassifier(wvmodel)
    >>> classifier.train(trainclassdict)    # assuming trainclassdict was previously loaded
    >>> classifier.score('artificial intelligence')
    """

    def __init__(self, wvmodel, classdict=None, vecsize=300):
        self.wvmodel = wvmodel
        self.classdict = classdict
        self.vecsize = vecsize
        self.trained = False

    def train(self):
        self.addvec = defaultdict(lambda : np.zeros(self.vecsize))
        for classtype in self.classdict:
            for shorttext in self.classdict[classtype]:
                self.addvec[classtype] += self.shorttext_to_embedvec(shorttext)
            self.addvec[classtype] /= np.linalg.norm(self.addvec[classtype])
        self.addvec = dict(self.addvec)
        self.trained = True

    def savemodel(self, nameprefix):
        if not self.trained:
            raise ModelNotTrainedException()
        pickle.dump(self.addvec, open(nameprefix+'_embedvecdict.pickle', 'w'))

    def loadmodel(self, nameprefix):
        self.addvec = pickle.load(open(nameprefix+'_embedvecdict.pickle', 'r'))
        self.trained = True

    def shorttext_to_embedvec(self, shorttext):
        vec = np.zeros(self.vecsize)
        tokens = word_tokenize(shorttext)
        for token in tokens:
            if token in self.wvmodel:
                vec += self.wvmodel[token]
        norm = np.linalg.norm(vec)
        if norm!=0:
            vec /= np.linalg.norm(vec)
        return vec

    def score(self, shorttext):
        if not self.trained:
            raise ModelNotTrainedException()
        vec = self.shorttext_to_embedvec(shorttext)
        scoredict = {}
        for classtype in self.addvec:
            try:
                scoredict[classtype] = 1 - cosine(vec, self.addvec[classtype])
            except ValueError:
                scoredict[classtype] = np.nan
        return scoredict