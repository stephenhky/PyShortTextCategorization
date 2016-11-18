

from nltk import word_tokenize

from utils import textpreprocessing as textpreprocess
from utils import gensim_corpora as gc

class LatentTopicModeler:
    def __init__(self,
                 nb_topics,
                 preprocessor=textpreprocess.standard_text_preprocessor_1,
                 algorithm='lda',
                 tfidf=True):
        self.nb_topics = nb_topics
        self.preprocessor = preprocessor
        self.algorithm = algorithm
        self.tfidf = tfidf

    def train(self, classdict):
        self.dictionary, self.corpus, self.classlabels = gc.generate_gensim_corpora(classdict,
                                                                                    preprocess_and_tokenize=lambda sent: word_tokenize(self.preprocessor(sent)))

