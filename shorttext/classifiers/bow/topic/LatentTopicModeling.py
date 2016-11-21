
from gensim.models import TfidfModel, LdaModel, LsiModel
from gensim.similarities import MatrixSimilarity
from nltk import word_tokenize
import numpy as np

from utils import textpreprocessing as textpreprocess
from utils import gensim_corpora as gc
import utils.classification_exceptions as e

topic_model_dict = {'lda': LdaModel,
                    'lsi': LsiModel}

class LatentTopicModeler:
    """
    This class facilitates the creation of topic models (two options: LDA (latent Dirichlet Allocation),
    and LSI (latent semantic indexing) with the given short text training data, and convert future
    short text into topic vectors using the trained topic model.
    """
    def __init__(self,
                 nb_topics,
                 preprocessor=textpreprocess.standard_text_preprocessor_1(),
                 algorithm='lda',
                 toweigh=True):
        """ Initialize the classifier.

        :param nb_topics: number of latent topics
        :param preprocessor: function that preprocesses the text. (Default: `utils.textpreprocess.standard_text_preprocessor_1`)
        :param algorithm: algorithm for topic modeling. Options: lda, lsi. (Default: lda)
        :param toweigh: whether to weigh the words using tf-idf. (Default: True)
        :type nb_topics: int
        :type preprocessor: function
        :type algorithm: str
        :type toweigh: bool
        """
        self.nb_topics = nb_topics
        self.preprocessor = preprocessor
        self.algorithm = algorithm
        self.toweigh = toweigh
        self.trained = False

    def train(self, classdict, *args, **kwargs):
        """ Train the classifier.

        :param classdict: training data
        :param args: arguments to pass to the `train` method for gensim topic models
        :param kwargs: arguments to pass to the `train` method for gensim topic models
        :return: None
        :type classdict: dict
        """
        self.dictionary, self.corpus, self.classlabels = gc.generate_gensim_corpora(classdict,
                                                                                    preprocess_and_tokenize=lambda sent: word_tokenize(self.preprocessor(sent)))
        if self.toweigh:
            self.tfidf = TfidfModel(self.corpus)
            normcorpus = self.tfidf[self.corpus]
        else:
            self.tfidf = None
            normcorpus = self.corpus

        self.topicmodel = topic_model_dict[self.algorithm](normcorpus,
                                                           num_topics=self.nb_topics,
                                                           *args,
                                                           **kwargs)
        self.matsim = MatrixSimilarity(self.topicmodel[normcorpus])

        # change the flag
        self.trained = True

    def retrieve_topicvec(self, shorttext, normalize=True):
        """ Calculate the topic vector representation of the short text.

        If neither :func:`~train` nor :func:`~loadmodel` was run, it will raise `ModelNotTrainedException`.

        :param shorttext: text to be represented
        :return: topic vector
        :raise: ModelNotTrainedException
        :type shorttext: str
        :rtype: numpy.ndarray
        """
        if not self.trained:
            raise e.ModelNotTrainedException()
        topicvec = np.zeros(self.nb_topics)
        bow = self.dictionary.doc2bow(word_tokenize(self.preprocessor(shorttext)))
        topicdist = self.topicmodel[self.tfidf[bow] if self.toweigh else bow]
        for topicid, frac in topicdist:
            topicvec[topicid] = frac
        if normalize:
            topicvec /= np.linalg.norm(topicvec)
        return topicvec

    def loadmodel(self, nameprefix):
        pass

    def savemodel(self, nameprefix):
        if not self.trained:
            raise e.ModelNotTrainedException()
        pass