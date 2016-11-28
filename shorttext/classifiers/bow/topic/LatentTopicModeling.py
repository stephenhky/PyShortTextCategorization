
import json

from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel, LsiModel, HdpModel, RpModel
from gensim.similarities import MatrixSimilarity
from nltk import word_tokenize
import numpy as np

from utils import textpreprocessing as textpreprocess
from utils import gensim_corpora as gc
import utils.classification_exceptions as e

topic_model_dict = {'lda': LdaModel,
                    'lsi': LsiModel,
                    'rp': RpModel}

class LatentTopicModeler:
    """
    This class facilitates the creation of topic models (options: LDA (latent Dirichlet Allocation),
    LSI (latent semantic indexing), and Random Projections
    with the given short text training data, and convert future
    short text into topic vectors using the trained topic model.
    """
    def __init__(self,
                 preprocessor=textpreprocess.standard_text_preprocessor_1(),
                 algorithm='lda',
                 toweigh=True,
                 normalize=True):
        """ Initialize the classifier.

        :param preprocessor: function that preprocesses the text. (Default: `utils.textpreprocess.standard_text_preprocessor_1`)
        :param algorithm: algorithm for topic modeling. Options: lda, lsi, rp. (Default: lda)
        :param toweigh: whether to weigh the words using tf-idf. (Default: True)
        :param normalize: whether the retrieved topic vectors are normalized. (Default: True)
        :type preprocessor: function
        :type algorithm: str
        :type toweigh: bool
        """
        self.preprocessor = preprocessor
        self.algorithm = algorithm
        self.toweigh = toweigh
        self.normalize = normalize
        self.trained = False

    def train(self, classdict, nb_topics, *args, **kwargs):
        """ Train the classifier.

        :param classdict: training data
        :param nb_topics: number of latent topics
        :param args: arguments to pass to the `train` method for gensim topic models
        :param kwargs: arguments to pass to the `train` method for gensim topic models
        :return: None
        :type classdict: dict
        :type nb_topics: int
        """
        self.nb_topics = nb_topics
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

    def retrieve_topicvec(self, shorttext):
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
        if self.normalize:
            topicvec /= np.linalg.norm(topicvec)
        return topicvec

    def __getitem__(self, shorttext):
        return self.retrieve_topicvec(shorttext)

    def __contains__(self, shorttext):
        if not self.trained:
            raise e.ModelNotTrainedException()
        return True

    def loadmodel(self, nameprefix):
        """ Load the topic model with the given prefix of the file paths.

        Given the prefix of the file paths, load the corresponding topic model. The files
        include a JSON (.json) file that specifies various parameters, a gensim dictionary (.gensimdict),
        and a topic model (.gensimmodel). If weighing is applied, load also the tf-idf model (.gensimtfidf).

        :param nameprefix: prefix of the file paths
        :return: None
        :type nameprefix: str
        """
        # load the JSON file (parameters)
        parameters = json.load(open(nameprefix+'.json', 'rb'))
        self.nb_topics = parameters['nb_topics']
        self.toweigh = parameters['toweigh']
        self.algorithm = parameters['algorithm']
        self.classlabels = parameters['classlabels']

        # load the dictionary
        self.dictionary = Dictionary.load(nameprefix+'.gensimdict')

        # load the topic model
        self.topicmodel = topic_model_dict[self.algorithm].load(nameprefix+'.gensimmodel')

        # load the similarity matrix
        self.matsim = MatrixSimilarity.load(nameprefix+'.gensimmat')

        # load the tf-idf modek
        if self.toweigh:
            self.tfidf = TfidfModel.load(nameprefix+'.gensimtfidf')

        # flag
        self.trained = True

    def savemodel(self, nameprefix):
        """ Save the model with names according to the prefix.

        Given the prefix of the file paths, save the corresponding topic model. The files
        include a JSON (.json) file that specifies various parameters, a gensim dictionary (.gensimdict),
        and a topic model (.gensimmodel). If weighing is applied, load also the tf-idf model (.gensimtfidf).

        If neither :func:`~train` nor :func:`~loadmodel` was run, it will raise `ModelNotTrainedException`.

        :param nameprefix: prefix of the file paths
        :return: None
        :raise: ModelNotTrainedException
        :type nameprefix: str
        """
        if not self.trained:
            raise e.ModelNotTrainedException()
        parameters = {}
        parameters['nb_topics'] = self.nb_topics
        parameters['toweigh'] = self.toweigh
        parameters['algorithm'] = self.algorithm
        parameters['classlabels'] = self.classlabels
        json.dump(parameters, open(nameprefix+'.json', 'wb'))

        self.dictionary.save(nameprefix+'.gensimdict')
        self.topicmodel.save(nameprefix+'.gensimmodel')
        self.matsim.save(nameprefix+'.gensimmat')
        if self.toweigh:
            self.tfidf.save(nameprefix+'.gensimtfidf')

def load_topicmodel(nameprefix,
                    preprocessor=textpreprocess.standard_text_preprocessor_1()):
    topicmodeler = LatentTopicModeler(preprocessor=preprocessor)
    topicmodeler.loadmodel(nameprefix)
    return topicmodeler