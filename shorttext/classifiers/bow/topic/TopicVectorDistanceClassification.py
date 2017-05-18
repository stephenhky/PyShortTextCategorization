
from shorttext.utils import textpreprocessing as textpreprocess
from .LatentTopicModeling import LatentTopicModeler, GensimTopicModeler
from .LatentTopicModeling import AutoencodingTopicModeler, load_autoencoder_topicmodel
from .LatentTopicModeling import load_gensimtopicmodel


class TopicVecCosineDistanceClassifier:
    """
    This is a class that implements a classifier that perform classification based on
    the cosine similarity between the topic vectors of the user-input short texts and various classes.
    The topic vectors are calculated using :class:`LatentTopicModeler`.
    """
    def __init__(self, topicmodeler):
        """ Initialize the classifier.

        :param topicmodeler: topic modeler
        :type topicmodeler: LatentTopicModeler
        """
        self.topicmodeler = topicmodeler

    def score(self, shorttext):
        """ Calculate the score, which is the cosine similarity with the topic vector of the model,
        of the short text against each class labels.

        :param shorttext: short text
        :return: dictionary of scores of the text to all classes
        :type shorttext: str
        :rtype: dict
        """
        # scoredict = defaultdict(lambda : 0.0)
        # similarities = self.topicmodeler.matsim[self.topicmodeler.retrieve_corpus_topicdist(shorttext)]
        # for label, similarity in zip(self.topicmodeler.classlabels, similarities):
        #     scoredict[label] = similarity
        # return dict(scoredict)
        return self.topicmodeler.get_batch_cos_similarities(shorttext)

    def loadmodel(self, nameprefix):
        """ Load the topic model with the given prefix of the file paths.

        Given the prefix of the file paths, load the corresponding topic model. The files
        include a JSON (.json) file that specifies various parameters, a gensim dictionary (.gensimdict),
        and a topic model (.gensimmodel). If weighing is applied, load also the tf-idf model (.gensimtfidf).

        This is essentialing loading the topic modeler :class:`LatentTopicModeler`.

        :param nameprefix: prefix of the file paths
        :return: None
        :type nameprefix: str
        """
        self.topicmodeler.loadmodel(nameprefix)

    def savemodel(self, nameprefix):
        """ Save the model with names according to the prefix.

        Given the prefix of the file paths, save the corresponding topic model. The files
        include a JSON (.json) file that specifies various parameters, a gensim dictionary (.gensimdict),
        and a topic model (.gensimmodel). If weighing is applied, load also the tf-idf model (.gensimtfidf).

        If neither :func:`~train` nor :func:`~loadmodel` was run, it will raise `ModelNotTrainedException`.

        This is essentialing saving the topic modeler :class:`LatentTopicModeler`.

        :param nameprefix: prefix of the file paths
        :return: None
        :raise: ModelNotTrainedException
        :type nameprefix: str
        """
        self.topicmodeler.savemodel(nameprefix)

    def load_compact_model(self, name):
        self.topicmodeler.load_compact_model(name)

    def save_compact_model(self, name):
        self.topicmodeler.save_compact_model(name)

def train_gensimtopicvec_cosineClassifier(classdict,
                                          nb_topics,
                                          preprocessor=textpreprocess.standard_text_preprocessor_1(),
                                          algorithm='lda',
                                          toweigh=True,
                                          normalize=True,
                                          *args, **kwargs):
    """ Return a cosine distance classifier, i.e., :class:`TopicVecCosineDistanceClassifier`, while
    training a gensim topic model in between.

    :param classdict: training data
    :param nb_topics: number of latent topics
    :param preprocessor: function that preprocesses the text. (Default: `utils.textpreprocess.standard_text_preprocessor_1`)
    :param algorithm: algorithm for topic modeling. Options: lda, lsi, rp. (Default: lda)
    :param toweigh: whether to weigh the words using tf-idf. (Default: True)
    :param normalize: whether the retrieved topic vectors are normalized. (Default: True)
    :param args: arguments to pass to the `train` method for gensim topic models
    :param kwargs: arguments to pass to the `train` method for gensim topic models
    :return: a classifier that scores the short text based on the topic model
    :type classdict: dict
    :type nb_topics: int
    :type preprocessor: function
    :type algorithm: str
    :type toweigh: bool
    :type normalize: bool
    :rtype: TopicVecCosineDistanceClassifier
    """
    # train topic model
    topicmodeler = GensimTopicModeler(preprocessor=preprocessor,
                                      algorithm=algorithm,
                                      toweigh=toweigh,
                                      normalize=normalize)
    topicmodeler.train(classdict, nb_topics, *args, **kwargs)

    # cosine distance classifier
    return TopicVecCosineDistanceClassifier(topicmodeler)

def load_gensimtopicvec_cosineClassifier(name,
                                         preprocessor=textpreprocess.standard_text_preprocessor_1(),
                                         compact=True):
    """ Load a gensim topic model from files and return a cosine distance classifier.

    Given the prefix of the files of the topic model, return a cosine distance classifier
    based on this model, i.e., :class:`TopicVecCosineDistanceClassifier`.

    The files include a JSON (.json) file that specifies various parameters, a gensim dictionary (.gensimdict),
    and a topic model (.gensimmodel). If weighing is applied, load also the tf-idf model (.gensimtfidf).

    :param name: name (if compact=True) or prefix (if compact=False) of the file paths
    :param preprocessor: function that preprocesses the text. (Default: `utils.textpreprocess.standard_text_preprocessor_1`)
    :param compact: whether model file is compact (Default: True)
    :return: a classifier that scores the short text based on the topic model
    :type name: str
    :type preprocessor: function
    :type compact: bool
    :rtype: TopicVecCosineDistanceClassifier
    """
    topicmodeler = load_gensimtopicmodel(name, preprocessor=preprocessor, compact=compact)
    return TopicVecCosineDistanceClassifier(topicmodeler)

def train_autoencoder_cosineClassifier(classdict,
                                       nb_topics,
                                       preprocessor=textpreprocess.standard_text_preprocessor_1(),
                                       normalize=True,
                                       *args, **kwargs):
    """ Return a cosine distance classifier, i.e., :class:`TopicVecCosineDistanceClassifier`, while
    training an autoencoder as a topic model in between.

    :param classdict: training data
    :param nb_topics: number of topics, i.e., number of encoding dimensions
    :param preprocessor: function that preprocesses the text. (Default: `utils.textpreprocess.standard_text_preprocessor_1`)
    :param normalize: whether the retrieved topic vectors are normalized. (Default: True)
    :param args: arguments to be passed to keras model fitting
    :param kwargs: arguments to be passed to keras model fitting
    :return: a classifier that scores the short text based on the autoencoder
    :type classdict: dict
    :type nb_topics: int
    :type preprocessor: function
    :type normalize: bool
    :rtype: TopicVecCosineDistanceClassifier
    """
    # train the autoencoder
    autoencoder = AutoencodingTopicModeler(preprocessor=preprocessor, normalize=normalize)
    autoencoder.train(classdict, nb_topics, *args, **kwargs)

    # cosine distance classifier
    return TopicVecCosineDistanceClassifier(autoencoder)

def load_autoencoder_cosineClassifier(name,
                                      preprocessor=textpreprocess.standard_text_preprocessor_1(),
                                      compact=True):
    """ Load an autoencoder from files for topic modeling, and return a cosine classifier.

    Given the prefix of the file paths, load the model into files, with name given by the prefix.
    There are files with names ending with "_encoder.json" and "_encoder.h5", which are
    the JSON and HDF5 files for the encoder respectively.
    They also include a gensim dictionary (.gensimdict).

    :param name: name (if compact=True) or prefix (if compact=False) of the file paths
    :param preprocessor: function that preprocesses the text. (Default: `utils.textpreprocess.standard_text_preprocessor_1`)
    :param compact: whether model file is compact (Default: True)
    :return: a classifier that scores the short text based on the autoencoder
    :type name: str
    :type preprocessor: function
    :type compact: bool
    :rtype: TopicVecCosineDistanceClassifier
    """
    autoencoder = load_autoencoder_topicmodel(name, preprocessor=preprocessor, compact=compact)
    return TopicVecCosineDistanceClassifier(autoencoder)