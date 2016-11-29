from collections import defaultdict

from utils import textpreprocessing as textpreprocess
from classifiers.bow.topic.LatentTopicModeling import LatentTopicModeler

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
        scoredict = defaultdict(lambda : 0.0)
        similarities = self.topicmodeler.matsim[self.topicmodeler.retrieve_corpus_topicdist(shorttext)]
        for label, similarity in zip(self.topicmodeler.classlabels, similarities):
            scoredict[label] = similarity
        return dict(scoredict)

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

def train_topicvecCosineClassifier(classdict,
                                   nb_topics,
                                   preprocessor=textpreprocess.standard_text_preprocessor_1(),
                                   algorithm='lda',
                                   toweigh=True,
                                   normalize=True,
                                   *args, **kwargs):
    """ Return a cosine distance classifier, i.e., :class:`TopicVecCosineDistanceClassifier`, while
    training a topic model in between.

    Reference:

    Xuan Hieu Phan, Cam-Tu Nguyen, Dieu-Thu Le, Minh Le Nguyen, Susumu Horiguchi, Quang-Thuy Ha,
    "A Hidden Topic-Based Framework toward Building Applications with Short Web Documents,"
    *IEEE Trans. Knowl. Data Eng.* 23(7): 961-976 (2011).

    Xuan Hieu Phan, Le-Minh Nguyen, Susumu Horiguchi, "Learning to Classify Short and Sparse Text & Web withHidden Topics from Large-scale Data Collections,"
    WWW '08 Proceedings of the 17th international conference on World Wide Web. (2008) [`ACL
    <http://dl.acm.org/citation.cfm?id=1367510>`_]

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
    topicmodeler = LatentTopicModeler(preprocessor=preprocessor,
                                      algorithm=algorithm,
                                      toweigh=toweigh,
                                      normalize=normalize)
    topicmodeler.train(classdict, nb_topics, *args, **kwargs)

    # cosine distance classifier
    return TopicVecCosineDistanceClassifier(topicmodeler)

def load_topicvecCosineClassifier(nameprefix,
                                  preprocessor=textpreprocess.standard_text_preprocessor_1(),
                                  normalize=True):
    """ Load a topic model from files and return a cosine distance classifier.

    Given the prefix of the files of the topic model, return a cosine distance classifier
    based on this model, i.e., :class:`TopicVecCosineDistanceClassifier`.

    The files include a JSON (.json) file that specifies various parameters, a gensim dictionary (.gensimdict),
    and a topic model (.gensimmodel). If weighing is applied, load also the tf-idf model (.gensimtfidf).

    Reference:

    Xuan Hieu Phan, Cam-Tu Nguyen, Dieu-Thu Le, Minh Le Nguyen, Susumu Horiguchi, Quang-Thuy Ha,
    "A Hidden Topic-Based Framework toward Building Applications with Short Web Documents,"
    *IEEE Trans. Knowl. Data Eng.* 23(7): 961-976 (2011).

    Xuan Hieu Phan, Le-Minh Nguyen, Susumu Horiguchi, "Learning to Classify Short and Sparse Text & Web withHidden Topics from Large-scale Data Collections,"
    WWW '08 Proceedings of the 17th international conference on World Wide Web. (2008) [`ACL
    <http://dl.acm.org/citation.cfm?id=1367510>`_]

    :param nameprefix: prefix of the file paths
    :param preprocessor: function that preprocesses the text. (Default: `utils.textpreprocess.standard_text_preprocessor_1`)
    :param normalize: whether the retrieved topic vectors are normalized. (Default: True)
    :return: a classifier that scores the short text based on the topic model
    :type nameprefix: str
    :type preprocessor: function
    :type normalize: bool
    :rtype: TopicVecCosineDistanceClassifier
    """
    topicmodeler = LatentTopicModeler(preprocessor=preprocessor, normalize=normalize)
    topicmodeler.loadmodel(nameprefix)

    return TopicVecCosineDistanceClassifier(topicmodeler)

