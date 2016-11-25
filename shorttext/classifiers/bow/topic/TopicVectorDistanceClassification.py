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
        for label, similarity in zip(self.topicmodeler.classlabels, self.topicmodeler.matsim[self.topicmodeler[shorttext]]):
            scoredict[label] = similarity
        return dict(scoredict)

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
    topicmodeler = LatentTopicModeler(nb_topics,
                                      preprocessor=preprocessor,
                                      algorithm=algorithm,
                                      toweigh=toweigh,
                                      normalize=normalize)
    topicmodeler.train(classdict, *args, **kwargs)

    # cosine distance classifier
    return TopicVecCosineDistanceClassifier(topicmodeler)