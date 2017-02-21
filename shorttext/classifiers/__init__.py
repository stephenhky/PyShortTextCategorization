from .embed import *
from .embed import SumEmbeddedVecClassifier, load_sumword2vec_classifier
from .embed import VarNNEmbeddedVecClassifier
from .embed import frameworks
from .embed.sumvec import frameworks as sumvecframeworks

from .bow.topic.LatentTopicModeling import GensimTopicModeler, AutoencodingTopicModeler
from .bow.topic.LatentTopicModeling import load_gensimtopicmodel, load_autoencoder_topic

from .bow.topic.TopicVectorDistanceClassification import TopicVecCosineDistanceClassifier as TopicVectorCosineDistanceClassifier
from .bow.topic.TopicVectorDistanceClassification import train_autoencoder_cosineClassifier, train_gensimtopicvec_cosineClassifier
from .bow.topic.TopicVectorDistanceClassification import load_autoencoder_cosineClassifier, load_gensimtopicvec_cosineClassifier

from .bow.topic.SkLearnClassification import TopicVectorSkLearnClassifier
from .bow.topic.SkLearnClassification import train_gensim_topicvec_sklearnclassifier, train_autoencoder_topic_sklearnclassifier
from .bow.topic.SkLearnClassification import load_gensim_topicvec_sklearnclassifier, load_autoencoder_topic_sklearnclassifier

# allowed algorithms
allowed_algos = {'sumword2vec', 'vnn'}
