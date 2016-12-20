from classifiers.embed.sumvec import VarNNSumEmbedVecClassification
from .bow.topic import LatentTopicModeling
from .embed.nnlib import CNNEmbedVecClassification
from .embed.nnlib import VarNNEmbedVecClassification
from .embed.sumvec import SumEmbedVecClassification

# allowed algorithms
allowed_algos = {'sumword2vec', 'autoencoder', 'vnn'}
