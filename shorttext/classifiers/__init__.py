from utils import classification_exceptions
from utils import kerasmodel_io as kerasmodel_io

from .embed.autoencode import AutoencoderEmbedVecClassification
from .embed.nnlib import VarNNEmbedVecClassification, VarNNSumEmbedVecClassification
from .embed.nnlib import CNNEmbedVecClassification
from .embed.sumvec import SumWord2VecClassification
from .bow.topic import LatentTopicModeling

# allowed algorithms
allowed_algos = {'sumword2vec', 'autoencoder', 'vnn'}
