from ..utils import classification_exceptions
from ..utils import kerasmodel_io as kerasmodel_io

from .embed.autoencode import AutoencoderEmbedVecClassification
from .embed.nnlib import CNNEmbedVecClassification
from .embed.sumvec import SumWord2VecClassification

# allowed algorithms
allowed_algos = {'sumword2vec', 'autoencoder', 'cnn', 'vnn'}
