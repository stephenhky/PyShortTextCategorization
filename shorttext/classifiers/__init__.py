from classifiers.embed.autoencode import AutoencoderEmbedVecClassification
from classifiers.embed.nnlib import CNNEmbedVecClassification
from classifiers.embed.sumvec import SumWord2VecClassification

# allowed algorithms
allowed_algos = {'sumword2vec', 'autoencoder', 'cnn', 'vnn'}