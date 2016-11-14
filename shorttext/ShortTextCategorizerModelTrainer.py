import argparse
import os

from gensim.models import Word2Vec

import classifiers.embed.autoencode.AutoencoderEmbedVecClassification as auto
import classifiers.embed.nnlib.CNNEmbedVecClassification as cnn
import classifiers.embed.sumvec.SumWord2VecClassification as sumwv
import data.data_retrieval as ret
from classifiers import allowed_algos
from utils.classification_exceptions import Word2VecModelNotExistException, AlgorithmNotExistException


def get_argparser():
    argparser = argparse.ArgumentParser(description='Train short text categorization model.')
    argparser.add_argument('trainfile', help='Path of the training file, first column being the class label, and second column the text.')
    argparser.add_argument('output_nameprefix', help='Prefix of the path of output model.')
    argparser.add_argument('algo', help='Algorithm. (Options: sumword2vec (Sum of Embedded Vectors), autoencoder (Autoencoder of Embedded Vectors), cnn (Convolutional Neural Network on Embedded Vectors)')
    argparser.add_argument('wvmodel_path', help='Path of the pre-trained Word2Vec model.')
    argparser.add_argument('--ngram', type=int, default=2, help='n-gram, used in convolutional neural network only. (Default: 2)')
    argparser.add_argument('--final_activation', default='softmax',
                           help='activation function in final layer, used in convolutional neural network only. (Default=''softmax'')')
    argparser.add_argument('--cnn_dropout_prob', type=float, default=0.0,
                           help='dropout probability in convolutional layer, used in convolutional neural network only. (Default: 0.0)')
    argparser.add_argument('--wl2regu', type=float, default=0.0,
                           help='regularization coefficients for weight L2 regularization, used in convolution neural network only. (Default: 0.0)')
    argparser.add_argument('--optimizer', default='adam',
                           help='optimizer for gradient descent (default: adam)')
    return argparser

if __name__ == '__main__':
    # parsing argument
    argparser = get_argparser()
    args = argparser.parse_args()

    # check validity
    if not os.path.exists(args.wvmodel_path):
        raise Word2VecModelNotExistException(args.wvmodel_path)
    if not (args.algo in allowed_algos):
        raise AlgorithmNotExistException(args.algo)
    if not os.path.exists(args.trainfile):
        raise IOError()

    # load models
    print "Loading Word Embedding model..."
    wvmodel = Word2Vec.load_word2vec_format(args.wvmodel_path, binary=True)
    classdict = ret.retrieve_data_as_dict(args.trainfile)

    # initialize instance
    print "Instantiating classifier..."
    if args.algo=='sumword2vec':
        classifier = sumwv.SumEmbeddedVecClassifier(wvmodel, classdict=classdict)
    elif args.algo=='autoencoder':
        classifier = auto.AutoEncoderWord2VecClassifier(wvmodel, classdict=classdict)
    elif args.algo=='cnn':
        classifier = cnn.CNNEmbeddedVecClassifier(wvmodel,
                                                  classdict=classdict,
                                                  n_gram=args.ngram,
                                                  final_activation=args.final_activation,
                                                  cnn_dropout=args.cnn_dropout_prob,
                                                  dense_wl2reg=args.wl2regu,
                                                  optimizer=args.optimizer)

    # train
    print "Training..."
    classifier.train()

    # save models
    print "Saving models..."
    classifier.savemodel(args.output_nameprefix)

    print "Done."