import argparse
import os

import classifiers.VarNNEmbedVecClassification as vnn
from gensim.models import Word2Vec

import classifiers.embed.autoencode.AutoencoderEmbedVecClassification as auto
import classifiers.embed.nnlib.CNNEmbedVecClassification as cnn
import classifiers.embed.sumvec.SumWord2VecClassification as sumwv
from classifiers import allowed_algos
from utils import Word2VecModelNotExistException, AlgorithmNotExistException


def get_argparser():
    argparser = argparse.ArgumentParser(description='Perform prediction on short text.')
    argparser.add_argument('input_nameprefix', help='Prefix of the path of input model.')
    argparser.add_argument('algo', help='Algorithm. (Options: sumword2vec (Sum of Embedded Vectors), autoencoder (Autoencoder of Embedded Vectors), cnn (Convolutional Neural Network on Embedded Vectors)')
    argparser.add_argument('wvmodel_path', help='Path of the pre-trained Word2Vec model.')
    argparser.add_argument('--ngram', type=int, default=2, help='n-gram, used in convolutional neural network only. (Default: 2)')
    argparser.add_argument('--test', action='store_true', default=False, help='Checked if the test input contains the label, and metrics will be output.')
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

    # load models
    print "Loading Word Embedding model..."
    wvmodel = Word2Vec.load_word2vec_format(args.wvmodel_path, binary=True)

    # initialize instance
    print "Instantiating classifier..."
    if args.algo=='vnn':
        classifier = vnn.VarNNEmbeddedVecClassifier(wvmodel)
    elif args.algo=='sumword2vec':
        classifier = sumwv.SumEmbeddedVecClassifier(wvmodel)
    elif args.algo=='autoencoder':
        classifier = auto.AutoEncoderWord2VecClassifier(wvmodel)
    elif args.algo=='cnn':
        classifier = cnn.CNNEmbeddedVecClassifier(wvmodel, n_gram=args.ngram)

    # load model
    print "Loading model..."
    classifier.loadmodel(args.input_nameprefix)

    # Console
    run = True
    while run:
        shorttext = raw_input('text> ')
        if len(shorttext) > 0:
            scoredict = classifier.score(shorttext)
            for label, score in sorted(scoredict.items(), key=lambda s: s[1], reverse=True):
                print label, ' : ', score
        else:
            run = False

    print "Done."
