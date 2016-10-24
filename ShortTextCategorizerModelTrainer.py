import argparse
import os

from gensim.models import Word2Vec

from utils.classification_exceptions import Word2VecModelNotExistException, AlgorithmNotExistException
import classifiers.SumWord2VecClassification as sumwv
import classifiers.AutoencoderEmbedVecClassification as auto
import classifiers.CNNEmbedVecClassification as cnn
import data.data_retrieval as ret
from classifiers import allowed_algos

def get_argparser():
    argparser = argparse.ArgumentParser(description='Train short text categorization model.')
    argparser.add_argument('trainfile', help='Path of the training file, first column being the class label, and second column the text.')
    argparser.add_argument('output_nameprefix', help='Prefix of the path of output model.')
    argparser.add_argument('algo', help='Algorithm. (Options: sumword2vec (Sum of Embedded Vectors), autoencoder (Autoencoder of Embedded Vectors), cnn (Convolutional Neural Network on Embedded Vectors)')
    argparser.add_argument('wvmodel_path', help='Path of the pre-trained Word2Vec model.')
    argparser.add_argument('--ngram', type=int, default=2, help='n-gram, used in convolutional neural network only. (Default: 2)')
    argparser.add_argument('--final_activation', default='softmax',
                           help='activation function in final layer, used in convolutional neural network only. (Default=''softmax'')')
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
        classifier = sumwv.SumEmbeddedVecClassifier(wvmodel, classdict)
    elif args.algo=='autoencoder':
        classifier = auto.AutoEncoderWord2VecClassifier(wvmodel, classdict)
    elif args.algo=='cnn':
        classifier = cnn.CNNEmbeddedVecClassifier(wvmodel, classdict, n_gram=args.ngram, final_activation=args.final_activation)

    # train
    print "Training..."
    classifier.train()

    # save models
    print "Saving models..."
    classifier.savemodel(args.output_nameprefix)

    print "Done."