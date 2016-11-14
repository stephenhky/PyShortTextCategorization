import argparse
import csv
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
    argparser.add_argument('input_file', help='Input file for prediction')
    argparser.add_argument('output_file', help='Output file')
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
    if not (os.path.exists(args.input_file)):
        raise IOError()

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

    # I/O
    print "Performing prediction..."
    inputfile = open(args.input_file, 'rb')
    outputfile = open(args.output_file, 'wb')
    reader = csv.reader(inputfile)
    writer = csv.writer(outputfile)
    header = reader.next()
    writer.writerow(header+['prediction', 'score'])
    numlines = 0
    numcorrects = 0
    for line in reader:
        text = line[1] if args.test else line[0]
        expert_label = line[0] if args.test else None
        scoredict = classifier.score(text)
        toplabel, topscore = max(scoredict.items(), key=lambda item: item[1])
        writer.writerow(line+[toplabel, topscore])
        if args.test:
            numlines += 1
            numcorrects += 1 if expert_label==toplabel else 0
    inputfile.close()
    outputfile.close()

    print "Done."

    if args.test:
        print 'Number of lines = ', numlines
        print 'Number of corrects = ', numcorrects
        print 'Accuracy = ', float(numcorrects)/numlines
