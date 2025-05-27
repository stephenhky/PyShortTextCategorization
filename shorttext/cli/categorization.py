
import os
from functools import partial
import argparse

from ..utils.compactmodel_io import get_model_classifier_name
from ..utils.classification_exceptions import AlgorithmNotExistException, WordEmbeddingModelNotExistException
from ..utils import load_word2vec_model, load_fasttext_model, load_poincare_model
from ..smartload import smartload_compact_model
from ..classifiers import TopicVectorCosineDistanceClassifier


allowed_classifiers = ['ldatopic', 'lsitopic', 'rptopic', 'kerasautoencoder', 'topic_sklearn',
                       'nnlibvec', 'sumvec', 'maxent']
needembedded_classifiers = ['nnlibvec', 'sumvec']
topicmodels = ['ldatopic', 'lsitopic', 'rptopic', 'kerasautoencoder']

load_word2vec_nonbinary_model = partial(load_word2vec_model, binary=False)
load_poincare_binary_model = partial(load_poincare_model, binary=True)

typedict = {'word2vec': load_word2vec_model,
            'word2vec_nonbinary': load_word2vec_nonbinary_model,
            'fasttext': load_fasttext_model,
            'poincare': load_poincare_model,
            'poincare_binary': load_poincare_binary_model}


def get_argparser():
    argparser = argparse.ArgumentParser(description='Perform prediction on short text with a given trained model.')
    argparser.add_argument('model_filepath', help='Path of the trained (compact) model.')
    argparser.add_argument('--wv', default='', help='Path of the pre-trained Word2Vec model. (None if not needed)')
    argparser.add_argument('--vecsize', default=300, type=int, help='Vector dimensions. (Default: 300)')
    argparser.add_argument('--topn', type=int, default=10, help='Number of top-scored results displayed. (Default: 10)')
    argparser.add_argument('--inputtext', default=None, help='single input text for classification. Run console if set to None. (Default: None)')
    argparser.add_argument('--type', default='word2vec',
                       choices=['word2vec', 'fasttext', 'poincare', 'word2vec_nonbinary', 'poincare_binary'],
                       help='Type of word-embedding model ...')

    return argparser

# main block
def main():
    # argument parsing
    args = get_argparser().parse_args()

    # check if the model file is given
    if not os.path.exists(args.model_filepath):
        raise IOError('Model file '+args.model_filepath+' not found!')

    # get the name of the classifier
    print('Retrieving classifier name...')
    classifier_name = get_model_classifier_name(args.model_filepath)
    if not (classifier_name in allowed_classifiers):
        raise AlgorithmNotExistException(classifier_name)

    # load the Word2Vec model if necessary
    wvmodel = None
    if classifier_name in needembedded_classifiers:
        # check if thw word embedding model is available
        if not os.path.exists(args.wv):
            raise WordEmbeddingModelNotExistException(args.wv)
        # if there, load it
        print('Loading word-embedding model: '+args.wv)
        wvmodel = typedict[args.type](args.wv)

    # load the classifier
    print('Initializing the classifier...')
    classifier = None
    if classifier_name in topicmodels:
        topicmodel = smartload_compact_model(args.model_filepath, wvmodel, vecsize=args.vecsize)
        classifier = TopicVectorCosineDistanceClassifier(topicmodel)
    else:
        classifier = smartload_compact_model(args.model_filepath, wvmodel, vecsize=args.vecsize)


    if args.inputtext != None:
        if len(args.inputtext) > 0:
            scoredict = classifier.score(args.inputtext)
            for label, score in sorted(scoredict.items(), key=lambda s: s[1], reverse=True)[:args.topn]:
                print(label, ' : ', score)
        else:
            print('No input text available!')
    else:
        # Console
        run = True
        while run:
            shorttext = input('text> ')
            if len(shorttext) > 0:
                scoredict = classifier.score(shorttext)
                for label, score in sorted(scoredict.items(), key=lambda s: s[1], reverse=True)[:args.topn]:
                    print(label+' : '+'%.4f' % (score))
            else:
                run = False

        print('Done.')
