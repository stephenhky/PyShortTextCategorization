
import os
from functools import partial
import argparse
import logging
from operator import itemgetter

from ..utils.compactmodel_io import get_model_classifier_name
from ..utils.classification_exceptions import AlgorithmNotExistException, WordEmbeddingModelNotExistException
from ..utils import load_word2vec_model, load_fasttext_model, load_poincare_model
from ..smartload import smartload_compact_model
from ..classifiers import TopicVectorCosineDistanceClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

allowed_classifiers = [
    'ldatopic', 'lsitopic', 'rptopic', 'kerasautoencoder',
    'topic_sklearn', 'nnlibvec', 'sumvec', 'maxent'
]
needembedded_classifiers = ['nnlibvec', 'sumvec']
topicmodels = ['ldatopic', 'lsitopic', 'rptopic', 'kerasautoencoder']

load_word2vec_nonbinary_model = partial(load_word2vec_model, binary=False)
load_poincare_binary_model = partial(load_poincare_model, binary=True)

typedict = {
    'word2vec': load_word2vec_model,
    'word2vec_nonbinary': load_word2vec_nonbinary_model,
    'fasttext': load_fasttext_model,
    'poincare': load_poincare_model,
    'poincare_binary': load_poincare_binary_model
}


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Perform prediction on short text with a given trained model.'
    )
    parser.add_argument('model_filepath', help='Path of the trained (compact) model.')
    parser.add_argument('--wv', default='', help='Path of the pre-trained Word2Vec model.')
    parser.add_argument('--vecsize', default=300, type=int, help='Vector dimensions. (Default: 300)')
    parser.add_argument('--topn', type=int, default=10, help='Number of top results to show.')
    parser.add_argument('--inputtext', default=None, help='Single input text for classification. If omitted, will enter console mode.')
    parser.add_argument('--type', default='word2vec', choices=typedict.keys(),
                        help='Type of word-embedding model (default: word2vec)')
    return parser

# main block
def main():
    # argument parsing
    args = get_argparser().parse_args()

    # check if the model file is given
    if not os.path.exists(args.model_filepath):
        raise IOError(f'Model file "{args.model_filepath}" not found!')
    
    # get the name of the classifier
    logger.info('Retrieving classifier name...')
    classifier_name = get_model_classifier_name(args.model_filepath)

    if classifier_name not in allowed_classifiers:
        raise AlgorithmNotExistException(classifier_name)

    # load the Word2Vec model if necessary
    wvmodel = None
    if classifier_name in needembedded_classifiers:
        # check if the word embedding model is available
        if not os.path.exists(args.wv):
            raise WordEmbeddingModelNotExistException(args.wv)
        # if there, load it
        logger.info(f'Loading word-embedding model from {args.wv}...')
        wvmodel = typedict[args.type](args.wv)

    # load the classifier
    logger.info('Initializing the classifier...')
    if classifier_name in topicmodels:
        topicmodel = smartload_compact_model(args.model_filepath, wvmodel, vecsize=args.vecsize)
        classifier = TopicVectorCosineDistanceClassifier(topicmodel)
    else:
        classifier = smartload_compact_model(args.model_filepath, wvmodel, vecsize=args.vecsize)

    # predict single input or run in console mode
    if args.inputtext is not None:
        if len(args.inputtext.strip()) == 0:
            print('No input text provided.')
            return
        scoredict = classifier.score(args.inputtext)
        for label, score in sorted(scoredict.items(), key=itemgetter(1), reverse=True)[:args.topn]:
            print(f'{label} : {score:.4f}')
    else:
        # Console 
        print('Enter text to classify (empty input to quit):')
        while True:
            shorttext = input('text> ').strip()
            if not shorttext:
                break
            scoredict = classifier.score(shorttext)
            for label, score in sorted(scoredict.items(), key=itemgetter(1), reverse=True)[:args.topn]:
                print(f'{label} : {score:.4f}')
        print('Done.')

if __name__ == "__main__":
    main()
