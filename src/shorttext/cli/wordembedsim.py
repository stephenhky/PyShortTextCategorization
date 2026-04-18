
import argparse
import time

from scipy.spatial.distance import cosine

from ..metrics.embedfuzzy import jaccardscore_sents
from ..utils import tokenize, load_word2vec_model, load_fasttext_model, load_poincare_model
from ..utils import shorttext_to_avgvec
from ..metrics.wasserstein import word_mover_distance
from ..metrics.dynprog.jaccard import soft_jaccard_score
from ..utils.compute import cosine_similarity

typedict = {
    'word2vec': load_word2vec_model,
    'fasttext': load_fasttext_model,
    'poincare': load_poincare_model
}


def getargparser():
    """Get argument parser for word embedding similarity CLI.

    Returns:
        ArgumentParser for command line arguments.
    """
    parser.add_argument('modelpath', help='Path of the Word2Vec model')
    parser.add_argument('--type', default='word2vec',
                        help='Type of word-embedding model (default: "word2vec"; other options: "fasttext", "poincare")')
    return parser


def main():
    # argument parsing
    args = getargparser().parse_args()

    # preload tokenizer
    tokenize('Mogu is cute.')

    time0 = time.time()
    print(f"Loading {args.type}   model: {args.modelpath}")
    wvmodel = typedict[args.type](args.modelpath)
    time1 = time.time()
    end = False
    print(f"... loading time: {time1 - time0} seconds")

    while not end:
        sent1 = input('sent1> ')
        if len(sent1)==0:
            end = True
        else:
            sent2 = input('sent2> ')

            # output results
            print(f"Cosine Similarity = {cosine_similarity(shorttext_to_avgvec(sent1, wvmodel), shorttext_to_avgvec(sent2, wvmodel)):.4f}")
            print(f"Word-embedding Jaccard Score Similarity = {jaccardscore_sents(sent1, sent2, wvmodel):.4f}")
            print(f"Word Mover's Distance = {word_mover_distance(tokenize(sent1), tokenize(sent2), wvmodel):.4f}")
            print(f"Soft Jaccard Score (edit distance) = {soft_jaccard_score(tokenize(sent1), tokenize(sent2)):.4f}")
