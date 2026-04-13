
from collections import Counter
from typing import Optional

import gensim
from deprecation import deprecated

from .textpreprocessing import tokenize


def generate_gensim_corpora(
        classdict: dict[str, list[str]],
        preprocess_and_tokenize: Optional[callable] = None
) -> tuple[gensim.corpora.Dictionary, list[list[tuple[int, int]]], list[str]]:
    """ Generate gensim bag-of-words dictionary and corpus.

    Given a text data, a dict with keys being the class labels, and the values
    being the list of short texts, in the same format output by `shorttext.data.data_retrieval`,
    return a gensim dictionary and corpus.

    :param classdict: text data, a dict with keys being the class labels, and each value is a list of short texts
    :param proprocess_and_tokenize: preprocessor function, that takes a short sentence, and return a list of tokens (Default: `shorttext.utils.tokenize`)
    :return: a tuple, consisting of a gensim dictionary, a corpus, and a list of class labels
    :type classdict: dict
    :type proprocess_and_tokenize: function
    :rtype: (gensim.corpora.Dictionary, list, list)
    """
    if preprocess_and_tokenize is None:
        preprocess_and_tokenize = tokenize

    classlabels = sorted(classdict.keys())
    doc = [preprocess_and_tokenize(' '.join(classdict[classlabel])) for classlabel in classlabels]
    dictionary = gensim.corpora.Dictionary(doc)
    corpus = [dictionary.doc2bow(doctokens) for doctokens in doc]
    return dictionary, corpus, classlabels


@deprecated(deprecated_in="5.0.0", removed_in="6.0.0")
def save_corpus(
        dictionary: gensim.corpora.Dictionary,
        corpus: list[list[tuple[int, int]]],
        prefix: str
) -> None:
    """ Save gensim corpus and dictionary.

    :param dictionary: dictionary to save
    :param corpus: corpus to save
    :param prefix: prefix of the files to save
    :return: None
    :type dictionary: gensim.corpora.Dictionary
    :type corpus: list
    :type prefix: str
    """
    dictionary.save(prefix+'_dictionary.dict')
    gensim.corpora.MmCorpus.serialize(prefix+'_corpus.mm', corpus)


@deprecated(deprecated_in="5.0.0", removed_in="6.0.0")
def load_corpus(prefix: str) -> tuple[gensim.corpora.MmCorpus, gensim.corpora.Dictionary]:
    """ Load gensim corpus and dictionary.

    :param prefix: prefix of the file to load
    :return: corpus and dictionary
    :type prefix: str
    :rtype: tuple
    """
    corpus = gensim.corpora.MmCorpus(prefix+'_corpus.mm')
    dictionary = gensim.corpora.Dictionary.load(prefix+'_dictionary.dict')
    return corpus, dictionary


def update_corpus_labels(
        dictionary: gensim.corpora.Dictionary,
        corpus: list[list[tuple[int, int]]],
        newclassdict: dict[str, list[str]],
        preprocess_and_tokenize: Optional[callable] = None
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    """ Update corpus with additional training data.
    
    With the additional training data, the dictionary and corpus are updated.
    
    :param dictionary: original dictionary
    :param corpus: original corpus
    :param newclassdict: additional training data
    :param preprocess_and_tokenize: preprocessor function, that takes a short sentence, and return a list of tokens (Default: `shorttext.utils.tokenize`)
    :return: a tuple, an updated corpus, and the new corpus (for updating model)
    :type dictionary: gensim.corpora.Dictionary
    :type corpus: list
    :type newclassdict: dict
    :type preprocess_and_tokenize: function
    :rtype: tuple
    """
    if preprocess_and_tokenize is None:
        preprocess_and_tokenize = tokenize

    newdoc = [preprocess_and_tokenize(' '.join(newclassdict[classlabel])) for classlabel in sorted(newclassdict.keys())]
    newcorpus = [dictionary.doc2bow(doctokens) for doctokens in newdoc]
    corpus += newcorpus

    return corpus, newcorpus


def tokens_to_fracdict(tokens: list[str]) -> dict[str, float]:
    """ Return normalized bag-of-words (BOW) vectors.

    :param tokens: list of tokens.
    :type tokens: list
    :return: normalized vectors of counts of tokens as a `dict`
    :rtype: dict
    """
    cntdict = Counter(tokens)
    totalcnt = sum(cntdict.values())
    return {token: cnt / totalcnt for token, cnt in cntdict.items()}
