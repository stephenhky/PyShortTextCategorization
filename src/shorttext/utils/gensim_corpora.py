
from collections import defaultdict

import gensim

from .textpreprocessing import tokenize


def generate_gensim_corpora(classdict, preprocess_and_tokenize=tokenize):
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
    classlabels = sorted(classdict.keys())
    doc = [preprocess_and_tokenize(' '.join(classdict[classlabel])) for classlabel in classlabels]
    dictionary = gensim.corpora.Dictionary(doc)
    corpus = [dictionary.doc2bow(doctokens) for doctokens in doc]
    return dictionary, corpus, classlabels


def save_corpus(dictionary, corpus, prefix):
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


def load_corpus(prefix):
    """ Load gensim corpus and dictionary.

    :param prefix: prefix of the file to load
    :return: corpus and dictionary
    :type prefix: str
    :rtype: tuple
    """
    corpus = gensim.corpora.MmCorpus(prefix+'_corpus.mm')
    dictionary = gensim.corpora.Dictionary.load(prefix+'_dictionary.dict')
    return corpus, dictionary


def update_corpus_labels(dictionary, corpus, newclassdict, preprocess_and_tokenize=tokenize):
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

    newdoc = [preprocess_and_tokenize(' '.join(newclassdict[classlabel])) for classlabel in sorted(newclassdict.keys())]
    newcorpus = [dictionary.doc2bow(doctokens) for doctokens in newdoc]
    corpus += newcorpus

    return corpus, newcorpus


def tokens_to_fracdict(tokens):
    """ Return normalized bag-of-words (BOW) vectors.

    :param tokens: list of tokens.
    :type tokens: list
    :return: normalized vectors of counts of tokens as a `dict`
    :rtype: dict
    """
    cntdict = defaultdict(lambda : 0)
    for token in tokens:
        cntdict[token] += 1
    totalcnt = sum(cntdict.values())
    return {token: float(cnt)/totalcnt for token, cnt in cntdict.items()}