import gensim
from nltk import word_tokenize

def generate_gensim_corpora(classdict, proprocess_and_tokenize=word_tokenize):
    """ Generate gensim bag-of-words dictionary and corpus.

    Given a text data, a dict with keys being the class labels, and the values
    being the list of short texts, in the same format output by `shorttext.data.data_retrieval`,
    return a gensim dictionary and corpus.

    :param classdict: text data, a dict with keys being the class labels, and each value is a list of short texts
    :param proprocess_and_tokenize: preprocessor function, that takes a short sentence, and return a list of tokens (Default: `nltk.word_tokenize`)
    :return: a tuple, consisting of a gensim dictionary and a corpus
    :type classdict: dict
    :type proprocess_and_tokenize: function,
    :rtype: (gensim.corpora.Dictionary, list)
    """
    doc = [proprocess_and_tokenize(classdict[classlabel]) for classlabel in sorted(classdict.keys())]
    dictionary = gensim.corpora.Dictionary(doc)
    corpus = [dictionary.doc2bow(doctokens) for doctokens in doc]
    return dictionary, corpus