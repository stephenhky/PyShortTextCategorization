
from collections import Counter
from typing import Optional

import gensim
from deprecation import deprecated

from .textpreprocessing import tokenize


def generate_gensim_corpora(
        classdict: dict[str, list[str]],
        preprocess_and_tokenize: Optional[callable] = None
) -> tuple[gensim.corpora.Dictionary, list[list[tuple[int, int]]], list[str]]:
    """Generate gensim dictionary and corpus from training data.

    Args:
        classdict: Training data with class labels as keys and lists of texts as values.
        preprocess_and_tokenize: Function to preprocess and tokenize text. Default: tokenize.

    Returns:
        Tuple of (dictionary, corpus, class_labels).
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
    """Save gensim corpus and dictionary to files.

    Args:
        dictionary: Dictionary to save.
        corpus: Corpus to save.
        prefix: Prefix for output files.

    Note:
        Deprecated since 5.0.0, will be removed in 6.0.0.
    """
    dictionary.save(prefix+'_dictionary.dict')
    gensim.corpora.MmCorpus.serialize(prefix+'_corpus.mm', corpus)


@deprecated(deprecated_in="5.0.0", removed_in="6.0.0")
def load_corpus(prefix: str) -> tuple[gensim.corpora.MmCorpus, gensim.corpora.Dictionary]:
    """Load gensim corpus and dictionary from files.

    Args:
        prefix: Prefix of files to load.

    Returns:
        Tuple of (corpus, dictionary).

    Note:
        Deprecated since 5.0.0, will be removed in 6.0.0.
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
    """Update corpus with additional training data.

    Args:
        dictionary: Existing dictionary.
        corpus: Existing corpus.
        newclassdict: Additional training data.
        preprocess_and_tokenize: Function to preprocess text. Default: tokenize.

    Returns:
        Tuple of (updated_corpus, new_corpus).
    """
    if preprocess_and_tokenize is None:
        preprocess_and_tokenize = tokenize

    newdoc = [preprocess_and_tokenize(' '.join(newclassdict[classlabel])) for classlabel in sorted(newclassdict.keys())]
    newcorpus = [dictionary.doc2bow(doctokens) for doctokens in newdoc]
    corpus += newcorpus

    return corpus, newcorpus


def tokens_to_fracdict(tokens: list[str]) -> dict[str, float]:
    """Convert tokens to normalized frequency dictionary.

    Args:
        tokens: List of tokens.

    Returns:
        Dictionary with tokens as keys and normalized frequencies as values.
    """
    cntdict = Counter(tokens)
    totalcnt = sum(cntdict.values())
    return {token: cnt / totalcnt for token, cnt in cntdict.items()}
