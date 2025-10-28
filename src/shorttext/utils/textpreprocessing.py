
import re
import os
import codecs
from io import TextIOWrapper
from types import FunctionType
from functools import partial

import snowballstemmer


# tokenizer
def tokenize(s: str) -> list[str]:
    return s.split(' ')


# stemmer
class StemmerSingleton:
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(StemmerSingleton, cls).__new__(cls)
            cls.stemmer = snowballstemmer.stemmer('english')
        return cls.instance

    def __call__(cls, s: str) -> str:
        return cls.stemmer.stemWord(s)

def stemword(s: str) -> str:
    return StemmerSingleton()(s)


def preprocess_text(text: str, pipeline: list[FunctionType]) -> str:
    """ Preprocess the text according to the given pipeline.

    Given the pipeline, which is a list of functions that process an
    input text to another text (e.g., stemming, lemmatizing, removing punctuations etc.),
    preprocess the text.

    :param text: text to be preprocessed
    :param pipeline: a list of functions that convert a text to another text
    :return: preprocessed text
    :type text: str
    :type pipeline: list
    :rtype: str
    """
    return text if len(pipeline)==0 else preprocess_text(pipeline[0](text), pipeline[1:])


def tokenize_text(
        text: str,
        presplit_pipeline: list[FunctionType],
        primitize_tokenizer: FunctionType,
        prosplit_pipeline: list[FunctionType],
        stopwordsfile: TextIOWrapper
) -> list[str]:
    # load stop words file
    stopwordset = set([stopword.strip() for stopword in stopwordsfile])

    # done
    presplit_text = text
    for func in presplit_pipeline:
        presplit_text = func(presplit_text)
    postsplit_tokens = primitize_tokenizer(presplit_text)
    for func in prosplit_pipeline:
        for i, token in enumerate(postsplit_tokens):
            postsplit_tokens[i] = func(token)
    postsplit_tokens = [
        token for token in postsplit_tokens
        if token not in stopwordset
    ]
    return postsplit_tokens


def text_preprocessor(pipeline: list[FunctionType]) -> FunctionType:
    """ Return the function that preprocesses text according to the pipeline.

    Given the pipeline, which is a list of functions that process an
    input text to another text (e.g., stemming, lemmatizing, removing punctuations etc.),
    return a function that preprocesses an input text outlined by the pipeline, essentially
    a function that runs :func:`~preprocess_text` with the specified pipeline.

    :param pipeline: a list of functions that convert a text to another text
    :return: a function that preprocesses text according to the pipeline
    :type pipeline: list
    :rtype: function
    """
    return partial(preprocess_text, pipeline=pipeline)


def oldschool_standard_text_preprocessor(stopwordsfile: TextIOWrapper) -> FunctionType:
    """ Return a commonly used text preprocessor.

    Return a text preprocessor that is commonly used, with the following steps:

    - removing special characters,
    - removing numerals,
    - converting all alphabets to lower cases,
    - removing stop words, and
    - stemming the words (using Porter stemmer).

    This function calls :func:`~text_preprocessor`.

    :param stopwordsfile: file object of the list of stop words
    :type stopwordsfile: file
    :return: a function that preprocesses text according to the pipeline
    :rtype: function
    """
    # load stop words file
    stopwordset = set([stopword.strip() for stopword in stopwordsfile])
    stopwordsfile.close()

    # the pipeline
    pipeline = [lambda s: re.sub('[^\w\s]', '', s),
                lambda s: re.sub('[\d]', '', s),
                lambda s: s.lower(),
                lambda s: ' '.join(filter(lambda s: not (s in stopwordset), tokenize(s))),
                lambda s: ' '.join([stemword(stemmed_token) for stemmed_token in tokenize(s)])
               ]
    return text_preprocessor(pipeline)


def standard_text_preprocessor_1() -> FunctionType:
    """ Return a commonly used text preprocessor.

    Return a text preprocessor that is commonly used, with the following steps:

    - removing special characters,
    - removing numerals,
    - converting all alphabets to lower cases,
    - removing stop words (NLTK list), and
    - stemming the words (using Porter stemmer).

    This function calls :func:`~oldschool_standard_text_preprocessor`.

    :return: a function that preprocesses text according to the pipeline
    :rtype: function
    """
    # load stop words
    this_dir, _ = os.path.split(__file__)
    stopwordsfile = codecs.open(os.path.join(this_dir, 'stopwords.txt'), 'r', 'utf-8')

    return oldschool_standard_text_preprocessor(stopwordsfile)


def standard_text_preprocessor_2() -> FunctionType:
    """ Return a commonly used text preprocessor.

    Return a text preprocessor that is commonly used, with the following steps:

    - removing special characters,
    - removing numerals,
    - converting all alphabets to lower cases,
    - removing stop words (NLTK list minus negation terms), and
    - stemming the words (using Porter stemmer).

    This function calls :func:`~oldschool_standard_text_preprocessor`.

    :return: a function that preprocesses text according to the pipeline
    :rtype: function
    """
    # load stop words
    this_dir, _ = os.path.split(__file__)
    stopwordsfile = codecs.open(os.path.join(this_dir, 'nonneg_stopwords.txt'), 'r', 'utf-8')

    return oldschool_standard_text_preprocessor(stopwordsfile)


def advanced_text_tokenizer_1() -> FunctionType:
    presplit_pipeline = [
        lambda s: re.sub('[^\w\s]', '', s),
        lambda s: re.sub('[\d]', '', s),
        lambda s: s.lower()
    ]
    tokenizer = tokenize
    postsplit_pipeline = [
        lambda s: ' '.join([stemword(stemmed_token) for stemmed_token in tokenize(s)])
    ]
    this_dir, _ = os.path.split(__file__)
    return partial(
        tokenize_text,
        presplit_pipeline=presplit_pipeline,
        tokenizer=tokenizer,
        postsplit_pipeline=postsplit_pipeline,
        stopwordsfile=codecs.open(os.path.join(this_dir, 'nonneg_stopwords.txt'), 'r', 'utf-8')
    )
