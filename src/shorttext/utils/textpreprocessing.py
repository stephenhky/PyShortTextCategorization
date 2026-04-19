
import re
import os
import codecs
from typing import TextIO
from functools import partial

import snowballstemmer


# tokenizer
def tokenize(s: str) -> list[str]:
    """Tokenize a string by splitting on whitespace.

    Args:
        s: Input string to tokenize.

    Returns:
        List of tokens split by whitespace.
    """
    return s.split(' ')


# stemmer
class StemmerSingleton:
    """Singleton class for Porter stemmer.

    Provides a singleton instance of the snowball stemmer for English.
    """

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(StemmerSingleton, cls).__new__(cls)
            cls.stemmer = snowballstemmer.stemmer('english')
        return cls.instance

    def __call__(cls, s: str) -> str:
        """Stem a word using Porter stemmer.

        Args:
            s: Word to stem.

        Returns:
            Stemmed word.
        """
        return cls.stemmer.stemWord(s)


def stemword(s: str) -> str:
    """Stem a word using Porter stemmer.

    Args:
        s: Word to stem.

    Returns:
        Stemmed word.
    """
    return StemmerSingleton()(s)


def preprocess_text(text: str, pipeline: list[callable]) -> str:
    """Preprocess text according to a given pipeline.

    Applies a sequence of preprocessing functions to the input text.
    Each function in the pipeline transforms the text (e.g., stemming,
    lemmatizing, removing punctuation).

    Args:
        text: Input text to preprocess.
        pipeline: List of functions that each transform a text string to another text string.

    Returns:
        The preprocessed text after applying all pipeline functions.
    """
    return text if len(pipeline)==0 else preprocess_text(pipeline[0](text), pipeline[1:])


def tokenize_text(
        text: str,
        presplit_pipeline: list[callable],
        primitize_tokenizer: callable,
        postsplit_pipeline: list[callable],
        stopwordsfile: TextIO
) -> list[str]:
    """Tokenize text with preprocessing pipelines.

    Applies pre-split and post-split pipelines to tokenize text,
    filtering out stopwords.

    Args:
        text: Input text to tokenize.
        presplit_pipeline: List of functions to apply before tokenization.
        primitize_tokenizer: Tokenizer function to split text into tokens.
        postsplit_pipeline: List of functions to apply to each token after tokenization.
        stopwordsfile: File containing stopwords to filter out.

    Returns:
        List of tokens after preprocessing and stopword filtering.
    """
    # load stop words file
    stopwordset = set([stopword.strip() for stopword in stopwordsfile])

    # done
    presplit_text = text
    for func in presplit_pipeline:
        presplit_text = func(presplit_text)
    postsplit_tokens = primitize_tokenizer(presplit_text)
    for func in postsplit_pipeline:
        for i, token in enumerate(postsplit_tokens):
            postsplit_tokens[i] = func(token)
    postsplit_tokens = [
        token for token in postsplit_tokens
        if token not in stopwordset
    ]
    return postsplit_tokens


def text_preprocessor(pipeline: list[callable]) -> callable:
    """Create a text preprocessor function from a pipeline.

    Returns a function that applies the given pipeline to preprocess text.
    This is a convenience function that wraps preprocess_text with
    a fixed pipeline.

    Args:
        pipeline: List of functions that transform text to text.

    Returns:
        A callable that takes text and returns preprocessed text.
    """
    return partial(preprocess_text, pipeline=pipeline)


def oldschool_standard_text_preprocessor(stopwordsfile: TextIO) -> callable:
    """Create a standard text preprocessor.

    Returns a text preprocessor with the following steps:
    - Remove special characters
    - Remove numerals
    - Convert to lowercase
    - Remove stop words
    - Stem words using Porter stemmer

    Args:
        stopwordsfile: File object containing stopwords to filter.

    Returns:
        A callable that takes text and returns preprocessed text.
    """
    # load stop words file
    stopwordset = set([stopword.strip() for stopword in stopwordsfile])
    stopwordsfile.close()

    # the pipeline
    pipeline = [lambda s: re.sub(r'[^\w\s]', '', s),
                lambda s: re.sub(r'[0-9]', '', s),
                lambda s: s.lower(),
                lambda s: ' '.join(filter(lambda s: not (s in stopwordset), tokenize(s))),
                lambda s: ' '.join([stemword(stemmed_token) for stemmed_token in tokenize(s)])
               ]
    return text_preprocessor(pipeline)


def standard_text_preprocessor_1() -> callable:
    """Create a standard text preprocessor using NLTK stopwords.

    Returns a text preprocessor with the following steps:
    - Remove special characters
    - Remove numerals
    - Convert to lowercase
    - Remove stop words (NLTK list)
    - Stem words using Porter stemmer

    Returns:
        A callable that takes text and returns preprocessed text.
    """
    # load stop words
    this_dir, _ = os.path.split(__file__)
    stopwordsfile = codecs.open(os.path.join(this_dir, 'stopwords.txt'), 'r', 'utf-8')

    return oldschool_standard_text_preprocessor(stopwordsfile)


def standard_text_preprocessor_2() -> callable:
    """Create a standard text preprocessor with negation-aware stopwords.

    Returns a text preprocessor with the following steps:
    - Remove special characters
    - Remove numerals
    - Convert to lowercase
    - Remove stop words (NLTK list minus negation terms)
    - Stem words using Porter stemmer

    Returns:
        A callable that takes text and returns preprocessed text.
    """
    # load stop words
    this_dir, _ = os.path.split(__file__)
    stopwordsfile = codecs.open(os.path.join(this_dir, 'nonneg_stopwords.txt'), 'r', 'utf-8')

    return oldschool_standard_text_preprocessor(stopwordsfile)


def advanced_text_tokenizer_1() -> callable:
    """Create an advanced text tokenizer.

    Returns a tokenizer function that applies preprocessing steps:
    - Remove special characters
    - Remove numerals
    - Convert to lowercase
    - Stem tokens using Porter stemmer
    - Filter out negation-aware stopwords

    Returns:
        A callable that takes text and returns a list of tokens.
    """
    presplit_pipeline = [
        lambda s: re.sub(r'[^\w\s]', '', s),
        lambda s: re.sub(r'[0-9]', '', s),
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
        primitize_tokenizer=tokenizer,
        postsplit_pipeline=postsplit_pipeline,
        stopwordsfile=codecs.open(os.path.join(this_dir, 'nonneg_stopwords.txt'), 'r', 'utf-8')
    )
