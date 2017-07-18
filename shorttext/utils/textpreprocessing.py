from builtins import str
from builtins import map
from builtins import object
import re
import pickle
import os

import spacy
from stemming.porter import stem

# load stop words
this_dir, _ = os.path.split(__file__)
from io import open
stopwordset = pickle.load(open(os.path.join(this_dir, 'stopwordset.pkl'), 'rb')) #PYTHON 3 FIX: 'r' to 'rb'

# initialize spacy
class SpaCyNLPHolder(object):
    def __init__(self):
        self.nlp = None

    def getNLPInstance(self):
        if self.nlp==None:
            self.nlp = spacy.load('en')
        return self.nlp
# prepare the singleton
spaCyNLPHolder = SpaCyNLPHolder()

def spacy_tokenize(text):
    """ Tokenize a sentence with spaCy.

    This works like `nltk.tokenize` which tokenize a sentence, but this runs faster.
    This returns the strings of tokens.

    :param text: sentence to tokenize
    :return: list of tokens
    :type text: str
    :rtype: list
    """
    nlp = spaCyNLPHolder.getNLPInstance()   # lazy loading
    tokenizer = nlp(str(text))
    return list(map(str, [token for token in tokenizer]))

def preprocess_text(text, pipeline):
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
    if len(pipeline)==0:
        return text
    else:
        return preprocess_text(pipeline[0](text), pipeline[1:])

def text_preprocessor(pipeline):
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
    return lambda text: preprocess_text(text, pipeline)

def standard_text_preprocessor_1():
    """ Return a commonly used text preprocessor.

    Return a text preprocessor that is commonly used, with the following steps:

    - removing special characters,
    - removing numerals,
    - converting all alphabets to lower cases,
    - removing stop words, and
    - stemming the words (using Porter stemmer).

    This function calls :func:`~text_preprocessor`.

    :return: a function that preprocesses text according to the pipeline
    :rtype: function
    """
    pipeline = [lambda s: re.sub('[^\w\s]', '', s),
                lambda s: re.sub('[\d]', '', s),
                lambda s: s.lower(),
                lambda s: ' '.join([s for s in spacy_tokenize(s) if not (s in stopwordset)]),
                lambda s: ' '.join(map(stem, spacy_tokenize(s)))
               ]
    return text_preprocessor(pipeline)