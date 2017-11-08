from . import kerasmodel_io
from . import classification_exceptions
from . import gensim_corpora
from . import textpreprocessing
from . import compactmodel_io

from .textpreprocessing import spacy_tokenize as tokenize
from .textpreprocessing import text_preprocessor, standard_text_preprocessor_1

from .wordembed import load_word2vec_model, load_fasttext_model, shorttext_to_avgembedvec

from .deprecation import deprecated
