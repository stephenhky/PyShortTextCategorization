from . import kerasmodel_io
from . import classification_exceptions
from . import gensim_corpora
from . import textpreprocessing
from .wordembed import load_word2vec_model
from . import compactmodel_io

from .textpreprocessing import spacy_tokenize as tokenize
from .textpreprocessing import text_preprocessor, standard_text_preprocessor_1

from .deprecation import deprecated
