
from .deprecation import deprecated

from . import misc
from . import kerasmodel_io
from . import classification_exceptions
from . import gensim_corpora
from . import textpreprocessing
from . import compactmodel_io
from . import dtm

from .textpreprocessing import spacy_tokenize as tokenize
from .textpreprocessing import text_preprocessor, standard_text_preprocessor_1

from .wordembed import load_word2vec_model, load_fasttext_model, load_poincare_model, shorttext_to_avgvec
from .dtm import load_DocumentTermMatrix
from .wordembed import shorttext_to_avgembedvec     # deprecated

from .dtm import DocumentTermMatrix, load_DocumentTermMatrix

