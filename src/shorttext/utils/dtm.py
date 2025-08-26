
import pickle
from typing import Optional, Any
from types import FunctionType

import numpy as np
import npdict
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from scipy.sparse import dok_matrix
from deprecation import deprecated
from nptyping import NDArray, Shape, Int

from .compactmodel_io import CompactIOMachine
from .classification_exceptions import NotImplementedException
from .textpreprocessing import advanced_text_tokenizer_1


dtm_suffices = ['_docids.pkl', '_dictionary.dict', '_dtm.pkl']
npdtm_suffices = []


def generate_npdict_document_term_matrix(
        corpus: list[str],
        doc_ids: list[Any],
        tokenize_func: FunctionType
) -> npdict.NumpyNDArrayWrappedDict:
    # grabbing tokens from each document in the corpus
    doc_tokens = [tokenize_func(document) for document in corpus]
    tokens_set = set([
        token
        for document in doc_tokens
        for token in document
    ])
    npdtm = npdict.SparseArrayWrappedDict(
        [doc_ids, sorted(list(tokens_set))],
        default_initial_value=0.0
    )
    for doc_id, document in zip(doc_ids, doc_tokens):
        for token in document:
            npdtm[doc_id, token] += 1
    return npdtm


def compute_document_frequency(
        npdtm: npdict.NumpyNDArrayWrappedDict
) -> NDArray[Shape["*"], Int]:
    if isinstance(npdtm, npdict.SparseArrayWrappedDict):
        return np.sum(npdtm.to_coo() > 0, axis=0).todense()
    else:
        return np.sum(npdtm.to_numpy() > 0, axis=0)


def compute_tfidf_document_term_matrix(
        npdtm: npdict.NumpyNDArrayWrappedDict,
        sparse: bool=True
) -> npdict.NumpyNDArrayWrappedDict:
    doc_frequencies = compute_document_frequency(npdtm)
    nbdocs = npdtm.dimension_sizes[0]
    if isinstance(npdtm, npdict.SparseArrayWrappedDict):
        new_dtm_sparray = npdtm.to_coo() * np.log(nbdocs / doc_frequencies)
        return npdict.SparseArrayWrappedDict.generate_dict(new_dtm_sparray, dense=not sparse)
    else:
        new_dtm_nparray = npdtm.to_numpy() * np.log(nbdocs / doc_frequencies)
        if sparse:
            # TODO: update npdict package
            pass
        else:
            return npdict.NumpyNDArrayWrappedDict.generate_dict(new_dtm_nparray)


class NumpyDocumentTermMatrix(CompactIOMachine):
    def __init__(
            self,
            corpus: Optional[list[str]]=None,
            docids: Optional[list[Any]]=None,
            tfidf: bool=False,
            tokenize_func: Optional[FunctionType]=None
    ):
        CompactIOMachine.__init__(self, {'classifier': 'npdtm'}, 'dtm', dtm_suffices)
        self.tokenize_func = tokenize_func if tokenize_func is not None else advanced_text_tokenizer_1

        # generate DTM
        if corpus is not None:
            self.generate_dtm(corpus, docids=docids, tfidf=tfidf)

    def generate_dtm(
            self,
            corpus: list[str],
            docids: Optional[list[Any]]=None,
            tfidf: bool=False
    ):
        # wrangling document IDs
        if docids is None:
            doc_ids = [f"doc{i}" for i in range(len(corpus))]
        else:
            if len(docids) == len(corpus):
                doc_ids = docids
            elif len(docids) > len(corpus):
                doc_ids = docids[:len(corpus)]
            else:
                doc_ids = docids + [f"doc{i}" for i in range(len(docids), len(corpus))]

        self.npdtm = generate_npdict_document_term_matrix(corpus, doc_ids, self.tokenize_func)

        if tfidf:
            self.npdtm = compute_tfidf_document_term_matrix(self.npdtm, sparse=True)


@deprecated(deprecated_in="3.0.1", removed_in="4.0.0",
            details="Use `npdict` instead")
class DocumentTermMatrix(CompactIOMachine):
    """ Document-term matrix for corpus.

    This is a class that handles the document-term matrix (DTM). With a given corpus, users can
    retrieve term frequency, document frequency, and total term frequency. Weighing using tf-idf
    can be applied.
    """
    def __init__(self, corpus, docids=None, tfidf=False):
        """ Initialize the document-term matrix (DTM) class with a given corpus.

        If document IDs (docids) are given, it will be stored and output as approrpriate.
        If not, the documents are indexed by numbers.

        Users can choose to weigh by tf-idf. The default is not to weigh.

        The corpus has to be a list of lists, with each of the inside list contains all the tokens
        in each document.

        :param corpus: corpus.
        :param docids: list of designated document IDs. (Default: None)
        :param tfidf: whether to weigh using tf-idf. (Default: False)
        :type corpus: list
        :type docids: list
        :type tfidf: bool
        """
        CompactIOMachine.__init__(self, {'classifier': 'dtm'}, 'dtm', dtm_suffices)
        if docids is None:
            self.docid_dict = {i: i for i in range(len(corpus))}
            self.docids = [i for i in range(len(corpus))]
        else:
            if len(docids) == len(corpus):
                self.docid_dict = {docid: i for i, docid in enumerate(docids)}
                self.docids = docids
            elif len(docids) > len(corpus):
                self.docid_dict = {docid: i for i, docid in zip(range(len(corpus)), docids[:len(corpus)])}
                self.docids = docids[:len(corpus)]
            else:
                self.docid_dict = {docid: i for i, docid in enumerate(docids)}
                self.docid_dict = {i: i for i in range(len(docids), len(corpus))}
                self.docids = docids + [i for i in range(len(docids), len(corpus))]
        # generate DTM
        self.generate_dtm(corpus, tfidf=tfidf)

    def generate_dtm(self, corpus, tfidf=False):
        """ Generate the inside document-term matrix and other peripherical information
        objects. This is run when the class is instantiated.

        :param corpus: corpus.
        :param tfidf: whether to weigh using tf-idf. (Default: False)
        :return: None
        :type corpus: list
        :type tfidf: bool
        """
        self.dictionary = Dictionary(corpus)
        self.dtm = dok_matrix((len(corpus), len(self.dictionary)), dtype=np.float_)
        bow_corpus = [self.dictionary.doc2bow(doctokens) for doctokens in corpus]
        if tfidf:
            weighted_model = TfidfModel(bow_corpus)
            bow_corpus = weighted_model[bow_corpus]
        for docid in self.docids:
            for tokenid, count in bow_corpus[self.docid_dict[docid]]:
                self.dtm[self.docid_dict[docid], tokenid] = count

    def get_termfreq(self, docid, token):
        """ Retrieve the term frequency of a given token in a particular document.

        Given a token and a particular document ID, compute the term frequency for this
        token. If `tfidf` is set to `True` while instantiating the class, it returns the weighted
        term frequency.

        :param docid: document ID
        :param token: term or token
        :return: term frequency or weighted term frequency of the given token in this document (designated by docid)
        :type docid: any
        :type token: str
        :rtype: numpy.float
        """
        return self.dtm[self.docid_dict[docid], self.dictionary.token2id[token]]

    def get_total_termfreq(self, token):
        """ Retrieve the total occurrences of the given token.

        Compute the total occurrences of the term in all documents. If `tfidf` is set to `True`
        while instantiating the class, it returns the sum of weighted term frequency.

        :param token: term or token
        :return: total occurrences of the given token
        :type token: str
        :rtype: numpy.float
        """
        return sum(self.dtm[:, self.dictionary.token2id[token]].values())

    def get_doc_frequency(self, token):
        """ Retrieve the document frequency of the given token.

        Compute the document frequency of the given token, i.e., the number of documents
        that this token can be found.

        :param token: term or token
        :return: document frequency of the given token
        :type token: str
        :rtype: int
        """
        return len(self.dtm[:, self.dictionary.token2id[token]].values())

    def get_token_occurences(self, token):
        """ Retrieve the term frequencies of a given token in all documents.

        Compute the term frequencies of the given token for all the documents. If `tfidf` is
        set to be `True` while instantiating the class, it returns the weighted term frequencies.

        This method returns a dictionary of term frequencies with the corresponding document IDs
        as the keys.

        :param token: term or token
        :return: a dictionary of term frequencies with the corresponding document IDs as the keys
        :type token: str
        :rtype: dict
        """
        return {self.docids[docidx]: count for (docidx, _), count in self.dtm[:, self.dictionary.token2id[token]].items()}

    def get_doc_tokens(self, docid):
        """ Retrieve the term frequencies of all tokens in the given document.

        Compute the term frequencies of all tokens for the given document. If `tfidf` is
        set to be `True` while instantiating the class, it returns the weighted term frequencies.

        This method returns a dictionary of term frequencies with the tokens as the keys.

        :param docid: document ID
        :return: a dictionary of term frequencies with the tokens as the keys
        :type docid: any
        :rtype: dict
        """
        return {self.dictionary[tokenid]: count for (_, tokenid), count in self.dtm[self.docid_dict[docid], :].items()}

    def generate_dtm_dataframe(self):
        """ Generate the data frame of the document-term matrix. (shorttext <= 1.0.3)

        Now it raises exception.

        :return: data frame of the document-term matrix
        :rtype: pandas.DataFrame
        :raise: NotImplementedException
        """
        raise NotImplementedException()

    def savemodel(self, prefix):
        """ Save the model.

        :param prefix: prefix of the files
        :return: None
        :type prefix: str
        """
        pickle.dump(self.docids, open(prefix+'_docids.pkl', 'wb'))
        self.dictionary.save(prefix+'_dictionary.dict')
        pickle.dump(self.dtm, open(prefix+'_dtm.pkl', 'wb'))

    def loadmodel(self, prefix):
        """ Load the model.

        :param prefix: prefix of the files
        :return: None
        :type prefix: str
        """
        self.docids = pickle.load(open(prefix+'_docids.pkl', 'rb'))
        self.docid_dict = {docid: i for i, docid in enumerate(self.docids)}
        self.dictionary = Dictionary.load(prefix+'_dictionary.dict')
        self.dtm = pickle.load(open(prefix+'_dtm.pkl', 'rb'))


@deprecated(deprecated_in="3.0.1", removed_in="4.0.0",
            details="Use `npdict` instead")
def load_DocumentTermMatrix(filename, compact=True):
    """ Load presaved Document-Term Matrix (DTM).

    Given the file name (if `compact` is `True`) or the prefix (if `compact` is `False`),
    return the document-term matrix.

    :param filename: file name or prefix
    :param compact: whether it is a compact model. (Default: `True`)
    :return: document-term matrix
    :type filename: str
    :type compact: bool
    :rtype: DocumentTermMatrix
    """
    dtm = DocumentTermMatrix([[]])
    if compact:
        dtm.load_compact_model(filename)
    else:
        dtm.loadmodel(filename)
    return dtm