
from typing import Optional, Any
from types import FunctionType

import numpy as np
import numpy.typing as npt
import npdict
from npdict import SparseArrayWrappedDict

from .compactmodel_io import CompactIOMachine
from .textpreprocessing import advanced_text_tokenizer_1


dtm_suffices = ['_docids.pkl', '_dictionary.dict', '_dtm.pkl']
npdtm_suffices = []


def generate_npdict_document_term_matrix(
        corpus: list[str],
        doc_ids: list[Any],
        tokenize_func: callable
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
) -> npt.NDArray[np.int32]:
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
        new_npdtm = npdict.NumpyNDArrayWrappedDict.generate_dict(new_dtm_nparray)
        if sparse:
            new_sparse_dtm = npdict.SparseArrayWrappedDict.from_NumpyNDArrayWrappedDict(
                new_npdtm, default_initial_value=0.0
            )
            return new_sparse_dtm
        else:
            return new_npdtm


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
    ) -> None:
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

    def get_termfreq(self, docid: str, token: str) -> float:
        return self.npdtm[docid, token]

    def get_total_termfreq(self, token: str) -> float:
        token_index = self.npdtm._keystrings_to_indices[1][token]
        if isinstance(self.npdtm, SparseArrayWrappedDict):
            matrix = self.npdtm.to_coo()
        else:
            matrix = self.npdtm.to_numpy()
        return np.sum(matrix[:, token_index])

    def get_doc_frequency(self, token) -> int:
        token_index = self.npdtm._keystrings_to_indices[1][token]
        if isinstance(self.npdtm, npdict.SparseArrayWrappedDict):
            freq_array = self.npdtm.to_coo()[:, token_index]
            return np.sum(freq_array > 0, axis=0).todense()
        else:
            freq_array = self.npdtm.to_numpy()[:, token_index]
            return np.sum(freq_array > 0, axis=0)

    def get_token_occurences(self, token: str) -> dict[str, float]:
        return {
            docid: self.npdtm[docid, token]
            for docid in self.npdtm._lists_keystrings[0]
        }

    def get_doc_tokens(self, docid: str) -> dict[str, float]:
        return {
            token: self.npdtm[docid, token]
            for token in self.npdtm._lists_keystrings[1]
        }
