
from collections import Counter
from typing import Optional, Any, Self, Annotated

import numpy as np
import numpy.typing as npt
import npdict
from os import PathLike

import sparse

from .classification_exceptions import UnequalArrayLengthsException
from .compactmodel_io import CompactIOMachine
from .textpreprocessing import advanced_text_tokenizer_1

npdtm_suffices = ["_npdict.npy"]


def _construct_sparse_coo_dtm_matrix(
        sorted_token_list: list[str],
        tokens_counters: list[list[tuple[str, int]]]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    token_index_map = {token: idx for idx, token in enumerate(sorted_token_list)}
    ids_counters = [
        {token_index_map[token]: counts for token, counts in counter}
        for counter in tokens_counters
    ]
    docs_nbtokens = [len(counter) for counter in ids_counters]
    nb_coo_data = sum(docs_nbtokens)
    coordx_array = np.empty(nb_coo_data, dtype=np.int64)
    coordy_array = np.empty(nb_coo_data, dtype=np.int64)
    val_array = np.empty(nb_coo_data)

    i = 0
    for doc_id, counter in enumerate(ids_counters):
        for tokenid, counts in counter.items():
            coordx_array[i] = doc_id
            coordy_array[i] = tokenid
            val_array[i] = counts
            i += 1

    return coordx_array, coordy_array, val_array


def generate_npdict_document_term_matrix(
        corpus: list[str],
        doc_ids: list[Any],
        tokenize_func: callable
) -> npdict.NumpyNDArrayWrappedDict:
    try:
        assert len(corpus) == len(doc_ids)
    except AssertionError:
        raise UnequalArrayLengthsException(corpus, doc_ids)

    # grabbing tokens from each document in the corpus
    doc_tokens = [tokenize_func(document) for document in corpus]
    tokens_set = set([
        token
        for document in doc_tokens
        for token in document
    ])
    sorted_tokens_list = sorted(list(tokens_set))
    tokens_counters = [dict(Counter(tokens)) for tokens in doc_tokens]
    tokens_counters_tuples = [[(token, counts) for token, counts in counter.items()] for counter in tokens_counters]
    coord_x, coord_y, data = _construct_sparse_coo_dtm_matrix(
        sorted_tokens_list, tokens_counters_tuples
    )
    npdtm = npdict.SparseArrayWrappedDict.from_sparsearray_given_keywords(
        [doc_ids, sorted_tokens_list],
        sparse.COO([coord_x, coord_y], data=data, shape=(len(doc_tokens), len(sorted_tokens_list)))
    )
    return npdtm


def convert_classdict_to_corpus(
        classdict: dict[str, list[str]],
        preprocess_func: callable
) -> tuple[list[str], list[str]]:
    corpus = [
        preprocess_func(datum)
        for doc_under_class in classdict.values()
        for datum in doc_under_class
    ]
    docids = [
        f"{label}-{i}"
        for label, doc_under_class in classdict.items()
        for i in range(len(doc_under_class))
    ]
    return corpus, docids


def convert_classdict_to_xy(
        classdict: dict[str, list[str]],
        labels2idx: dict[str, int],
        preprocess_func: callable,
        tokenize_func: callable
) -> tuple[npdict.NumpyNDArrayWrappedDict, Annotated[sparse.SparseArray, "2D Array"]]:
    nbdata = sum(len(data) for data in classdict.values())
    nblabels = len(labels2idx)

    # making x
    corpus, docids = convert_classdict_to_corpus(classdict, preprocess_func=preprocess_func)
    dtm_npdict_matrix = generate_npdict_document_term_matrix(corpus, docids, tokenize_func)

    # making y
    y = sparse.COO(
        [
            list(range(nbdata)),
            [
                labels2idx[label]
                for label, doc_under_class in classdict.items()
                for _ in doc_under_class
            ]
        ],
        [1.]*nbdata,
        shape=(nbdata, nblabels)
    )

    return dtm_npdict_matrix, y


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
        return npdtm.generate_dict(new_dtm_sparray, dense=not sparse)

    new_dtm_nparray = npdtm.to_numpy() * np.log(nbdocs / doc_frequencies)
    new_npdtm = npdtm.generate_dict(new_dtm_nparray)
    if sparse:
        return npdict.SparseArrayWrappedDict.from_NumpyNDArrayWrappedDict(
            new_npdtm, default_initial_value=0.0
        )
    else:
        return new_npdtm


class NumpyDocumentTermMatrix(CompactIOMachine):
    def __init__(
            self,
            corpus: Optional[list[str]]=None,
            docids: Optional[list[Any]]=None,
            tfidf: bool=False,
            tokenize_func: Optional[callable]=None
    ):
        super().__init__({'classifier': 'npdtm'}, 'npdtm', npdtm_suffices)
        self.tokenize_func = tokenize_func if tokenize_func is not None else advanced_text_tokenizer_1()

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
        if isinstance(self.npdtm, npdict.SparseArrayWrappedDict):
            matrix = self.npdtm.to_coo()
        else:
            matrix = self.npdtm.to_numpy()
        return np.sum(matrix[:, token_index])

    def get_doc_frequency(self, token) -> int:
        token_index = self.npdtm._keystrings_to_indices[1][token]
        if isinstance(self.npdtm, npdict.SparseArrayWrappedDict):
            freq_array = self.npdtm.to_coo()[:, token_index]
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

    def savemodel(self, nameprefix: str) -> None:
        self.npdtm.save(nameprefix+"_npdict.npy")

    def loadmodel(self, nameprefix: str) -> Self:
        self.npdtm = npdict.SparseArrayWrappedDict.load(nameprefix+"_npdict.npy")

    @property
    def docids(self) -> list[str]:
        return self.npdtm._lists_keystrings[0]

    @property
    def tokens(self) -> list[str]:
        return self.npdtm._lists_keystrings[1]

    @property
    def nbdocs(self) -> int:
        return len(self.docids)

    @property
    def nbtokens(self) -> int:
        return len(self.tokens)


def load_numpy_documentmatrixmatrix(filepath: str | PathLike) -> NumpyDocumentTermMatrix:
    npdtm = NumpyDocumentTermMatrix()
    npdtm.load_compact_model(filepath)
    return npdtm

