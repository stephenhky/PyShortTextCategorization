
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
    """Construct sparse COO matrix for document-term matrix.

    Args:
        sorted_token_list: Sorted list of tokens.
        tokens_counters: List of token counters for each document.

    Returns:
        Tuple of (x_coords, y_coords, data) for sparse COO matrix.
    """
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
    """Generate document-term matrix as numpy dict.

    Args:
        corpus: List of documents.
        doc_ids: List of document IDs.
        tokenize_func: Tokenization function.

    Returns:
        NumpyNDArrayWrappedDict containing the document-term matrix.

    Raises:
        UnequalArrayLengthsException: If corpus and doc_ids have different lengths.
    """
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
    """Convert class dictionary to corpus and document IDs.

    Args:
        classdict: Training data with class labels as keys and texts as values.
        preprocess_func: Text preprocessing function.

    Returns:
        Tuple of (corpus, doc_ids).
    """
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
    """Convert class dictionary to feature matrix and labels.

    Args:
        classdict: Training data.
        labels2idx: Mapping from labels to indices.
        preprocess_func: Text preprocessing function.
        tokenize_func: Tokenization function.

    Returns:
        Tuple of (document-term matrix, label matrix).
    """
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
    """Compute document frequency for each token.

    Args:
        npdtm: Document-term matrix.

    Returns:
        Array of document frequencies for each token.
    """
    if isinstance(npdtm, npdict.SparseArrayWrappedDict):
        return np.sum(npdtm.to_coo() > 0, axis=0).todense()
    else:
        return np.sum(npdtm.to_numpy() > 0, axis=0)


def compute_tfidf_document_term_matrix(
        npdtm: npdict.NumpyNDArrayWrappedDict,
        sparse: bool=True
) -> npdict.NumpyNDArrayWrappedDict:
    """Compute TF-IDF weighted document-term matrix.

    Args:
        npdtm: Document-term matrix.
        sparse: Whether to return sparse format. Default: True.

    Returns:
        TF-IDF weighted document-term matrix.
    """
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
    """Document-term matrix using numpy dict.

    Provides an interface for working with document-term matrices
    with compact model I/O support.
    """

    def __init__(
            self,
            corpus: Optional[list[str]]=None,
            docids: Optional[list[Any]]=None,
            tfidf: bool=False,
            tokenize_func: Optional[callable]=None
    ):
        """Initialize the document-term matrix.

        Args:
            corpus: List of documents.
            docids: List of document IDs.
            tfidf: Whether to apply TF-IDF weighting. Default: False.
            tokenize_func: Tokenization function. Default: advanced_text_tokenizer_1.
        """
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
        """Generate document-term matrix from corpus.

        Args:
            corpus: List of documents.
            docids: List of document IDs.
            tfidf: Whether to apply TF-IDF weighting. Default: False.
        """
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
        """Get term frequency for a document and token.

        Args:
            docid: Document ID.
            token: Token.

        Returns:
            Term frequency.
        """
        return self.npdtm[docid, token]

    def get_total_termfreq(self, token: str) -> float:
        """Get total frequency of a token across all documents.

        Args:
            token: Token.

        Returns:
            Total term frequency.
        """
        token_index = self.npdtm._keystrings_to_indices[1][token]
        if isinstance(self.npdtm, npdict.SparseArrayWrappedDict):
            matrix = self.npdtm.to_coo()
        else:
            matrix = self.npdtm.to_numpy()
        return np.sum(matrix[:, token_index])

    def get_doc_frequency(self, token) -> int:
        """Get document frequency of a token.

        Args:
            token: Token.

        Returns:
            Number of documents containing the token.
        """
        token_index = self.npdtm._keystrings_to_indices[1][token]
        if isinstance(self.npdtm, npdict.SparseArrayWrappedDict):
            freq_array = self.npdtm.to_coo()[:, token_index]
        else:
            freq_array = self.npdtm.to_numpy()[:, token_index]
        return np.sum(freq_array > 0, axis=0)

    def get_token_occurences(self, token: str) -> dict[str, float]:
        """Get token occurrences across all documents.

        Args:
            token: Token.

        Returns:
            Dictionary mapping document IDs to term frequencies.
        """
        return {
            docid: self.npdtm[docid, token]
            for docid in self.npdtm._lists_keystrings[0]
        }

    def get_doc_tokens(self, docid: str) -> dict[str, float]:
        """Get tokens for a specific document.

        Args:
            docid: Document ID.

        Returns:
            Dictionary mapping tokens to frequencies.
        """
        return {
            token: self.npdtm[docid, token]
            for token in self.npdtm._lists_keystrings[1]
        }

    def savemodel(self, nameprefix: str) -> None:
        """Save the document-term matrix.

        Args:
            nameprefix: Prefix for output file.
        """
        self.npdtm.save(nameprefix+"_npdict.npy")

    def loadmodel(self, nameprefix: str) -> Self:
        """Load the document-term matrix.

        Args:
            nameprefix: Prefix for input file.
        """
        self.npdtm = npdict.SparseArrayWrappedDict.load(nameprefix+"_npdict.npy")

    @property
    def docids(self) -> list[str]:
        """List of document IDs."""
        return self.npdtm._lists_keystrings[0]

    @property
    def tokens(self) -> list[str]:
        """List of tokens."""
        return self.npdtm._lists_keystrings[1]

    @property
    def nbdocs(self) -> int:
        """Number of documents."""
        return len(self.docids)

    @property
    def nbtokens(self) -> int:
        """Number of unique tokens."""
        return len(self.tokens)


def load_numpy_documentmatrixmatrix(filepath: str | PathLike) -> NumpyDocumentTermMatrix:
    """Load a document-term matrix from a compact file.

    Args:
        filepath: Path to the compact model file.

    Returns:
        NumpyDocumentTermMatrix instance.
    """
    npdtm = NumpyDocumentTermMatrix()
    npdtm.load_compact_model(filepath)
    return npdtm

