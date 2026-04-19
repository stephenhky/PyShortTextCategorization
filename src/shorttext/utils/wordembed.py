
from os import PathLike
from typing import Any, Annotated, Optional, TextIO

import numpy as np
import numpy.typing as npt
import gensim
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.fasttext import FastTextKeyedVectors
from gensim.models.poincare import PoincareModel, PoincareKeyedVectors
import requests

from .textpreprocessing import tokenize


def load_word2vec_model(
        path: str | PathLike,
        binary: bool = True
) -> KeyedVectors:
    """Load a pre-trained Word2Vec model.

    Args:
        path: Path to the Word2Vec model file.
        binary: Whether the file is in binary format. Default: True.

    Returns:
        A KeyedVectors model containing word embeddings.
    """
    return KeyedVectors.load_word2vec_format(path, binary=binary)


def load_fasttext_model(
        path: str | PathLike,
        encoding: Any = 'utf-8'
) -> FastTextKeyedVectors:
    """Load a pre-trained FastText model.

    Args:
        path: Path to the FastText model file.
        encoding: File encoding. Default: 'utf-8'.

    Returns:
        A FastTextKeyedVectors model.
    """
    return gensim.models.fasttext.load_facebook_vectors(path, encoding=encoding)


def load_poincare_model(
        path: str | PathLike,
        word2vec_format: bool = True,
        binary: bool = False
) -> PoincareKeyedVectors:
    """Load a Poincaré embedding model.

    Args:
        path: Path to the Poincaré model file.
        word2vec_format: Whether to load from word2vec format. Default: True.
        binary: Whether file is binary. Default: False.

    Returns:
        A PoincareKeyedVectors model.
    """
    if word2vec_format:
        return PoincareKeyedVectors.load_word2vec_format(path, binary=binary)
    else:
        return PoincareModel.load(path).kv


def shorttext_to_avgvec(
        shorttext: str,
        wvmodel: KeyedVectors
) -> Annotated[npt.NDArray[np.float64], "1D array"]:
    """Convert short text to averaged embedding vector.

    Converts each token to its word embedding, averages them,
    and normalizes the result.

    Args:
        shorttext: Input text.
        wvmodel: Word embedding model.

    Returns:
        A normalized vector representation of the text.
    """
    vec = np.sum(
        [
            wvmodel[token].astype(np.float64)
            if token in wvmodel
            else np.array([1.]*wvmodel.vector_size) / np.sqrt(wvmodel.vector_size)
            for token in tokenize(shorttext)
        ],
        axis=0
    )

    # normalize
    norm = np.linalg.norm(vec)
    if norm != 0:
        vec /= norm

    return vec


class RESTfulKeyedVectors(KeyedVectors):
    """Remote word vector client via REST API.

    Connects to a remote WordEmbedAPI service to access word
    embeddings via HTTP requests.

    Attributes:
        url: Base URL of the API.
        port: Port number for the API.
    """

    def __init__(self, url: str, port: str | int='5000'):
        """Initialize the client.

        Args:
            url: Base URL of the API (e.g., 'http://localhost').
            port: Port number. Default: '5000'.
        """
        self.url = url
        self.port = port

    def closer_than(self, entity1: str, entity2: str) -> list | dict:
        """Find words closer to entity1 than entity2 is.

        Args:
            entity1: First word.
            entity2: Reference word.

        Returns:
            List of words closer to entity1 than entity2.
        """
        r = requests.post(self.url + ':' + self.port + '/closerthan',
                          json={'entity1': entity1, 'entity2': entity2})
        return r.json()

    def distance(self, entity1: str, entity2: str) -> float:
        """Compute distance between two words.

        Args:
            entity1: First word.
            entity2: Second word.

        Returns:
            Distance between the word vectors.
        """
        r = requests.post(self.url + ':' + self.port + '/distance',
                          json={'entity1': entity1, 'entity2': entity2})
        return r.json()['distance']

    def distances(
            self,
            entity1: str,
            other_entities: Optional[list[str]] = None
    ) -> Annotated[npt.NDArray[np.float64], "1D array"]:
        """Compute distances from one word to multiple words.

        Args:
            entity1: First word.
            other_entities: List of words to compare against.

        Returns:
            Array of distances.
        """
        if other_entities is None:
            other_entities = []

        r = requests.post(self.url + ':' + self.port + '/distances',
                          json={'entity1': entity1, 'other_entities': other_entities})
        return np.array(r.json()['distances'], dtype=np.float32)

    def get_vector(self, entity: str) -> Annotated[npt.NDArray[np.float64], "1D array"]:
        """Get word vector for a word.

        Args:
            entity: Word to get vector for.

        Returns:
            Word embedding vector.

        Raises:
            KeyError: If word not in vocabulary.
        """
        r = requests.post(self.url + ':' + self.port + '/get_vector', json={'token': entity})
        returned_dict = r.json()
        if 'vector' in returned_dict:
            return np.array(returned_dict['vector'])
        else:
            raise KeyError(f'The token {entity} does not exist in the model.')

    def most_similar(self, **kwargs) -> list[tuple[str, float]]:
        """Find most similar words.

        Args:
            **kwargs: Arguments passed to the API (e.g., positive, negative).

        Returns:
            List of (word, similarity) tuples.
        """
        r = requests.post(self.url + ':' + self.port + '/most_similar', json=kwargs)
        return [tuple(pair) for pair in r.json()]

    def most_similar_to_given(self, entity1: str, entities_list: list[str]) -> list[str]:
        """Find most similar word from a list to a given word.

        Args:
            entity1: Reference word.
            entities_list: List of candidate words.

        Returns:
            List of words sorted by similarity.
        """
        r = requests.post(self.url + ':' + self.port + '/most_similar_to_given',
                          json={'entity1': entity1, 'entities_list': entities_list})
        return r.json()['token']

    def rank(self, entity1: str, entity2: str) -> int:
        """Get similarity rank between two words.

        Args:
            entity1: First word.
            entity2: Second word.

        Returns:
            Rank of entity2 relative to entity1.
        """
        r = requests.post(self.url + ':' + self.port + '/rank',
                          json={'entity1': entity1, 'entity2': entity2})
        return r.json()['rank']

    def save(self, fname_or_handle: TextIO, **kwargs) -> None:
        """Save is not supported for remote vectors.

        Raises:
            IOError: Always, since remote vectors cannot be saved locally.
        """
        raise IOError('The class RESTfulKeyedVectors do not persist models to a file.')

    def similarity(self, entity1: str, entity2: str) -> float:
        """Compute similarity between two words.

        Args:
            entity1: First word.
            entity2: Second word.

        Returns:
            Similarity score between 0 and 1.
        """
        r = requests.post(self.url + ':' + self.port + '/similarity',
                          json={'entity1': entity1, 'entity2': entity2})
        return r.json()['similarity']

# reference: https://radimrehurek.com/gensim/models/keyedvectors.html
