
from functools import partial
from os import PathLike

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix
from gensim.corpora import Dictionary
from sklearn.preprocessing import OneHotEncoder

from ...utils.misc import textfile_generator


class SentenceToCharVecEncoder:
    """One-hot encoder for character-level text representations.

    Converts sentences into one-hot encoded vectors at the character
    level. Useful for character-level sequence models.

    Reference:
        General architecture inspired by char-RNN and related models.
    """

    def __init__(self, dictionary: Dictionary, signalchar: str='\n'):
        """Initialize the character vector encoder.

        Args:
            dictionary: Gensim Dictionary mapping characters to indices.
            signalchar: Signal character for sequence markers. Default: '\\n'.
        """
        self.dictionary = dictionary
        self.signalchar = signalchar
        numchars = len(self.dictionary)
        self.onehot_encoder = OneHotEncoder()
        self.onehot_encoder.fit(np.arange(numchars).reshape((numchars, 1)))

    def calculate_prelim_vec(self, sent: str) -> npt.NDArray[np.float64]:
        """Convert sentence to one-hot character vectors.

        Args:
            sent: Input sentence.

        Returns:
            One-hot encoded sparse matrix where each row represents
            a character's encoding.
        """
        return self.onehot_encoder.transform(
            np.array([self.dictionary.token2id[c] for c in sent]).reshape((len(sent), 1))
        ).astype(np.float64)

    def encode_sentence(
            self,
            sent: str,
            maxlen: int,
            startsig: bool = False,
            endsig=False
    ) -> csc_matrix:
        """Encode a sentence to a sparse character vector matrix.

        Args:
            sent: Input sentence to encode.
            maxlen: Maximum length of the encoded sequence.
            startsig: Whether to prepend signal character. Default: False.
            endsig: Whether to append signal character. Default: False.

        Returns:
            Sparse matrix representing the sentence with shape
            (maxlen + startsig + endsig, num_chars).
        """
        cor_sent = (self.signalchar if startsig else '') + sent[:min(maxlen, len(sent))] + (self.signalchar if endsig else '')
        sent_vec = self.calculate_prelim_vec(cor_sent).tocsc()
        if sent_vec.shape[0] == maxlen + startsig + endsig:
            return sent_vec
        else:
            return csc_matrix((sent_vec.data, sent_vec.indices, sent_vec.indptr),
                              shape=(maxlen + startsig + endsig, sent_vec.shape[1]),
                              dtype=np.float64)

    def encode_sentences(
            self,
            sentences: list[str],
            maxlen: int,
            sparse: bool = True,
            startsig: bool = False,
            endsig: bool = False
    ) -> list[npt.NDArray[np.float64]] | npt.NDArray[np.float64]:
        """Encode multiple sentences into character vectors.

        Args:
            sentences: List of sentences to encode.
            maxlen: Maximum length for each encoded sentence.
            sparse: Whether to return sparse matrices. Default: True.
            startsig: Whether to prepend signal character. Default: False.
            endsig: Whether to append signal character. Default: False.

        Returns:
            If sparse=True: list of sparse matrices.
            If sparse=False: numpy array of shape (n_sentences, maxlen, num_chars).
        """
        encode_sent_func = partial(self.encode_sentence, startsig=startsig, endsig=endsig, maxlen=maxlen)
        list_encoded_sentences_map = map(encode_sent_func, sentences)
        if sparse:
            return list(list_encoded_sentences_map)
        else:
            return np.array([sparsevec.toarray() for sparsevec in list_encoded_sentences_map])

    def __len__(self) -> int:
        """Return the number of unique characters in the dictionary."""
        return len(self.dictionary)


def initSentenceToCharVecEncoder(
        textfile: str | PathLike,
        encoding: bool=None
) -> SentenceToCharVecEncoder:
    """Create a SentenceToCharVecEncoder from a text file.

    Builds a character dictionary from the given text file and returns
    an encoder instance.

    Args:
        textfile: Path to the text file for building the character dictionary.
        encoding: Encoding of the text file. Default: None.

    Returns:
        A SentenceToCharVecEncoder instance.
    """
    dictionary = Dictionary(
        map(
            lambda line: [c for c in line],
            textfile_generator(textfile, encoding=encoding)
        )
    )
    return SentenceToCharVecEncoder(dictionary)
