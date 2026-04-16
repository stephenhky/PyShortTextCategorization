
from typing import Literal
from os import PathLike

import numpy as np
import numpy.typing as npt
import gensim
import orjson

from .s2skeras import Seq2SeqWithKeras, loadSeq2SeqWithKeras, kerasseq2seq_suffices
from ..charbase.char2vec import SentenceToCharVecEncoder
from ...utils.compactmodel_io import CompactIOMachine


charbases2s_suffices = kerasseq2seq_suffices + ['_dictionary.dict', '_charbases2s.json']


class CharBasedSeq2SeqGenerator(CompactIOMachine):
    """Character-based sequence-to-sequence model.

    Implements seq2seq at the character level. Uses Seq2SeqWithKeras internally.

    Reference:
        Oriol Vinyals, Quoc Le, "A Neural Conversational Model," arXiv:1506.05869 (2015).
        https://arxiv.org/abs/1506.05869
    """
    def __init__(
            self,
            sent2charvec_encoder: SentenceToCharVecEncoder,
            latent_dim: int,
            maxlen: int
    ):
        """Initialize the generator.

        Args:
            sent2charvec_encoder: Character encoder.
            latent_dim: Number of latent dimensions.
            maxlen: Maximum length of a sentence.
        """
        super().__init__(
            {'classifier': 'charbases2s'},
            'charbases2s',
            charbases2s_suffices
        )
        self.compiled = False
        if sent2charvec_encoder != None:
            self.sent2charvec_encoder = sent2charvec_encoder
            self.dictionary = self.sent2charvec_encoder.dictionary
            self.nbelem = len(self.dictionary)
            self.latent_dim = latent_dim
            self.maxlen = maxlen
            self.s2sgenerator = Seq2SeqWithKeras(self.nbelem, self.latent_dim)

    def compile(
            self,
            optimizer: Literal["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"] = 'rmsprop',
            loss: str = 'categorical_crossentropy'
    ) -> None:
        """Compile the Keras model.

        Args:
            optimizer: Optimizer for gradient descent. Options: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam. Default: rmsprop.
            loss: Loss function from tensorflow.keras. Default: 'categorical_crossentropy'.
        """
        if not self.compiled:
            self.s2sgenerator.prepare_model()
            self.s2sgenerator.compile(optimizer=optimizer, loss=loss)
            self.compiled = True

    def prepare_trainingdata(
            self,
            txtseq: str
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Transform text to numerical vector format.

        Args:
            txtseq: Input text.

        Returns:
            Tuple of (encoder_input, decoder_input, decoder_output) as rank-3 tensors.
        """
        encoder_input = self.sent2charvec_encoder.encode_sentences(txtseq[:-1], startsig=True, maxlen=self.maxlen, sparse=False)
        decoder_input = self.sent2charvec_encoder.encode_sentences(txtseq[1:], startsig=True, maxlen=self.maxlen, sparse=False)
        decoder_output = self.sent2charvec_encoder.encode_sentences(txtseq[1:], endsig=True, maxlen=self.maxlen, sparse=False)
        return encoder_input, decoder_input, decoder_output

    def train(
            self,
            txtseq: str,
            batch_size: int = 64,
            epochs: int = 100,
            optimizer: Literal["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax", "nadam"] = 'rmsprop',
            loss: str = 'categorical_crossentropy'
    ) -> None:
        """Train the character-based seq2seq model.

        Args:
            txtseq: Training text.
            batch_size: Batch size. Default: 64.
            epochs: Number of epochs. Default: 100.
            optimizer: Optimizer for gradient descent. Default: rmsprop.
            loss: Loss function from tensorflow.keras. Default: 'categorical_crossentropy'.
        """
        encoder_input, decoder_input, decoder_output = self.prepare_trainingdata(txtseq)
        self.compile(optimizer=optimizer, loss=loss)
        self.s2sgenerator.fit(encoder_input, decoder_input, decoder_output, batch_size=batch_size, epochs=epochs)

    def decode(self, txtseq: str, stochastic: bool=True) -> str:
        """Generate output text from input text.

        Args:
            txtseq: Input text.
            stochastic: Whether to use stochastic sampling. Default: True.

        Returns:
            Generated output text.
        """
        # Encode the input as state vectors.
        inputvec = np.array([self.sent2charvec_encoder.encode_sentence(txtseq, maxlen=self.maxlen, endsig=True).toarray()])
        states_value = self.s2sgenerator.encoder_model.predict(inputvec)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, self.nbelem))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.dictionary.token2id['\n']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_txtseq = ''
        while not stop_condition:
            output_tokens, h, c = self.s2sgenerator.decoder_model.predict([target_seq] + states_value)

            # Sample a token
            if stochastic:
                sampled_token_index = np.random.choice(np.arange(output_tokens.shape[2]),
                                                       p=output_tokens[0, -1, :])
            else:
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.dictionary[sampled_token_index]
            decoded_txtseq += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or len(decoded_txtseq) > self.maxlen):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, self.nbelem))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_txtseq

    def savemodel(self, prefix: str, final: bool=False) -> None:
        """Save the trained model to files.

        For compact save, use save_compact_model instead.

        Args:
            prefix: Prefix of the file path.
            final: Whether the model is final (cannot be further trained). Default: False.

        Raises:
            ModelNotTrainedException: If no trained model exists.
        """
        self.s2sgenerator.savemodel(prefix, final=final)
        self.dictionary.save(prefix+'_dictionary.dict')
        open(prefix + '_charbases2s.json', 'wb').write(
            orjson.dumps({
                'maxlen': self.maxlen, 'latent_dim': self.latent_dim
            })
        )

    def loadmodel(self, prefix: str) -> None:
        """Load a trained model from files.

        For compact load, use load_compact_model instead.

        Args:
            prefix: Prefix of the file path.
        """
        self.dictionary = gensim.corpora.Dictionary.load(prefix+'_dictionary.dict')
        self.s2sgenerator = loadSeq2SeqWithKeras(prefix, compact=False)
        self.sent2charvec_encoder = SentenceToCharVecEncoder(self.dictionary)
        self.nbelem = len(self.dictionary)
        hyperparameters = orjson.loads(open(prefix+'_charbases2s.json', 'rb').read())
        self.latent_dim, self.maxlen = hyperparameters['latent_dim'], hyperparameters['maxlen']
        self.compiled = True


def loadCharBasedSeq2SeqGenerator(
        path: str | PathLike,
        compact: bool = True
) -> CharBasedSeq2SeqGenerator:
    """Load a trained CharBasedSeq2SeqGenerator from file.

    Args:
        path: Path of the model file.
        compact: Whether to load a compact model. Default: True.

    Returns:
        CharBasedSeq2SeqGenerator instance for seq2seq inference.
    """
    seq2seqer = CharBasedSeq2SeqGenerator(None, 0, 0)
    if compact:
        seq2seqer.load_compact_model(path)
    else:
        seq2seqer.loadmodel(path)
    return seq2seqer
