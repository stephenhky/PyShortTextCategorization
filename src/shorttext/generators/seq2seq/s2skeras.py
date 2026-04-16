
from typing import Literal
from os import PathLike

import numpy as np
import numpy.typing as npt
import orjson

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

from ...utils.compactmodel_io import CompactIOMachine
from ...utils.classification_exceptions import ModelNotTrainedException

# Reference: https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

kerasseq2seq_suffices = ['.weights.h5', '.json', '_s2s_hyperparam.json', '_encoder.weights.h5', '_encoder.json', '_decoder.h5', '_decoder.weights.json']


class Seq2SeqWithKeras(CompactIOMachine):
    """Sequence-to-sequence (seq2seq) model using Keras.

    Implements encoder-decoder architecture for sequence generation tasks.

    Reference:
        Ilya Sutskever, James Martens, Geoffrey Hinton, "Generating Text with Recurrent Neural Networks,"
        ICML (2011). https://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf

        Ilya Sutskever, Oriol Vinyals, Quoc V. Le, "Sequence to Sequence Learning with Neural Networks,"
        arXiv:1409.3215 (2014). https://arxiv.org/abs/1409.3215

        Francois Chollet, "A ten-minute introduction to sequence-to-sequence learning in Keras,"
        The Keras Blog. https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

        Aurelien Geron, Hands-On Machine Learning with Scikit-Learn and TensorFlow (Sebastopol, CA: O'Reilly Media, 2017).
    """
    def __init__(self, vecsize: int, latent_dim: int):
        """Initialize the model.

        Args:
            vecsize: Vector size of the sequence.
            latent_dim: Latent dimension in the RNN cell.
        """
        super().__init__(
            {'classifier': 'kerasseq2seq'},
            'kerasseq2seq',
            kerasseq2seq_suffices
        )
        self.vecsize = vecsize
        self.latent_dim = latent_dim
        self.compiled = False
        self.trained = False

    def prepare_model(self) -> None:
        """Prepare the Keras model."""
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(None, self.vecsize))
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.vecsize))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.vecsize, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Define sampling models
        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                              [decoder_outputs] + decoder_states)

        self.model = model
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

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
        self.model.compile(optimizer=optimizer, loss=loss)
        self.compiled = True

    def fit(
            self,
            encoder_input: npt.NDArray[np.float64],
            decoder_input: npt.NDArray[np.float64],
            decoder_output: npt.NDArray[np.float64],
            batch_size: int = 64,
            epochs: int = 100
    ) -> None:
        """Fit the seq2seq model.

        Args:
            encoder_input: Encoder input, a rank-3 tensor.
            decoder_input: Decoder input, a rank-3 tensor.
            decoder_output: Decoder output, a rank-3 tensor.
            batch_size: Batch size. Default: 64.
            epochs: Number of epochs. Default: 100.
        """
        self.model.fit([encoder_input, decoder_input], decoder_output,
                       batch_size=batch_size,
                       epochs=epochs)
        self.trained = True

    def savemodel(self, prefix: str, final: bool=False) -> None:
        """Save the trained model to files.

        For compact save, use save_compact_model instead.

        Args:
            prefix: Prefix of the file path.
            final: Whether the model is final (cannot be further trained). Default: False.

        Raises:
            ModelNotTrainedException: If no trained model exists.
        """
        if not self.trained:
            raise ModelNotTrainedException()

        # save hyperparameters
        open(prefix + '_s2s_hyperparam.json', 'wb').write(
            orjson.dumps({'vecsize': self.vecsize, 'latent_dim': self.latent_dim})
        )

        # save whole model
        if final:
            self.model.save_weights(prefix+'.weights.h5')
        else:
            self.model.save(prefix+'.weights.h5')
        open(prefix+'.json', 'w').write(self.model.to_json())

        # save encoder and decoder
        if final:
            self.encoder_model.save_weights(prefix+'_encoder.weights.h5')
            self.decoder_model.save_weights(prefix + '_decoder.weights.h5')
        else:
            self.encoder_model.save(prefix + '_encoder.weights.h5')
            self.decoder_model.save(prefix+'_decoder.weights.h5')
        open(prefix+'_encoder.json', 'w').write(self.encoder_model.to_json())
        open(prefix+'_decoder.json', 'w').write(self.decoder_model.to_json())

    def loadmodel(self, prefix: str) -> None:
        """Load a trained model from files.

        For compact load, use load_compact_model instead.

        Args:
            prefix: Prefix of the file path.
        """
        hyperparameters = orjson.loads(open(prefix+'_s2s_hyperparam.json', 'rb').read())
        self.vecsize, self.latent_dim = hyperparameters['vecsize'], hyperparameters['latent_dim']
        self.model = load_model(prefix+'.weights.h5')
        self.encoder_model = load_model(prefix+'_encoder.weights.h5')
        self.decoder_model = load_model(prefix+'_decoder.weights.h5')
        self.trained = True


def loadSeq2SeqWithKeras(path: str | PathLike, compact: bool=True) -> Seq2SeqWithKeras:
    """Load a trained Seq2SeqWithKeras model from file.

    Args:
        path: Path of the model file.
        compact: Whether to load a compact model. Default: True.

    Returns:
        Seq2SeqWithKeras instance for sequence-to-sequence inference.
    """
    generator = Seq2SeqWithKeras(0, 0)
    if compact:
        generator.load_compact_model(path)
    else:
        generator.loadmodel(path)
    generator.compiled = True
    return generator
