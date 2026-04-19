
from dataclasses import dataclass

from tensorflow.keras import Model


@dataclass
class AutoEncoderPackage:
    """Package containing autoencoder components.

    Attributes:
        autoencoder: The full autoencoder model.
        encoder: The encoder part of the autoencoder.
        decoder: The decoder part of the autoencoder.
    """
    autoencoder: Model
    encoder: Model
    decoder: Model
