
from dataclasses import dataclass

from tensorflow.keras import Model


@dataclass
class AutoEncoderPackage:
    autoencoder: Model
    encoder: Model
    decoder: Model
