import tensorflow
from tensorflow.keras.models import model_from_json


def save_model(nameprefix: str, model: tensorflow.keras.models.Model) -> None:
    """Save a Keras model to files.

    Args:
        nameprefix: Prefix for output files.
        model: Keras model to save.
    """
    model_json = model.to_json()
    open(nameprefix+'.json', 'w').write(model_json)
    model.save_weights(nameprefix+'.weights.h5')


def load_model(nameprefix: str) -> tensorflow.keras.models.Model:
    """Load a Keras model from files.

    Args:
        nameprefix: Prefix for input files.

    Returns:
        Loaded Keras model.
    """
    model = model_from_json(open(nameprefix+'.json', 'r').read())
    model.load_weights(nameprefix+'.weights.h5')
    return model
