
from keras.models import model_from_json


def save_model(nameprefix, model):
    """ Save a keras sequential model into files.

    Given a keras sequential model, save the model with the given file path prefix.
    It saves the model into a JSON file, and an HDF5 file (.h5).

    :param nameprefix: Prefix of the paths of the model files
    :param model: keras sequential model to be saved
    :return: None
    :type nameprefix: str
    :type model: keras.models.Model
    """
    model_json = model.to_json()
    open(nameprefix+'.json', 'w').write(model_json)
    model.save_weights(nameprefix+'.h5')


def load_model(nameprefix):
    """ Load a keras sequential model from files.

    Given the prefix of the file paths, load a keras sequential model from
    a JSON file and an HDF5 file.

    :param nameprefix: Prefix of the paths of the model files
    :return: keras sequential model
    :type nameprefix: str
    :rtype: keras.models.Model
    """
    model = model_from_json(open(nameprefix+'.json', 'r').read())
    model.load_weights(nameprefix+'.h5')
    return model