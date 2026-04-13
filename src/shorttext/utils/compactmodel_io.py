"""
This module contains general routines to zip all model files into one compact file. The model can be copied
or transferred with handiness.

The methods and decorators in this module are called by other codes. It is not recommended for developers
to call them directly.
"""

from abc import ABC, abstractmethod
from tempfile import mkdtemp
import zipfile
import json
import os
from os import PathLike
from typing import Any, Self

import orjson

from . import classification_exceptions as e


def removedir(dir: str) -> None:
    """ Remove all subdirectories and files under the specified path.

    :param dir: path of the directory to be clean
    :return: None
    """
    for filename in os.listdir(dir):
        if os.path.isdir(filename):
            removedir(os.path.join(dir, filename))
            os.rmdir(os.path.join(filename))
        else:
            os.remove(os.path.join(dir, filename))
    os.rmdir(dir)


def save_compact_model(
        filename: str,
        savefunc: callable,
        prefix: str,
        suffices: str,
        infodict: dict[str, Any]
) -> None:
    """ Save the model in one compact file by zipping all the related files.

    :param filename: name of the model file
    :param savefunc: method or function that performs the saving action. Only one argument (str), the prefix of the model files, to be passed.
    :param prefix: prefix of the names of the files related to the model
    :param suffices: list of suffices
    :param infodict: dictionary that holds information about the model. Must contain the key 'classifier'.
    :return: None
    :type filename: str
    :type savefunc: function
    :type prefix: str
    :type suffices: list
    :type infodict: dict
    """
    # create temporary directory
    tempdir = mkdtemp()
    savefunc(os.path.join(tempdir, prefix))

    # zipping
    outputfile = zipfile.ZipFile(filename, mode='w', allowZip64 = True)
    for suffix in suffices:
        outputfile.write(os.path.join(tempdir, prefix+suffix), prefix+suffix)
    outputfile.writestr('modelconfig.json', json.dumps(infodict))
    outputfile.close()

    # delete temporary files
    removedir(tempdir)


def load_compact_model(
        filename: str,
        loadfunc: callable,
        prefix: str,
        infodict: dict[str, Any]
) -> Any:     # returning CompactModelIO obj
    """ Load a model from a compact file that contains multiple files related to the model.

    :param filename: name of the model file
    :param loadfunc: method or function that performs the loading action. Only one argument (str), the prefix of the model files, to be passed.
    :param prefix: prefix of the names of the files
    :param infodict: dictionary that holds information about the model. Must contain the key 'classifier'.
    :return: instance of the model
    :type filename: str
    :type loadfunc: function
    :type prefix: str
    :type infodict: dict
    """
    # create temporary directory
    tempdir = mkdtemp()

    # unzipping
    inputfile = zipfile.ZipFile(filename, mode='r')
    inputfile.extractall(tempdir)
    inputfile.close()

    # check model config
    readinfodict = json.load(open(os.path.join(tempdir, 'modelconfig.json'), 'r'))
    if readinfodict['classifier'] != infodict['classifier']:
        raise e.IncorrectClassificationModelFileException(
            infodict['classifier'],
            readinfodict['classifier']
        )

    # load the model
    returnobj = loadfunc(os.path.join(tempdir, prefix))

    # delete temporary files
    removedir(tempdir)

    return returnobj


class CompactIOMachine(ABC):
    """ Base class that implements compact model I/O.

    This is to replace the original :func:`compactio` decorator.

    """
    def __init__(
            self,
            infodict: dict[str, Any],
            prefix: str,
            suffices: list[str]
    ):
        """

        :param infodict: information about the model. Must contain the key 'classifier'.
        :param prefix: prefix of names of the model file
        :param suffices: suffices of the names of the model file
        :type infodict: dict
        :type prefix: str
        :type suffices: list
        """
        self.infodict = infodict
        self.prefix = prefix
        self.suffices = suffices

    @abstractmethod
    def savemodel(self, nameprefix: str) -> None:
        """ Abstract method for `savemodel`.

        :param nameprefix: prefix of the model path
        :type nameprefix: str
        """
        raise NotImplemented()

    @abstractmethod
    def loadmodel(self, nameprefix: str) -> Self:
        """ Abstract method for `loadmodel`.

        :param nameprefix: prefix of the model path
        :type nameprefix: str
        """
        raise NotImplemented()

    def save_compact_model(self, filename: str, *args, **kwargs) -> None:
        """ Save the model in a compressed binary format.

        :param filename: name of the model file
        :param args: arguments
        :param kwargs: arguments
        :type filename: str
        :type args: dict
        :type kwargs: dict
        """
        save_compact_model(filename, self.savemodel, self.prefix, self.suffices, self.infodict, *args, **kwargs)

    def load_compact_model(self, filename: str, *args, **kwargs) -> Self:
        """ Load the model in a compressed binary format.

        :param filename: name of the model file
        :param args: arguments
        :param kwargs: arguments
        :type filename: str
        :type args: dict
        :type kwargs: dict
        """
        return load_compact_model(filename, self.loadmodel, self.prefix, self.infodict, *args, **kwargs)

    def get_info(self) -> dict[str, Any]:
        """ Getting information for the dressed machine.

        :return: dictionary of the information for the dressed machine.
        :rtype: dict
        """
        return {'classifier': self.infodict['classifier'],
                'prefix': self.prefix,
                'suffices': self.suffices}


def get_model_config_field(filename: str | PathLike, parameter: str) -> str:
    """ Return the configuration parameter of a model file.

    Read the file `modelconfig.json` in the compact model file, and return
    the value of a particular parameter.

    :param filename: path of the model file
    :param parameter: parameter to look in
    :return: value of the parameter of this model
    :type filename: str
    :type parameter: str
    :rtype: str
    """
    inputfile = zipfile.ZipFile(filename, mode='r')
    modelconfig_json = orjson.loads(inputfile.open("modelconfig.json", "rb").read())
    return readinfodict[parameter]


def get_model_classifier_name(filename: str| PathLike) -> str:
    """ Return the name of the classifier from a model file.

    Read the file `modelconfig.json` in the compact model file, and return
    the name of the classifier.

    :param filename: path of the model file
    :return: name of the classifier
    :type filename: str
    :rtype: str
    """
    return get_model_config_field(filename, 'classifier')
