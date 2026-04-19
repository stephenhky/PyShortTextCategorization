"""
This module contains general routines to zip all model files into one compact file.
The model can be copied or transferred easily.

The methods and decorators in this module are called by other codes. It is not
recommended for developers to call them directly.
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
    """Remove all subdirectories and files under the specified path.

    Args:
        dir: Path of the directory to clean.
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
    """Save the model in one compact file by zipping all related files.

    Args:
        filename: Name of the output model file.
        savefunc: Function that performs the saving action. Takes one argument (str) - the prefix.
        prefix: Prefix of the names of the files related to the model.
        suffices: List of file suffixes.
        infodict: Dictionary with model information. Must contain the key 'classifier'.
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
) -> Any:
    """Load a model from a compact file.

    Args:
        filename: Name of the model file.
        loadfunc: Function that performs the loading action. Takes one argument (str) - the prefix.
        prefix: Prefix of the names of the files.
        infodict: Dictionary with model information. Must contain the key 'classifier'.

    Returns:
        The loaded model instance.
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
    """Base class that implements compact model I/O.

    Replaces the original compactio decorator.
    """

    def __init__(
            self,
            infodict: dict[str, Any],
            prefix: str,
            suffices: list[str]
    ):
        """Initialize the compact I/O machine.

        Args:
            infodict: Dictionary with model information. Must contain 'classifier'.
            prefix: Prefix for model file names.
            suffices: List of file suffixes for the model files.
        """
        self.infodict = infodict
        self.prefix = prefix
        self.suffices = suffices

    @abstractmethod
    def savemodel(self, nameprefix: str) -> None:
        """Save the model to files.

        Args:
            nameprefix: Prefix for model file paths.
        """
        raise NotImplemented()

    @abstractmethod
    def loadmodel(self, nameprefix: str) -> Self:
        """Load the model from files.

        Args:
            nameprefix: Prefix for model file paths.
        """
        raise NotImplemented()

    def save_compact_model(self, filename: str, *args, **kwargs) -> None:
        """Save the model in a compressed binary format.

        Args:
            filename: Name of the model file.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        save_compact_model(filename, self.savemodel, self.prefix, self.suffices, self.infodict, *args, **kwargs)

    def load_compact_model(self, filename: str, *args, **kwargs) -> Self:
        """Load the model from a compressed binary format.

        Args:
            filename: Name of the model file.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        """
        return load_compact_model(filename, self.loadmodel, self.prefix, self.infodict, *args, **kwargs)

    def get_info(self) -> dict[str, Any]:
        """Get model metadata.

        Returns:
            Dictionary with classifier, prefix, and suffices.
        """
        return {'classifier': self.infodict['classifier'],
                'prefix': self.prefix,
                'suffices': self.suffices}


def get_model_config_field(filename: str | PathLike, parameter: str) -> str:
    """Get a configuration parameter from a compact model file.

    Args:
        filename: Path to the model file.
        parameter: Parameter name to retrieve.

    Returns:
        The parameter value.
    """
    inputfile = zipfile.ZipFile(filename, mode='r')
    readinfodict = json.load(inputfile.open("modelconfig.json", "r"))
    return readinfodict[parameter]


def get_model_classifier_name(filename: str| PathLike) -> str:
    """Get the classifier name from a compact model file.

    Args:
        filename: Path to the model file.

    Returns:
        The classifier name.
    """
    return get_model_config_field(filename, 'classifier')
