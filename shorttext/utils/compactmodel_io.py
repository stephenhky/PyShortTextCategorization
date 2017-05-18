"""
This module contains general routines to zip all model files into one compact file. The model can be copied
or transferred with handiness.

The methods and decorators in this module are called by other codes. It is not recommended for developers
to call them directly.
"""

from tempfile import mkdtemp
import zipfile
import json
import os
from functools import partial

from . import classification_exceptions as e

def removedir(dir):
    """ Remove all subdirectories and files under the specified path.

    :param dir: path of the directory to be clean
    :return: None
    """
    for filename in os.listdir(dir):
        if os.path.isdir(filename):
            removedir(os.path.join(dir, filename))
            os.rmdir(os.path.isdir(filename))
        else:
            os.remove(dir+'/'+filename)
    os.rmdir(dir)


def save_compact_model(filename, savefunc, prefix, suffices, infodict):
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
    savefunc(tempdir+'/'+prefix)

    # zipping
    outputfile = zipfile.ZipFile(filename, mode='w')
    for suffix in suffices:
        outputfile.write(tempdir+'/'+prefix+suffix, prefix+suffix)
    outputfile.writestr('modelconfig.json', json.dumps(infodict))
    outputfile.close()

    # delete temporary files
    removedir(tempdir)

def load_compact_model(filename, loadfunc, prefix, infodict):
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
    readinfodict = json.load(open(tempdir+'/modelconfig.json', 'r'))
    if readinfodict['classifier'] != infodict['classifier']:
        raise e.IncorrectClassificationModelFileException(infodict['classifier'],
                                                          readinfodict['classifier'])

    # load the model
    returnobj = loadfunc(tempdir+'/'+prefix)

    # delete temporary files
    removedir(tempdir)

    return returnobj

# decorator that adds compact model methods to classifier dynamically
def CompactIOClassifier(Classifier, infodict, prefix, suffices):
    """ Returns a decorated class object with additional methods for compact model I/O.

    The class itself must have methods :func:`loadmodel` and :func:`savemodel` that
    takes the prefix of the model files as the argument.

    :param Classifier: class to be decorated
    :param infodict: information about the model. Must contain the key 'classifier'.
    :param prefix: prefix of names of the model file
    :param suffices: suffices of the names of the model file
    :return: the decorated class
    :type Classifier: classobj
    :type infodict: dict
    :type prefix: str
    :type suffices: list
    :rtype: classobj
    """
    # define the inherit class
    class DressedClassifier(Classifier):
        def save_compact_model(self, filename):
            save_compact_model(filename, self.savemodel, prefix, suffices, infodict)

        def load_compact_model(self, filename):
            return load_compact_model(filename, self.loadmodel, prefix, infodict)

        def get_info(self):
            return {'classifier': infodict['classifier'],
                    'prefix': prefix,
                    'suffices': suffices}

    # return decorated classifier
    return DressedClassifier

# decorator for use
def compactio(infodict, prefix, suffices):
    """ Returns a decorator that performs the decoration by :func:`CompactIOClassifier`.

    :param infodict: information about the model. Must contain the key 'classifier'.
    :param prefix: prefix of names of the model file
    :param suffices: suffices of the names of the model file
    :return: the decorator
    :type infodict: dict
    :type prefix: str
    :type suffices: list
    :rtype: function
    """
    return partial(CompactIOClassifier, infodict=infodict, prefix=prefix, suffices=suffices)

def get_model_config_field(filename, parameter):
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
    readinfodict = json.load(inputfile.open('modelconfig.json', 'r'))
    return readinfodict[parameter]

def get_model_classifier_name(filename):
    """ Return the name of the classifier from a model file.

    Read the file `modelconfig.json` in the compact model file, and return
    the name of the classifier.

    :param filename: path of the model file
    :return: name of the classifier
    :type filename: str
    :rtype: str
    """
    return get_model_config_field(filename, 'classifier')