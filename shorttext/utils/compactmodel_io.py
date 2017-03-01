from tempfile import mkdtemp
import zipfile
import json
import os
from functools import partial

import utils.classification_exceptions as e

def removedir(dir):
    for filename in os.listdir(dir):
        if os.path.isdir(filename):
            removedir(os.path.join(dir, filename))
            os.rmdir(os.path.isdir(filename))
        else:
            os.remove(dir+'/'+filename)
    os.rmdir(dir)


def save_compact_model(filename, savefunc, prefix, suffices, infodict):
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
    # create temporary directory
    tempdir = mkdtemp()

    # unzipping
    inputfile = zipfile.ZipFile(filename, mode='r')
    inputfile.extractall(tempdir)
    inputfile.close()

    # check model config
    readinfodict = json.load(tempdir+'/modelconfig.json')
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
    # define the inherit class
    class DressedClassifier(Classifier):
        def save_compact_model(self, filename):
            save_compact_model(filename, self.savemodel, prefix, suffices, infodict)

        def load_compact_model(self, filename):
            return load_compact_model(filename, self.loadmodel, prefix, infodict)

    # return decorated classifier
    return DressedClassifier

# decorator for use
def compactio(infodict, prefix, suffices):
    return partial(CompactIOClassifier, infodict=infodict, prefix=prefix, suffices=suffices)