from tempfile import mkdtemp
import zipfile
import json
import os

def removedir(dir):
    for filename in os.listdir(dir):
        if os.path.isdir(filename):
            removedir(os.path.join(dir, filename))
            os.rmdir(os.path.isdir(filename))
        else:
            os.remove(dir+'/'+filename)
    os.rmdir(dir)


def save_compact_model(filename, savefunc, prefix, suffices, infodict={}):
    # create temporary directory
    tempdir = mkdtemp()
    savefunc(tempdir+'/'+prefix)

    # zipping
    outputfile = zipfile.ZipFile(filename, mode='w')
    for suffix in suffices:
        outputfile.write(tempdir+'/'+prefix+suffix)
    outputfile.writestr('modelconfig.json', json.dumps(infodict))
    outputfile.close()

    # delete temporary files
    removedir(tempdir)

def load_compact_model(filename, loadfunc, prefix):
    # create temporary directory
    tempdir = mkdtemp()

    # unzipping
    inputfile = zipfile.ZipFile(filename, mode='r')
    inputfile.extractall(tempdir)
    # filenames = inputfile.namelist()
    inputfile.close()

    # load the model
    returnobj = loadfunc(tempdir+'/'+prefix)

    # delete temporary files
    removedir(tempdir)

    return returnobj