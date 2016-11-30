import random
from collections import defaultdict
import json
import os
import zipfile

import pandas as pd
import numpy as np


def retrieve_csvdata_as_dict(filepath):
    """ Retrieve the training data in a CSV file.

    Retrieve the training data in a CSV file, with the first column being the
    class labels, and second column the text data. It returns a dictionary with
    the class labels as keys, and a list of short texts as the value for each key.

    :param filepath: path of the training data (CSV)
    :return: a dictionary with class labels as keys, and lists of short texts
    :type filepath: str
    :rtype: dict
    """
    df = pd.read_csv(filepath)
    category_col, descp_col = df.columns.values.tolist()
    shorttextdict = defaultdict(lambda : [])
    for category, descp in zip(df[category_col], df[descp_col]):
        if type(descp)==str:
            shorttextdict[category] += [descp]
    return dict(shorttextdict)

# for backward compatibility
def retrieve_data_as_dict(filepath):
    """ Retrieve the training data in a CSV file.

    This calls :func:`~retrieve_csvdata_as_dict` for backward compatibility.

    :param filepath: path of the training data (CSV)
    :return: a dictionary with class labels as keys, and lists of short texts
    :type filepath: str
    :rtype: dict
    """
    return retrieve_csvdata_as_dict(filepath)

def retrieve_jsondata_as_dict(filepath):
    """ Retrieve the training data in a JSON file.

    Retrieve the training data in a JSON file, with
    the class labels as keys, and a list of short texts as the value for each key.
    It returns the corresponding dictionary.

    :param filepath: path of the training data (JSON)
    :return: a dictionary with class labels as keys, and lists of short texts
    :type filepath: str
    :rtype: dict
    """
    return json.load(open(filepath, 'r'))

def subjectkeywords():
    """ Return an example data set of subjects.

    Return an example data set, with three subjects and corresponding keywords.
    This is in the format of the training input.

    :return: example data set
    :rtype: dict

    Examples

    >>> import shorttext
    >>> classdict = shorttext.data.data_retrieval.subjectkeywords()
    >>> classdict
    {'mathematics': ['linear algebra',
      'topology',
      'algebra',
      'calculus',
      'variational calculus',
      'functional field',
      'real analysis',
      'complex analysis',
      'differential equation',
      'statistics',
      'statistical optimization',
      'probability',
      'stochastic calculus',
      'numerical analysis',
      'differential geometry'],
     'physics': ['renormalization',
      'classical mechanics',
      'quantum mechanics',
      'statistical mechanics',
      'functional field',
      'path integral',
      'quantum field theory',
      'electrodynamics',
      'condensed matter',
      'particle physics',
      'topological solitons',
      'astrophysics',
      'spontaneous symmetry breaking',
      'atomic molecular and optical physics',
      'quantum chaos'],
     'theology': ['divine providence',
      'soteriology',
      'anthropology',
      'pneumatology',
      'Christology',
      'Holy Trinity',
      'eschatology',
      'scripture',
      'ecclesiology',
      'predestination',
      'divine degree',
      'creedal confessionalism',
      'scholasticism',
      'prayer',
      'eucharist']}
    """
    this_dir, _ = os.path.split(__file__)
    return retrieve_csvdata_as_dict(os.path.join(this_dir, 'shorttext_exampledata.csv'))

def nihreports(txt_col='PROJECT_TITLE', label_col='FUNDING_ICs', sample_size=512):
    """ Return an example data set, sampled from NIH REPORTER.

    Return an example data set from NIH (National Institutes of Health),
    data publicly available from their RePORTER
    website. (`link
    <https://exporter.nih.gov/ExPORTER_Catalog.aspx>`_).
    The data is with `txt_col` being either project titles ('PROJECT_TITLE')
    or proposal abstracts ('ABSTRACT_TEXT'), and label_col being the names of the ICs (Institutes or Centers),
    with 'IC_NAME' the whole form, and 'FUNDING_ICs' the abbreviated form).

    Dataset directly adapted from the NIH data from `R` package `textmineR
    <https://cran.r-project.org/web/packages/textmineR/index.html>`_.

    :param txt_col: column for the text (Default: 'PROJECT_TITLE')
    :param label_col: column for the labels (Default: 'FUNDING_ICs')
    :param sample_size: size of the sample (Default: 512)
    :return: example data set
    :type txt_col: str
    :type label_col: str
    :type sample_size: int
    :rtype: dict
    """
    # txt_col = 'PROJECT_TITLE' or 'ABSTRACT_TEXT'
    # label_col = 'FUNDING_ICs' or 'IC_NAME'
    this_dir, _ = os.path.split(__file__)
    zfile = zipfile.ZipFile(os.path.join(this_dir, 'nih_full.csv.zip'))
    nih = pd.read_csv(zfile.open('nih_full.csv'), na_filter=False)
    nb_data = len(nih)
    sample_size = nb_data if sample_size==None else min(nb_data, sample_size)

    classdict = defaultdict(lambda : [])

    for rowidx in np.random.randint(nb_data, size=min(nb_data, sample_size)):
        label = nih.ix[rowidx, label_col]
        if label_col=='FUNDING_ICs':
            if label=='':
                label = 'OTHER'
            else:
                endpos = label.index(':')
                label = label[:endpos]
        classdict[label] += [nih.ix[rowidx, txt_col]]

    return dict(classdict)

def mergedict(dicts):
    """ Merge data dictionary.

    Merge dictionaries of the data in the training data format.

    :param dicts: dicts to merge
    :return: merged dict
    :type dicts: list
    :rtype: dict
    """
    mdict = defaultdict(lambda : [])
    for thisdict in dicts:
        for label in thisdict:
            mdict[label] += thisdict[label]
    return dict(mdict)

def yield_crossvalidation_classdicts(classdict, nb_partitions, shuffle=False):
    """ Yielding test data and training data for cross validation by partitioning it.

    Given a training data, partition the data into portions, each will be used as test
    data set, while the other training data set. It returns a generator.

    :param classdict: training data
    :param nb_partitions: number of partitions
    :param shuffle: whether to shuffle the data before partitioning
    :return: generator, producing a test data set and a training data set each time
    :type classdict: dict
    :type nb_partitions: int
    :type shuffle: bool
    :rtype: generator
    """
    crossvaldicts = []
    for i in range(nb_partitions):
        crossvaldicts.append(defaultdict(lambda: []))

    for label in classdict:
        nb_data = len(classdict[label])
        partsize = nb_data / nb_partitions
        sentences = classdict[label] if not shuffle else random.shuffle(sentences)
        for i in range(nb_partitions):
            crossvaldicts[i][label] += sentences[i * partsize:min(nb_data, (i + 1) * partsize)]
    crossvaldicts = map(dict, crossvaldicts)

    for i in range(nb_partitions):
        testdict = crossvaldicts[i]
        traindict = mergedict([crossvaldicts[j] for j in range(nb_partitions) if j != i])
        yield testdict, traindict