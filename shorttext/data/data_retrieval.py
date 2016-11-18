import random
from collections import defaultdict
import json
import os

import pandas as pd


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
    """ Return an example data set.

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