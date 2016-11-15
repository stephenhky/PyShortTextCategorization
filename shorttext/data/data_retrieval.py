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