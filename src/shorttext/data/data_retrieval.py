
import random
from collections import defaultdict
import json
import os
from os import PathLike
from pathlib import Path
import zipfile
import sys
import csv
from urllib.request import urlretrieve
from io import TextIOWrapper
from typing import Generator
from functools import reduce

import pandas as pd
import numpy as np
import orjson
from deprecation import deprecated


def retrieve_csvdata_as_dict(filepath: str | PathLike) -> dict[str, list[str]]:
    """Retrieve the training data in a CSV file.

    Reads a CSV file where the first column contains class labels and the second column
    contains text data. Returns a dictionary mapping class labels to lists of
    short texts.

    Args:
        filepath: Path to the CSV training data file.

    Returns:
        A dictionary with class labels as keys and lists of short texts as values.

    Reference:
        Data format inspired by common text classification benchmarks.
    """
    datafile = open(filepath, 'r')
    reader = csv.reader(datafile)
    headerread = False
    shorttextdict = defaultdict(lambda: [])
    for label, content in reader:
        if headerread:
            if isinstance(content, str):
                shorttextdict[label] += [content]
        else:
            headerread = True
    return dict(shorttextdict)


def retrieve_jsondata_as_dict(filepath: str | PathLike) -> dict:
    """Retrieve the training data in a JSON file.

    Reads a JSON file where class labels are keys and lists of short texts
    are values. Returns the corresponding dictionary.

    Args:
        filepath: Path to the JSON training data file.

    Returns:
        A dictionary with class labels as keys and lists of short texts as values.
    """
    return orjson.loads(open(filepath, 'rb').read())


def get_or_download_data(
        filename: str,
        origin: str,
        asbytes: bool = False
) -> TextIOWrapper:
    """Retrieve or download a data file.

    Checks if the file exists in the user's home directory under .shorttext.
    If not present, downloads from the given origin URL.

    Args:
        filename: Name of the file to retrieve.
        origin: URL to download the file from if not present locally.
        asbytes: If True, opens the file in binary mode. Default is False.

    Returns:
        A file object (text or binary mode depending on asbytes).
    """
    # determine path
    homedir = os.path.expanduser('~')
    datadir = os.path.join(homedir, '.shorttext')
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    targetfilepath = os.path.join(datadir, filename)
    # download if not exist
    if not os.path.exists(os.path.join(datadir, filename)):
        print('Downloading...', file=sys.stderr)
        print(f'Source: {origin}', file=sys.stderr)
        print(f'Target: {targetfilepath}', file=sys.stderr)
        try:
            urlretrieve(origin, targetfilepath)
        except:
            print('Failure to download file!', file=sys.stderr)
            print(sys.exc_info(), file=sys.stderr)
            os.remove(targetfilepath)

    # return
    return open(targetfilepath, 'rb' if asbytes else 'r')


def subjectkeywords() -> dict[str, list[str]]:
    """Return an example dataset of subjects with keywords.

    Returns a small example dataset with three subjects and their
    corresponding keywords, in the training input format.

    Returns:
        A dictionary with subject labels as keys and lists of keywords as values.
    """
    parentdir = Path(__file__).parent
    return retrieve_csvdata_as_dict(parentdir / "shorttext_exampledata.csv")


def inaugural() -> dict[str, list[str]]:
    """Return the Inaugural Addresses of US Presidents.

    Returns an example dataset containing the Inaugural Addresses of all
    Presidents of the United States from George Washington to Barack Obama.

    Each key is formatted as "year-lastname" and the value is a list of
    sentences from the address.

    Returns:
        A dictionary with president identifiers as keys and lists of sentences as values.

    Reference:
        https://www.presidency.us/kisa_exec/inaugural.html
    """
    zfile = zipfile.ZipFile(
        get_or_download_data(
            "USInaugural.zip",
            "https://shorttext-data-northernvirginia.s3.amazonaws.com/trainingdata/USInaugural.zip",
            asbytes=True
        )
    )
    address_jsonstr = zfile.open("addresses.json").read()
    zfile.close()
    return json.loads(address_jsonstr.decode('utf-8'))


def nihreports(txt_col='PROJECT_TITLE', label_col='FUNDING_ICs', sample_size=512):
    """Return an example dataset sampled from NIH RePORT.

    Returns an example dataset from NIH (National Institutes of Health)
    RePORT (Research Portfolio Online Reporting Tools) website.

    Args:
        txt_col: Column for text data. Options: 'PROJECT_TITLE' or 'ABSTRACT_TEXT'.
                Default: 'PROJECT_TITLE'.
        label_col: Column for labels. Options: 'FUNDING_ICs' or 'IC_NAME'.
                Default: 'FUNDING_ICs'.
        sample_size: Number of samples to return. Set to None for all rows. Default: 512.

    Returns:
        A dictionary with IC identifiers as keys and lists of text data as values.

    Reference:
        https://exporter.nih.gov/ExPORTER_Catalog.aspx
        Dataset adapted from the R package textmineR:
        https://cran.r-project.org/web/packages/textmineR/index.html
    """
    # validation
    # txt_col = 'PROJECT_TITLE' or 'ABSTRACT_TEXT'
    # label_col = 'FUNDING_ICs' or 'IC_NAME'
    if not (txt_col in ['PROJECT_TITLE', 'ABSTRACT_TEXT']):
        raise KeyError(f'Undefined text column: {txt_col}. Must be PROJECT_TITLE or ABSTRACT_TEXT.')
    if not (label_col in ['FUNDING_ICs', 'IC_NAME']):
        raise KeyError(f'Undefined label column: {label_col}. Must be FUNDING_ICs or IC_NAME.')

    zfile = zipfile.ZipFile(get_or_download_data('nih_full.csv.zip',
                                                 'https://shorttext-data-northernvirginia.s3.amazonaws.com/trainingdata/nih_full.csv.zip',
                                                 asbytes=True),
                            'r',
                            zipfile.ZIP_DEFLATED)
    nih = pd.read_csv(zfile.open('nih_full.csv'), na_filter=False, usecols=[label_col, txt_col], encoding='cp437')
    zfile.close()
    nb_data = len(nih)
    sample_size = nb_data if sample_size==None else min(nb_data, sample_size)

    classdict = defaultdict(lambda : [])

    for rowidx in np.random.randint(nb_data, size=min(nb_data, sample_size)):
        label = nih.iloc[rowidx, nih.columns.get_loc(label_col)]
        if label_col=='FUNDING_ICs':
            if label=='':
                label = 'OTHER'
            else:
                endpos = label.index(':')
                label = label[:endpos]
        classdict[label] += [nih.iloc[rowidx, nih.columns.get_loc(txt_col)]]

    return dict(classdict)


@deprecated(deprecated_in="4.0.0", removed_in="5.0.0")
def mergedict(dicts: list[dict]) -> dict:
    """Merge multiple training data dictionaries.

    Combines multiple data dictionaries in the training data format
    into a single dictionary.

    Args:
        dicts: List of dictionaries to merge, each with class labels
              as keys and lists of texts as values.

    Returns:
        A merged dictionary with all class labels and texts combined.
    """
    mdict = defaultdict(lambda : [])
    for thisdict in dicts:
        for label in thisdict:
            mdict[label] += thisdict[label]
    return dict(mdict)


def yield_crossvalidation_classdicts(
        classdict: dict[str, list[str]],
        nb_partitions: int,
        shuffle: bool = False
) -> Generator[tuple[dict[str, list[str]], dict[str, list[str]]], None, None]:
    """Yield training and test data partitions for cross-validation.

    Partitions the training data into multiple sets. Each iteration yields
    a (test_dict, train_dict) pair where one partition is used as test
    data and the remaining partitions are combined as training data.

    Args:
        classdict: Training data dictionary with class labels as keys
                 and lists of texts as values.
        nb_partitions: Number of partitions to create.
        shuffle: Whether to shuffle data before partitioning. Default: False.

    Yields:
        Tuples of (test_dict, train_dict) for each partition.
    """
    crossvaldicts = []
    for _ in range(nb_partitions):
        crossvaldicts.append(defaultdict(lambda: []))

    for label in classdict:
        nb_data = len(classdict[label])
        partsize = nb_data / nb_partitions
        sentences = classdict[label] if not shuffle else random.shuffle(sentences)
        for i in range(nb_partitions):
            crossvaldicts[i][label] += sentences[i * partsize:min(nb_data, (i + 1) * partsize)]
    crossvaldicts = [dict(crossvaldict) for crossvaldict in crossvaldicts]

    for i in range(nb_partitions):
        testdict = crossvaldicts[i]
        # traindict = mergedict([crossvaldicts[j] for j in range(nb_partitions) if j != i])
        traindict = reduce(lambda a, b: a | b, [crossvaldicts[j] for j in range(nb_partitions) if j != i])
        yield testdict, traindict
