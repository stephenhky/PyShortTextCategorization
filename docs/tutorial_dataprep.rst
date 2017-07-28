Data Preparation
================

This package deals with short text. While the text data for predictions or
classifications are simply `str` or list of `str`, the training data does
take a specific format, in terms of `dict`, the Python dictionary (or hash
map). The package provides two sets of data as an example.

Example Training Data 1: Subject Keywords
-----------------------------------------

The first example dataset is about the subject keywords, which can be loaded by:

>>> trainclassdict = shorttext.data.subjectkeywords()

This returns a dictionary, with keys being the label and the values being lists of
the subject keywords, as below:

::

    {'mathematics': ['linear algebra', 'topology', 'algebra', 'calculus',
      'variational calculus', 'functional field', 'real analysis', 'complex analysis',
      'differential equation', 'statistics', 'statistical optimization', 'probability',
      'stochastic calculus', 'numerical analysis', 'differential geometry'],
     'physics': ['renormalization', 'classical mechanics', 'quantum mechanics',
      'statistical mechanics', 'functional field', 'path integral',
      'quantum field theory', 'electrodynamics', 'condensed matter',
      'particle physics', 'topological solitons', 'astrophysics',
      'spontaneous symmetry breaking', 'atomic molecular and optical physics',
      'quantum chaos'],
     'theology': ['divine providence', 'soteriology', 'anthropology', 'pneumatology', 'Christology',
      'Holy Trinity', 'eschatology', 'scripture', 'ecclesiology', 'predestination',
      'divine degree', 'creedal confessionalism', 'scholasticism', 'prayer', 'eucharist']}

Example Training Data 2: NIH RePORT
-----------------------------------

The second example dataset is from NIH RePORT (Research Portfolio Online Reporting Tools).
The data can be downloaded from its `ExPORTER
<https://exporter.nih.gov/about.aspx>`_ page. The current data in this package was directly
adapted from Thomas Jones' `textMineR
<https://github.com/TommyJones/textmineR>`_ R package.

Enter:

>>> trainclassdict = shorttext.data.nihreports()

Upon the installation of the package, the NIH RePORT data are still not
installed. But the first time it was ran, it will be downloaded from the Internet.

This will output a similar dictionary with FUNDING_IC (Institutes and Centers in NIH)
 as the class labels, and PROJECT_TITLE (title of the funded projects)
as the short texts under the corresponding labels. This dictionary has 512 projects in total,
randomly drawn from the original data.

However, there are other configurations:

nihreports(txt_col='PROJECT_TITLE', label_col='FUNDING_ICs', sample_size=512)
    Return an example data set, sampled from NIH RePORT (Research Portfolio
    Online Reporting Tools).

    Return an example data set from NIH (National Institutes of Health),
    data publicly available from their RePORT
    website. (`link
    <https://exporter.nih.gov/ExPORTER_Catalog.aspx>`_).
    The data is with `txt_col` being either project titles ('PROJECT_TITLE')
    or proposal abstracts ('ABSTRACT_TEXT'), and label_col being the names of the ICs (Institutes or Centers),
    with 'IC_NAME' the whole form, and 'FUNDING_ICs' the abbreviated form).

    Dataset directly adapted from the NIH data from `R` package `textmineR
    <https://cran.r-project.org/web/packages/textmineR/index.html>`_.

    :param txt_col: column for the text (Default: 'PROJECT_TITLE')
    :param label_col: column for the labels (Default: 'FUNDING_ICs')
    :param sample_size: size of the sample. Set to None if all rows. (Default: 512)
    :return: example data set
    :type txt_col: str
    :type label_col: str
    :type sample_size: int
    :rtype: dict

If `sample_size` is specified to be `None`, all the data will be retrieved without sampling.

Example Training Data 3: Inaugural Addresses
--------------------------------------------

This contains all the Inaugural Addresses of all the Presidents of the United States, from
George Washington to Barack Obama. Upon the installation of the package, the Inaugural Addresses
data are still not installed. But the first time it was ran, it will be downloaded from the Internet.

The addresses are available publicly, and I extracted them from `nltk
<http://www.nltk.org/>`_ package.

Enter:

>>> trainclassdict = shorttext.data.inaugural()


User-Provided Training Data
---------------------------

Users can provide their own training data. If it is already in JSON format, it can be loaded easily
with standard Python's `json` package, or by calling:

>>> trainclassdict = shorttext.data.retrieve_jsondata_as_dict('/path/to/file.json')

However, if it is in CSV format, it has to obey the rules:

- there is a heading; and
- there are at least two columns: first the labels, and second the short text under the labels (everything being the second column will be neglected).

An excerpt of this type of data is as follow:

::

    subject,content
    mathematics,linear algebra
    mathematics,topology
    mathematics,algebra
    ...
    physics,spontaneous symmetry breaking
    physics,atomic molecular and optical physics
    physics,quantum chaos
    ...
    theology,divine providence
    theology,soteriology
    theology,anthropology

To load this data file, just enter:

>>> trainclassdict = shorttext.data.retrieve_csvdata_as_dict('/path/to/file.csv')

Home: :doc:`index`
