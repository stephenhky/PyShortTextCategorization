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

>>> trainclassdict = shorttext.data.nihreports()

This will output a similar dictionary with FUNDING_IC (Institutes and Centers in NIH)
 as the class labels, and PROJECT_TITLE (title of the funded projects)
as the short texts under the corresponding labels. This dictionary has 512 projects in total,
randomly drawn from the original data.

However, there are other configurations:

.. autofunction:: shorttext.data.nihreports

If `sample_size` is specified to be `None`, all the data will be retrieved without sampling.

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
