Data Preparation
================

This package deals with short text. While the text data for predictions or
classifications are simply `str` or list of `str`, the training data does
take a specific format, in terms of `dict`, the Python dictionary (or hash
map). The package provides two sets of data as an example.

Example Training Data 1: Subject Keywords
-----------------------------------------

The first example dataset is about the subject keywords, which can be loaded by:

>>> import shorttext.data.subjectkeywords()

Example Training Data 2: NIH Reporters
--------------------------------------

The second example dataset is from NIH Reporters.

