Console Scripts
===============

This package provides two scripts. If the user gives a trained compact model,
and a pre-trained word-embedding model, one can perform classification
on short texts on a simple console.

The scripts are *not stable* yet, and more scripts will be added.

ShortTextCategorizerConsole
---------------------------

::

    usage: ShortTextCategorizerConsole [-h] [--wv WV] [--topn TOPN] model_filepath

    Perform prediction on short text with a given trained model.

    positional arguments:
      model_filepath  Path of the trained (compact) model.

    optional arguments:
      -h, --help      show this help message and exit
      --wv WV         Path of the pre-trained Word2Vec model. (None if not needed)
      --topn TOPN     Number of top-scored results displayed. (Default: 10)


Home: :doc:`index`
