Console Scripts
===============

This package provides two scripts. If the user gives a trained model,
and a pre-trained word-embedding model, one can perform classification
on a file of short texts, or run a command-line console.

The scripts are *not stable* yet, and more scripts will be added.

ShortTextDNNCategorizerConsole
------------------------------

::

    usage: ShortTextDNNCategorizerConsole [-h] [--ngram NGRAM] [--test]
                                          [--topn TOPN]
                                          input_nameprefix algo wvmodel_path

    Perform prediction on short text.with a given trained model.

    positional arguments:
      input_nameprefix  Prefix of the path of input model.
      algo              Algorithm architecture. (Options: sumword2vec (Summed
                        Embedded Vectors), vnn (Neural Network on Embedded
                        Vectors)
      wvmodel_path      Path of the pre-trained Word2Vec model.

    optional arguments:
      -h, --help        show this help message and exit
      --ngram NGRAM     n-gram, used in convolutional neural network only.
                        (Default: 2)
      --test            Checked if the test input contains the label, and metrics
                        will be output. (Default: False)
      --topn TOPN       Number of top-scored results displayed. (Default: 10)


Home: :doc:`index`
