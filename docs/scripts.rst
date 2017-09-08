Console Scripts
===============

This package provides two scripts.

The development of the scripts is *not stable* yet, and more scripts will be added.

ShortTextCategorizerConsole
---------------------------

::

    usage: ShortTextCategorizerConsole [-h] [--wv WV] [--vecsize VECSIZE]
                                       [--topn TOPN]
                                       model_filepath

    Perform prediction on short text with a given trained model.

    positional arguments:
      model_filepath     Path of the trained (compact) model.

    optional arguments:
      -h, --help         show this help message and exit
      --wv WV            Path of the pre-trained Word2Vec model. (None if not
                         needed)
      --vecsize VECSIZE  Vector dimensions. (Default: 100)
      --topn TOPN        Number of top-scored results displayed. (Default: 10)


ShortTextWord2VecSimilarity
---------------------------

::

    usage: ShortTextWord2VecSimilarity [-h] [--vecsize VECSIZE] word2vec_modelpath

    Find the similarity between two short sentences using Word2Vec.

    positional arguments:
      word2vec_modelpath  Path of the Word2Vec model

    optional arguments:
      -h, --help          show this help message and exit
      --vecsize VECSIZE   Vector dimensions. (Default: 100)

Home: :doc:`index`
