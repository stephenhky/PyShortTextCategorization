Console Scripts
===============

This package provides two scripts.

The development of the scripts is *not stable* yet, and more scripts will be added.

ShortTextCategorizerConsole
---------------------------

::

    usage: ShortTextCategorizerConsole [-h] [--wv WV] [--vecsize VECSIZE]
                                       [--topn TOPN] [--inputtext INPUTTEXT]
                                       model_filepath

    Perform prediction on short text with a given trained model.

    positional arguments:
      model_filepath        Path of the trained (compact) model.

    optional arguments:
      -h, --help            show this help message and exit
      --wv WV               Path of the pre-trained Word2Vec model. (None if not
                            needed)
      --vecsize VECSIZE     Vector dimensions. (Default: 300)
      --topn TOPN           Number of top-scored results displayed. (Default: 10)
      --inputtext INPUTTEXT
                            single input text for classification. Run console if
                            set to None. (Default: None)


ShortTextWordEmbedSimilarity
----------------------------

::

    usage: ShortTextWordEmbedSimilarity [-h] [--type TYPE] modelpath

    Find the similarities between two short sentences using Word2Vec.

    positional arguments:
      modelpath    Path of the Word2Vec model

    optional arguments:
      -h, --help   show this help message and exit
      --type TYPE  Type of word-embedding model (default: "word2vec"; other
                   options: "fasttext", "poincare")


Home: :doc:`index`
