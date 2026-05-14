Console Scripts
===============

This package provides two scripts.

The development of the scripts is *not stable* yet, and more scripts will be added.

ShortTextCategorizerConsole
---------------------------

::

    usage: ShortTextCategorizerConsole [-h] [--wv WV] [--vecsize VECSIZE]
                                       [--topn TOPN] [--inputtext INPUTTEXT]
                                       [--type {word2vec,word2vec_nonbinary,fasttext,poincare,poincare_binary}]
                                       model_filepath

    Perform prediction on short text with a given trained model.

    positional arguments:
      model_filepath        Path of the trained (compact) model.

    options:
      -h, --help            show this help message and exit
      --wv WV               Path of the pre-trained Word2Vec model.
      --vecsize VECSIZE     Vector dimensions. (Default: 300)
      --topn TOPN           Number of top results to show.
      --inputtext INPUTTEXT
                            Single input text for classification. If omitted, will
                            enter console mode.
      --type {word2vec,word2vec_nonbinary,fasttext,poincare,poincare_binary}
                            Type of word-embedding model (default: word2vec)


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
