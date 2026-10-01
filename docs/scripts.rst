Console Scripts
===============

This package provides two scripts.

The development of the scripts is *not stable* yet, and more scripts will be added.

shorttext_categorize
--------------------

::

    Usage: shorttext_categorize [OPTIONS] MODEL_FILEPATH

      Perform prediction on short text with a given trained model.

      MODEL_FILEPATH     Path of the trained (compact) model.

    Options:
      --wv PATH                       Path of the pre-trained Word2Vec model, if
                                      needed.
      --vecsize INTEGER               Vector dimensions. (Default: 300)
      --topn INTEGER                  Number of top results to show.
      --inputtext TEXT                Single input text for classification. If
                                      omitted, will enter console mode.
      --type [word2vec|word2vec_nonbinary|fasttext|poincare|poincare_binary]
                                      Type of word-embedding model (default:
                                      word2vec)
      --help                          Show this message and exit.



find_sentences_similarity
-------------------------

::

    Usage: find_sentences_similarity [OPTIONS] MODELPATH

      Find the similarities between two short sentences using Word2Vec.

      MODELPATH    Path of the embedding model

    Options:
      --type [word2vec|fasttext|poincare]
                                      Type of word-embedding model (default:
                                      "word2vec"; other options: "fasttext",
                                      "poincare")
      --help                          Show this message and exit.



Home: :doc:`index`
