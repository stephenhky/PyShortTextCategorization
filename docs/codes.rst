API
===

Complete API reference for the shorttext library.

.. contents::
   :local:
   :backlinks: none

Top-Level Modules
-----------------

.. automodule:: shorttext
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.smartload
   :members:
   :undoc-members:
   :show-inheritance:

Classifiers
-----------

.. automodule:: shorttext.classifiers
   :members:
   :undoc-members:
   :show-inheritance:

Base Classifier
^^^^^^^^^^^^^^^

.. automodule:: shorttext.classifiers.base
   :members:
   :undoc-members:
   :show-inheritance:

Bag-of-Words Classifiers
^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: shorttext.classifiers.bow
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.classifiers.bow.topic.SkLearnClassification
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.classifiers.bow.topic.TopicVectorDistanceClassification
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.classifiers.bow.maxent.MaxEntClassification
   :members:
   :undoc-members:
   :show-inheritance:

Embedding-Based Classifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: shorttext.classifiers.embed
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.classifiers.embed.sumvec.SumEmbedVecClassification
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.classifiers.embed.sumvec.VarNNSumEmbedVecClassification
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.classifiers.embed.sumvec.frameworks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.classifiers.embed.nnlib.VarNNEmbedVecClassification
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.classifiers.embed.nnlib.frameworks
   :members:
   :undoc-members:
   :show-inheritance:

Generators
----------

.. automodule:: shorttext.generators
   :members:
   :undoc-members:
   :show-inheritance:

Bag-of-Words Generators
^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: shorttext.generators.bow
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.generators.bow.GensimTopicModeling
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.generators.bow.LatentTopicModeling
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.generators.bow.AutoEncodingTopicModeling
   :members:
   :undoc-members:
   :show-inheritance:

Sequence-to-Sequence Generators
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: shorttext.generators.seq2seq
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.generators.seq2seq.s2skeras
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.generators.seq2seq.charbaseS2S
   :members:
   :undoc-members:
   :show-inheritance:

Character-Based Generators
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: shorttext.generators.charbase
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.generators.charbase.char2vec
   :members:
   :undoc-members:
   :show-inheritance:

Metrics
-------

.. automodule:: shorttext.metrics
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.metrics.dynprog
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.metrics.dynprog.jaccard
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.metrics.dynprog.dldist
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.metrics.dynprog.lcp
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.metrics.wasserstein
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.metrics.wasserstein.wordmoverdist
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.metrics.embedfuzzy
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.metrics.embedfuzzy.jaccard
   :members:
   :undoc-members:
   :show-inheritance:

Spell Correction
----------------

.. automodule:: shorttext.spell
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.spell.basespellcorrector
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.spell.norvig
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.spell.editor
   :members:
   :undoc-members:
   :show-inheritance:

Stacking
--------

.. automodule:: shorttext.stack
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.stack.stacking
   :members:
   :undoc-members:
   :show-inheritance:

Data
----

.. automodule:: shorttext.data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.data.data_retrieval
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: shorttext.utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.utils.kerasmodel_io
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.utils.compactmodel_io
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.utils.gensim_corpora
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.utils.textpreprocessing
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.utils.wordembed
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.utils.compute
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.utils.misc
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.utils.dtm
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.utils.classification_exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Schemas
-------

.. automodule:: shorttext.schemas
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.schemas.models
   :members:
   :undoc-members:
   :show-inheritance:

CLI
---

.. automodule:: shorttext.cli
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.cli.categorization
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: shorttext.cli.wordembedsim
   :members:
   :undoc-members:
   :show-inheritance:


Home: :doc:`index`