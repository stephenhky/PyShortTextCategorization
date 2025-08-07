Word Embedding Models
=====================

Word2Vec
--------

The most commonly used word-embedding model is Word2Vec. Its model can be downloaded from
their page. To load the model, call:

>>> import shorttextfrom src import shorttext
>>> wvmodel = shorttext.utils.load_word2vec_model('/path/to/GoogleNews-vectors-negative300.bin.gz')

It is a binary file, and the default is set to be
their page. To load the model, call:

>>> import shorttextfrom src import shorttext
>>> wvmodel = shorttext.utils.load_word2vec_model('/path/to/GoogleNews-vectors-negative300.bin.gz')

It is a binary file, and the default is set to be
their page. To load the model, call:

>>> import shorttext
from src import shorttext
>>> wvmodel = shorttext.utils.load_word2vec_model('/path/to/GoogleNews-vectors-negative300.bin.gz')

It is a binary file, and the default is set to be
their page. To load the model, call:

>>> import shorttext
from src import shorttext
>>> wvmodel = shorttext.utils.load_word2vec_model('/path/to/GoogleNews-vectors-negative300.bin.gz')

It is a binary file, and the default is set to be
their page. To load the model, call:

>>> import shorttext
from src import shorttext
>>> wvmodel = shorttext.utils.load_word2vec_model('/path/to/GoogleNews-vectors-negative300.bin.gz')

It is a binary file, and the default is set to be
their page. To load the model, call:

>>> import shorttext
from src import shorttext
>>> wvmodel = shorttext.utils.load_word2vec_model('/path/to/GoogleNews-vectors-negative300.bin.gz')

It is a binary file, and the default is set to be
their page. To load the model, call:

>>>
from src import shorttext
>>> wvmodel = shorttext.utils.load_word2vec_model('/path/to/GoogleNews-vectors-negative300.bin.gz')

It is a binary file, and the default is set to be
their page. To load the model, call:

>>>
from src import shorttext
>>> wvmodel = shorttext.utils.load_word2vec_model('/path/to/GoogleNews-vectors-negative300.bin.gz')

It is a binary file, and the default is set to be
their page. To load the model, call:

>>>
from src import shorttext
>>> wvmodel = shorttext.utils.load_word2vec_model('/path/to/GoogleNews-vectors-negative300.bin.gz')

It is a binary file, and the default is set to be
their page. To load the model, call:

>>>
from src import shorttext
>>> wvmodel = shorttext.utils.load_word2vec_model('/path/to/GoogleNews-vectors-negative300.bin.gz')

It is a binary file, and the default is set to be
their page. To load the model, call:

>>>
from src import shorttext
>>> wvmodel = shorttext.utils.load_word2vec_model('/path/to/GoogleNews-vectors-negative300.bin.gz')

It is a binary file, and the default is set to be
their page. To load the model, call:

>>>
from src import shorttext
>>> wvmodel = shorttext.utils.load_word2vec_model('/path/to/GoogleNews-vectors-negative300.bin.gz')

It is a binary file, and the default is set to be
their page. To load the model, call:

>>> import shorttext
>>> wvmodel = shorttext.utils.load_word2vec_model('/path/to/GoogleNews-vectors-negative300.bin.gz')

It is a binary file, and the default is set to be `binary=True`.

.. automodule:: shorttext.utils.wordembed
   :members: load_word2vec_model

It is equivalent to calling,

>>> import gensim
>>> wvmodel = gensim.models.KeyedVectors.load_word2vec_format('/path/to/GoogleNews-vectors-negative300.bin.gz', binary=True)


Word2Vec is a neural network model that embeds words into semantic vectors that carry semantic meaning.
It is easy to extract the vector of a word, like for the word 'coffee':

>>> wvmodel['coffee']   # an ndarray for the word will be output

One can find the most similar words to 'coffee' according to this model:

>>> wvmodel.most_similar('coffee')

which outputs:

::

    [(u'coffees', 0.721267819404602),
     (u'gourmet_coffee', 0.7057087421417236),
     (u'Coffee', 0.6900454759597778),
     (u'o_joe', 0.6891065835952759),
     (u'Starbucks_coffee', 0.6874972581863403),
     (u'coffee_beans', 0.6749703884124756),
     (u'latt\xe9', 0.664122462272644),
     (u'cappuccino', 0.662549614906311),
     (u'brewed_coffee', 0.6621608138084412),
     (u'espresso', 0.6616827249526978)]

Or if you want to find the cosine similarity between 'coffee' and 'tea', enter:

>>> wvmodel.similarity('coffee', 'tea')   # outputs: 0.56352921707810621

Semantic meaning can be reflected by their differences. For example, we can vaguely
say `Francis` - `Paris` = `Taiwan` - `Taipei`, or `man` - `actor` = `woman` - `actress`.
Define first the cosine similarity for readability:

>>> from scipy.spatial.distance import cosine
>>> similarity = lambda u, v: 1-cosine(u, v)

Then

>>> similarity(wvmodel['France'] + wvmodel['Taipei'] - wvmodel['Taiwan'], wvmodel['Paris'])  # outputs: 0.70574580801216202
>>> similarity(wvmodel['woman'] + wvmodel['actor'] - wvmodel['man'], wvmodel['actress'])  # outputs: 0.876354245612604

GloVe
-----

Stanford NLP Group developed a similar word-embedding algorithm, with a good theory explaining how
it works. It is extremely similar to Word2Vec.

One can convert a text-format GloVe model into a text-format Word2Vec model. More information can be found
in the documentation of `gensim`: `Converting GloVe to Word2Vec
<https://radimrehurek.com/gensim/scripts/glove2word2vec.html>`_

FastText
--------

FastText is a similar word-embedding model from Facebook. You can download pre-trained models here:

`Pre-trained word vectors
<https://github.com/facebookresearch/fastText/blob/master/docs/pretrained-vectors.md>`_

To load a pre-trained FastText model, run:

>>> import shorttext
>>> ftmodel = shorttext.utils.load_fasttext_model('/path/to/model.bin')

And it is used exactly the same way as Word2Vec.

.. automodule:: shorttext.utils.wordembed
   :members: load_fasttext_model

Poincaré Embeddings
-------------------

Poincaré embeddings is a new embedding that learns both semantic similarity and hierarchical structures. To load a
pre-trained model, run:

>>> import shorttext
>>> pemodel = shorttext.utils.load_poincare_model('/path/to/model.txt')

For preloaded word-embedding models, please refer to :doc:`tutorial_wordembed`.

.. automodule:: shorttext.utils.wordembed
   :members: load_poincare_model

BERT
----

BERT_ (Bidirectional Transformers for Language Understanding)
is a transformer-based language model. This package supports tokens
and sentence embeddings using pre-trained language models, supported
by the package written by HuggingFace_. In `shorttext`, to run:

>>> from shorttext import WrappedBERTEncoder
>>> encoder = WrappedBERTEncoder()   # the default model and tokenizer are loaded
>>> sentences_embedding, tokens_embedding, tokens = encoder.encode_sentences(['The car should turn right.', 'The answer is right.'])

The third line returns the embeddings of all sentences, embeddings of all tokens in each sentence,
and the tokens (with

>>> from shorttext import WrappedBERTEncoder
>>> encoder = WrappedBERTEncoder()   # the default model and tokenizer are loaded
>>> sentences_embedding, tokens_embedding, tokens = encoder.encode_sentences(['The car should turn right.', 'The answer is right.'])

The third line returns the embeddings of all sentences, embeddings of all tokens in each sentence,
and the tokens (with

>>> from shorttext import WrappedBERTEncoder
>>> encoder = WrappedBERTEncoder()   # the default model and tokenizer are loaded
>>> sentences_embedding, tokens_embedding, tokens = encoder.encode_sentences(['The car should turn right.', 'The answer is right.'])

The third line returns the embeddings of all sentences, embeddings of all tokens in each sentence,
and the tokens (with

>>> from shorttext import WrappedBERTEncoder
>>> encoder = WrappedBERTEncoder()   # the default model and tokenizer are loaded
>>> sentences_embedding, tokens_embedding, tokens = encoder.encode_sentences(['The car should turn right.', 'The answer is right.'])

The third line returns the embeddings of all sentences, embeddings of all tokens in each sentence,
and the tokens (with

>>> from shorttext import WrappedBERTEncoder
>>> encoder = WrappedBERTEncoder()   # the default model and tokenizer are loaded
>>> sentences_embedding, tokens_embedding, tokens = encoder.encode_sentences(['The car should turn right.', 'The answer is right.'])

The third line returns the embeddings of all sentences, embeddings of all tokens in each sentence,
and the tokens (with

>>> from shorttext import WrappedBERTEncoder
>>> encoder = WrappedBERTEncoder()   # the default model and tokenizer are loaded
>>> sentences_embedding, tokens_embedding, tokens = encoder.encode_sentences(['The car should turn right.', 'The answer is right.'])

The third line returns the embeddings of all sentences, embeddings of all tokens in each sentence,
and the tokens (with

>>> from shorttext.utils import WrappedBERTEncoder
>>> encoder = WrappedBERTEncoder()   # the default model and tokenizer are loaded
>>> sentences_embedding, tokens_embedding, tokens = encoder.encode_sentences(['The car should turn right.', 'The answer is right.'])

The third line returns the embeddings of all sentences, embeddings of all tokens in each sentence,
and the tokens (with `CLS` and `SEP`) included. Unlike previous embeddings,
token embeddings depend on the context; in the above example, the embeddings of the
two "right"'s are different as they have different meanings.

The default BERT models and tokenizers are `bert-base_uncase`.
If you want to use others, refer to `HuggingFace's model list
<https://huggingface.co/models>`_ .

.. autoclass:: shorttext.utils.transformers.BERTObject
   :members:

.. autoclass:: shorttext.utils.transformers.WrappedBERTEncoder
   :members:


Other Functions
---------------

.. automodule:: shorttext.utils.wordembed
   :members: shorttext_to_avgvec


Links
-----

- Word2Vec_
- GloVe_
- FastText_
- BERT_
- HuggingFace_

Reference
---------

Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding," arXiv:1810.04805 (2018). [`arXiv
<https://arxiv.org/abs/1810.04805>`_]

Jayant Jain, "Implementing Poincaré Embeddings," RaRe Technologies (2017). [`RaRe
<https://rare-technologies.com/implementing-poincare-embeddings/#h2-2>`_]

Jeffrey Pennington, Richard Socher, Christopher D. Manning, “GloVe: Global Vectors for Word Representation,” *Empirical Methods in Natural Language Processing (EMNLP)*, pp. 1532-1543 (2014). [`PDF
<http://www.aclweb.org/anthology/D14-1162>`_]

Maximilian Nickel, Douwe Kiela, "Poincaré Embeddings for Learning Hierarchical Representations," arXiv:1705.08039 (2017). [`arXiv
<https://arxiv.org/abs/1705.08039>`_]

Piotr Bojanowski, Edouard Grave, Armand Joulin, Tomas Mikolov, "Enriching Word Vectors with Subword Information," arXiv:1607.04606 (2016). [`arXiv
<https://arxiv.org/abs/1607.04606>`_]

Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean, “Efficient Estimation of Word Representations in Vector Space,” *ICLR* 2013 (2013). [`arXiv
<https://arxiv.org/abs/1301.3781>`_]

Radim Řehůřek, "Making sense of word2vec," RaRe Technologies (2014). [`RaRe
<https://rare-technologies.com/making-sense-of-word2vec/>`_]

"Probabilistic Theory of Word Embeddings: GloVe," *Everything About Data Analytics*, WordPress (2016). [`WordPress
<https://datawarrior.wordpress.com/2016/07/25/probabilistic-theory-of-word-embeddings-glove/>`_]

"Toying with Word2Vec," *Everything About Data Analytics*, WordPress (2015). [`WordPress
<https://datawarrior.wordpress.com/2015/10/25/codienerd-2-toying-with-word2vec/>`_]

"Word-Embedding Algorithms," *Everything About Data Analytics*, WordPress (2016). [`WordPress
<https://datawarrior.wordpress.com/2016/05/15/word-embedding-algorithms/>`_]

Home: :doc:`index`

.. _Word2Vec: https://code.google.com/archive/p/word2vec/
.. _GloVe: http://nlp.stanford.edu/projects/glove/
.. _FastText: https://github.com/facebookresearch/fastText
.. _BERT: https://arxiv.org/abs/1810.04805
.. _HuggingFace: https://huggingface.co/