Document-Term Matrix
====================

Preparing for the Corpus
------------------------

We can create and handle document-term matrix (DTM) with `shorttext`. Use the dataset of Presidents'
Inaugural Addresses as an example.

>>> import shorttext
>>> usprez = shorttext.data.inaugural()

We have to make each presidents' address to be one document to achieve our purpose. Enter this:

>>> docids = sorted(usprez.keys())
>>> usprez = [' '.join(usprez[docid]) for docid in docids]

Now the variable `usprez` is a list of 56 Inaugural Addresses from George Washington (1789) to
Barack Obama (2009), with the IDs stored in `docids`. We apply the standard text preprocessor and
produce a list of lists (of tokens) (or a corpus in `gensim`):

>>> preprocess = shorttext.utils.standard_text_preprocessor_1()
>>> corpus = [preprocess(address).split(' ') for address in usprez]

Then now the variable `corpus` is a list of lists of tokens. For example,

>>> corpus[0]     # shows all the preprocessed tokens of the first Presidential Inaugural Addresses

Using Class `DocumentTermMatrix`
--------------------------------

With the corpus ready in this form, we can create a `DocumentTermMatrix` class for DTM by:

>>> usprez_dtm = shorttext.utils.DocumentTermMatrix(corpus, docids=docids)

One can get the document frequency of any token (the number of documents that the given
token is in) by:

>>> usprez_dtm.get_doc_frequency('peopl')  # gives 54, the document frequency of the token "peopl"

or the total term frequencies (the total number of occurrences of the given tokens in all documents) by:

>>> usprez_dtm.get_total_termfreq('justic')   # gives 134.0, the total term frequency of the token "justic"

or the term frequency for a token in a given document by:

>>> usprez_dtm.get_termfreq('2009-Obama', 'chang')    # gives 2.0

We can also query the number of occurrences of a particular word of all documents,
stored in a dictionary, by:

>>> usprez_dtm.get_token_occurences('god')

Of course, we can always reweigh the counts above (except document frequency) by imposing
tf-idf while creating the instance of the class by enforceing `tfidf` to be `True`:

>>> usprez_dtm = shorttext.utils.DocumentTermMatrix(corpus, docids=docids, tfidf=True)

To save the class, enter:

>>> usprez_dtm.save_compact_model('/path/to/whatever.bin')

To load this class later, enter:

>>> usprez_dtm2 = shorttext.utils.load_DocumentTermMatrix('/path/to/whatever.bin')

Reference
---------

Christopher Manning, Hinrich Schuetze, *Foundations of Statistical Natural Language Processing* (Cambridge, MA: MIT Press, 1999). [`MIT Press
<https://mitpress.mit.edu/books/foundations-statistical-natural-language-processing>`_]

"Document-Term Matrix: Text Mining in R and Python," *Everything About Data Analytics*, WordPress (2018). [`WordPress
<https://datawarrior.wordpress.com/2018/01/22/document-term-matrix-text-mining-in-r-and-python/>`_]

Home: :doc:`index`