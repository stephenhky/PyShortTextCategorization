Document-Term Matrix
====================

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

With the corpus ready in this form, we can create a `DocumentTermMatrix` class for DTM by:

>>> usprez_dtm = shorttext.utils.DocumentTermMatrix(corpus, docids=docids)

With the DTM class set up, you can view the matrix in terms of a Pandas DataFrame:

>>> tbl = usprez_dtm.generate_dtm_dataframe()
>>> tbl

::

                           accomplish  accordingli  acknowledg  acquit  act  \
    1789-Washington  12.0         1.0          1.0         1.0     1.0  1.0
    1793-Washington   3.0         0.0          0.0         0.0     0.0  1.0
    1797-Adams       15.0         0.0          0.0         1.0     0.0  0.0
    1801-Jefferson   24.0         0.0          0.0         1.0     0.0  0.0
    1805-Jefferson    3.0         0.0          0.0         1.0     0.0  4.0

                     actual  actuat  add  addit   ...     starv  sweatshop  swill  \
    1789-Washington     1.0     1.0  1.0    1.0   ...       0.0        0.0    0.0
    1793-Washington     0.0     0.0  0.0    0.0   ...       0.0        0.0    0.0
    1797-Adams          0.0     0.0  2.0    0.0   ...       0.0        0.0    0.0
    1801-Jefferson      0.0     0.0  0.0    0.0   ...       0.0        0.0    0.0
    1805-Jefferson      0.0     0.0  2.0    0.0   ...       0.0        0.0    0.0

                     taker  tank  tast  tirelessli  unclench  whip  whisper
    1789-Washington    0.0   0.0   0.0         0.0       0.0   0.0      0.0
    1793-Washington    0.0   0.0   0.0         0.0       0.0   0.0      0.0
    1797-Adams         0.0   0.0   0.0         0.0       0.0   0.0      0.0
    1801-Jefferson     0.0   0.0   0.0         0.0       0.0   0.0      0.0
    1805-Jefferson     0.0   0.0   0.0         0.0       0.0   0.0      0.0

    [5 rows x 5404 columns]

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

Home: :doc:`index`