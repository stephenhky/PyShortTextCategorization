Text Preprocessing
==================

Standard Preprocessor
---------------------

When the bag-of-words (BOW) model is used to represent the content, it is essential to
specify how the text is preprocessed before it is passed to the trainers or the
classifiers.

This package provides a standard way of text preprocessing, which goes through the
following steps:

- removing special characters,
- removing numerals,
- converting all alphabets to lower cases,
- removing stop words, and
- stemming the words (using Porter stemmer).

To do this, load the preprocesser generator:

>>> from shorttext.utils import standard_text_preprocessor_1

Then define the preprocessor, a function, by just calling:

>>> preprocessor1 = standard_text_preprocessor_1()

It is a function that perform the preprocessing in the steps above:

>>> preprocessor1('Maryland Blue Crab')  # output:  'maryland blue crab'
>>> preprocessor1('filing electronic documents and goes home. eat!!!')   # output: 'file electron document goe home eat'

Customized Text Preprocessor
----------------------------

The standard preprocessor is good for many general natural language processing tasks,
but some users may want to define their own preprocessors for their own purposes.
This preprocessor is used in topic modeling, and is desired to be *a function that takes
a string, and returns a string*.

If the user wants to develop a preprocessor that contains a few steps, he can make it by providing
the pipeline, which is a list of functions that input a string and return a string. For example,
let's develop a preprocessor that 1) convert it to base form if it is a verb, or keep it original;
2) convert it to upper case; and 3) tag the number of characters after each token.

Load the function that generates the preprocessor function:

>>> from shorttext.utils import text_preprocessor

Initialize a WordNet lemmatizer using `nltk`:

>>> from nltk.stem import WordNetLemmatizer
>>> lemmatizer = WordNetLemmatizer()

Define the pipeline. Functions for each of the steps are:

>>> step1fcn = lambda s: ' '.join([lemmatizer.lemmatize(s1) for s1 in s.split(' ')])
>>> step2fcn = lambda s: s.upper()
>>> step3fcn = lambda s: ' '.join([s1+'-'+str(len(s1)) for s1 in s.split(' ')])

Then the pipeline is:

>>> pipeline = [step1fcn, step2fcn, step3fcn]

The preprocessor function can be generated with the defined pipeline:

>>> preprocessor2 = text_preprocessor(pipeline)

The function `preprocessor2` is a function that input a string and returns a string.
Some examples are:

>>> preprocessor2('Maryland blue crab in Annapolis')  # output: 'MARYLAND-8 BLUE-4 CRAB-4 IN-2 ANNAPOLIS-9'
>>> preprocessor2('generative adversarial networks')  # output: 'GENERATIVE-10 ADVERSARIAL-11 NETWORK-7'

Tokenization
------------

Users are free to choose any tokenizer they wish. In `shorttext`, the tokenizer is
implemented with `spaCy`, and can be called:

>>> shorttext.utils.tokenize('Maryland blue crab')   # output: ['Maryland', 'blue', 'crab']

Reference
---------

Christopher Manning, Hinrich Schuetze, *Foundations of Statistical Natural Language Processing* (Cambridge, MA: MIT Press, 1999). [`MIT Press
<https://mitpress.mit.edu/books/foundations-statistical-natural-language-processing>`_]

"R or Python on Text Mining," *Everything About Data Analytics*, WordPress (2015). [`WordPress
<https://datawarrior.wordpress.com/2015/08/12/codienerd-1-r-or-python-on-text-mining>`_]

Home: :doc:`index`