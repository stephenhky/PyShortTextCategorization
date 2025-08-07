Spell Correctors
================

This package supports the use of spell correctors, because typos are very common in relatively short text data.

There are two types of spell correctors provided: the one described by Peter Norvig (using n-grams Bayesian method),
and another by Keisuke Sakaguchi and his colleagues (using semi-character level recurrent neural network).

>>> import shorttext

We use the Norvig's training corpus as an example. To load it,

>>> from urllib.request import urlopen
>>> text = urlopen('https://norvig.com/big.txt').read()

The developer just has to instantiate the spell corrector, and then train it with a corpus to get a correction model.
Then one can use it for correction.

Norvig
------

Peter Norvig described a spell corrector based on Bayesian approach and edit distance. You can refer to his blog for
more information.

>>> norvig_corrector = shorttext.spell.NorvigSpellCorrector()
>>> norvig_corrector.train(text)
>>> norvig_corrector.correct('oranhe')   # gives "orange"

.. automodule:: shorttext.spell.norvig
   :members:



Reference
---------

Peter Norvig, "How to write a spell corrector." (2016) [`Norvig
<https://norvig.com/spell-correct.html>`_]
