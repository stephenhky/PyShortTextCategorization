Spell Correctors
================

This package supports the use of spell correctors, because typos are very common in relatively short text data.

There are two types of spell correctors provided: the one described by Peter Norvig (using n-grams Bayesian method),
and another by Keisuke Sakaguchi and his colleagues (using semi-character level recurrent neural network).

>>> import shorttext

We use the Norvig's training corpus as an example. To load it, in Python 2.7, enter

>>> from urllib2 import urlopen
>>> text = urlopen('https://norvig.com/big.txt').read()

Or in Python 3.5 and 3.6,

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

Sakaguchi (SCRNN - semi-character recurrent neural network)
-----------------------------------------------------------

Keisuke Sakaguchi and his colleagues developed this spell corrector with the insight that most of the typos happen
in between the spellings. They developed a recurrent neural network that trains possible change within the spellings. There are
six modes:

- JUMBLE-WHOLE
- JUMBLE-BEG
- JUMBLE-END
- JUMBLE-INT
- NOISE-INSERT
- NOISE-DELETE
- NOISE-REPLACE

The original intent of their work was not to invent a new spell corrector but to study the "Cmabrigde Uinervtisy" effect,
but it is nice to see how it can be implemented as a spell corrector.

>>> scrnn_corrector = shorttext.spell.SCRNNSpellCorrector('JUMBLE-WHOLE')
>>> scrnn_corrector.train(text)
>>> scrnn_corrector.correct('oranhe')   # gives "orange"

We can persist the SCRNN corrector for future use:

>>> scrnn_corrector.save_compact_model('/path/to/spellscrnn.bin')

To load,

>>> corrector = shorttext.spell.loadSCRNNSpellCorrector('/path/to/spellscrnn.bin')

Reference
---------

Keisuke Sakaguchi, Kevin Duh, Matt Post, Benjamin Van Durme, "Robsut Wrod Reocginiton via semi-Character Recurrent Neural Networ," arXiv:1608.02214 (2016). [`arXiv
<https://arxiv.org/abs/1608.02214>`_]

Peter Norvig, "How to write a spell corrector." (2016) [`Norvig
<https://norvig.com/spell-correct.html>`_]
