Spell Correctors
================

There are two types of spell correctors provided: the one described by Peter Norvig, and another by Keisuke Sakaguchi.

>>> import shorttext

Load the training text:

>>> text = urllib2.urlopen('https://norvig.com/big.txt').read()

Norvig
------

>>> norvig_corrector = shorttext.spell.NorvigSpellCorrector()
>>> norvig_corrector.train(text)
>>> norvig_corrector.correct('oranhe')   # gives "orange"

Sakaguchi
---------

>>> scrnn_corrector = shorttext.spell.SCRNNSpellCorrector('JUMBLE-WHOLE')
>>> scrnn_corrector.train(text)
>>> scrnn_corrector.correct('oranhe')   # gives "orange"

Reference
---------

Keisuke Sakaguchi, Kevin Duh, Matt Post, Benjamin Van Durme, "Robsut Wrod Reocginiton via semi-Character Recurrent Neural Networ," arXiv:1608.02214 (2016). [`arXiv
<https://arxiv.org/abs/1608.02214>`_]

Peter Norvig, "How to write a spell corrector." (2016) [`Norvig
<https://norvig.com/spell-correct.html>`_]
