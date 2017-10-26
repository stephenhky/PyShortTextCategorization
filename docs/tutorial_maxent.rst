Maximum Entropy (MaxEnt) Classifier
===================================

Maxent
------

Maximum entropy (maxent) classifier has been a popular text classifier, by parameterizing the model
to achieve maximum categorical entropy, with the constraint that the resulting probability
on the training data with the model being equal to the real distribution.

The maxent classifier in `shorttext` is impleneted by `keras`. The optimization algorithm is
defaulted to be the Adam optimizer, although other gradient-based or momentum-based optimizers
can be used. The traditional methods such as generative iterative scaling (GIS) or
L-BFGS cannot be used here.

To use the maxent classifier, import the package:

>>> import shorttext
>>> from shorttext.classifiers import MaxEntClassifier

Loading NIH reports as an example:

>>> classdict = shorttext.data.nihreports()

The classifier can be instantiated by:

>>> classifier = MaxEntClassifier()

Train the classifier:

>>> classifier.train(classdict, nb_epochs=1000)

After training, it can be used for classification, such as

>>> classifier.score('cancer immunology')   # NCI tops the score
>>> classifier.score('children health')     # NIAID tops the score
>>> classifier.score('Alzheimer disease and aging')    # NIAID tops the score

To save the model,

>>> classifier.save_compact_model('/path/to/filename.bin')

To load the model to be a classifier, enter:

>>> classifier2 = shorttext.classifiers.load_maxent_classifier('/path/to/filename.bin')

Reference
---------

Adam L. Berger, Stephen A. Della Pietra, Vincent J. Della Pietra, "A Maximum Entropy Approach to Natural Language Processing," *Computational Linguistics* 22(1): 39-72 (1996). [`ACM
<http://dl.acm.org/citation.cfm?id=234289>`_]

Daniel E. Russ, Kwan-Yuet Ho, Joanne S. Colt, Karla R. Armenti, Dalsu Baris, Wong-Ho Chow, Faith Davis, Alison Johnson, Mark P. Purdue, Margaret R. Karagas, Kendra Schwartz, Molly Schwenn, Debra T. Silverman, Patricia A. Stewart, Calvin A. Johnson, Melissa C. Friesen, “Computer-based coding of free-text job descriptions to efficiently and reliably incorporate occupational risk factors into large-scale epidemiological studies”, *Occup. Environ. Med.* 73, 417-424 (2016). [`BMJ
<http://oem.bmj.com/content/73/6/417.long>`_]

Daniel Russ, Kwan-yuet Ho, Melissa Friesen, "It Takes a Village To Solve A Problem in Data Science," Data Science Maryland, presentation at Applied Physics Laboratory (APL), Johns Hopkins University, on June 19, 2017. (2017) [`Slideshare
<https://www.slideshare.net/DataScienceMD/it-takes-a-village-to-solve-a-problem-in-data-science>`_]

Home: :doc:`index`