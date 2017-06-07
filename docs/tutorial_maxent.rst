Maximum Entropy (MaxEnt) Classifier
===================================

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

>>> classifier.score('/path/to/filename.bin')

To load the model to be a classifier, enter:

>>> classifier2 = shorttext.classifiers.load_maxent_classifier('/path/to/filename.bin')
