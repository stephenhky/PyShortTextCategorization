# This code runs cross validation to the models developed, not intended for
# others' use.
# It uses the example data as the example.

# argument parsing
import argparse

from classifiers import allowed_algos

argparser = argparse.ArgumentParser(description='Run cross validation.')
argparser.add_argument('algo', help='Algorithm to run. Options: '+', '.join(list(allowed_algos)))
argparser.add_argument('word2vec_path', help='Path of the binary Word2Vec model.')
args = argparser.parse_args()

# import other libraries
from gensim.models import Word2Vec
import numpy as np
import pandas as pd

import classifiers.embed.sumvec.SumWord2VecClassification as sumwv
import classifiers.embed.autoencode.AutoencoderEmbedVecClassification as auto
import classifiers.embed.nnlib.CNNEmbedVecClassification as cnn
import data.data_retrieval as ret
from utils import AlgorithmNotExistException

# loading Word2Vec model
print "Loading embedding model..."
wvmodel = Word2Vec.load_word2vec_format(args.word2vec_path, binary=True)

# data partition
partnum = 5
repetitions = 6
length = 3
master_classdict = ret.retrieve_data_as_dict('data/shorttext_exampledata.csv')
partitioned_classdicts = []
for grp in range(repetitions):
    shuffled = {}
    for classlabel in master_classdict:
        shuffled[classlabel] = list(master_classdict[classlabel])
        np.random.shuffle(shuffled[classlabel])
    for part in range(partnum):
        testdict = {}
        traindict = {}
        for classlabel in shuffled:
            testdict[classlabel] = shuffled[classlabel][part*length:(part+1)*length]
            traindict[classlabel] = np.append(shuffled[classlabel][:part*length], shuffled[classlabel][(part+1)*length:])
            partitioned_classdicts.append({'train': traindict, 'test': testdict})

print 'Number of tests = ', len(partitioned_classdicts)

accuracies = []
attemptidx = 0
attemptdata = []
parsed_shorttexts = []
system_predictions = []
expert_predictions = []
scores = []
for classdicts in partitioned_classdicts:
    # train model
    if args.algo=='sumword2vec':
        classifier = sumwv.SumEmbeddedVecClassifier(wvmodel, classdicts['train'])
    elif args.algo=='autoencoder':
        classifier = auto.AutoEncoderWord2VecClassifier(wvmodel, classdicts['train'])
    elif args.algo=='cnn':
        classifier = cnn.CNNEmbeddedVecClassifier(wvmodel, classdicts['train'], 2)
    else:
        raise AlgorithmNotExistException(args.algo)
    classifier.train()

    numdata = 0
    numcorrects = 0
    for classlabel in classdicts['test']:
        for shorttext in classdicts['test'][classlabel]:
            predictions = classifier.score(shorttext)
            predicted_label, predicted_score = max(predictions.items(), key=lambda s: s[1])
            numdata += 1
            numcorrects += 1 if predicted_label==classlabel else 0

            attemptdata.append(attemptidx)
            parsed_shorttexts.append(shorttext)
            system_predictions.append(predicted_label)
            expert_predictions.append(classlabel)
            scores.append(predicted_score)

    print 'numdata = ', numdata
    print 'numcorrects = ', numcorrects
    print 'accuracy = ', float(numcorrects)/numdata
    accuracies.append(float(numcorrects)/numdata)

    attemptidx += 1

accdf = pd.DataFrame({'attempt': range(len(accuracies)), 'accuracy': accuracies})
accdf.to_csv('crossval/'+args.algo+'_accuracies.csv', index=False)
datadf = pd.DataFrame({'attempt': attemptdata,
                       'shorttext': parsed_shorttexts,
                       'prediction': system_predictions,
                       'expert_prediction': expert_predictions,
                       'score': scores})
datadf.to_csv('crossval/'+args.algo+'_testdata.csv', index=False)