# Import statements
import numpy as np
import pandas as pd

from sklearn import datasets, svm, ensemble, tree
from sklearn.utils import shuffle

from load import loadCongress, loadAmazon, loadACI, loadKdd
from util import getRandomState
from classifier import kNN, decisionTreeGini, decisionTreeEntropy, decisionTreePrePruning1, decisionTreePrePruning2, randomForrest, randomForrestParams, SVC, LinarSVC, \
    kNNParams, decisionTree
from classifier import naiveBayes
from classifier import perceptron
from mlPrint import printHeaderDataset

import csv

def runAllClassifier(data, target):
    kNN(data, target)
    decisionTreeGini(data, target)
    decisionTreeEntropy(data, target)
    decisionTreePrePruning1(data, target)
    decisionTreePrePruning2(data, target)
    randomForrest(data, target)
    SVC(data, target)
    LinarSVC(data, target)
    naiveBayes(data, target)
    perceptron(data, target)
    return

def RunAmazonClassifiers(data, target):

    #for e in np.linspace(1000,10000, dtype=np.dtype(np.int16)):
    #for e in [10000,30000]:
    for e in [6000]:
        randomForrestParams(data, target,e=e)
    return

def RunKDDClassifiers(data, target):

    for k in range(20,150):
        kNNParams(data, target,  k, weight="uniform")

    for minWeightFractionLeaf in np.linspace(0,0.5):
        for minSamplesLeaf in range(1, 50):
            decisionTree(data, target, criterion="gini", minWeightFractionLeaf=minWeightFractionLeaf, minSamplesLeaf=minSamplesLeaf, maxDepth=None)
    return

def RunACIClassifiers(data, target):
    #
    # for k in range(20,150):
    #     kNNParams(data, target,  k, weight="uniform")

    for minWeightFractionLeaf in np.linspace(0,0.5):
        for minSamplesLeaf in range(1, 50):
            decisionTree(data, target, criterion="gini", minWeightFractionLeaf=minWeightFractionLeaf, minSamplesLeaf=minSamplesLeaf, maxDepth=None)
    return

def writeToFile(classifier, dataset, filename, headers):

    classifier.fit(dataset.data_train, dataset.target)
    result = classifier.predict(dataset.data_test)
    resultTransformed = dataset.le.inverse_transform(result)

    text_file = open(filename, "w")
    text_file.write(headers)
    for i in range(0, len(resultTransformed)):
        text_file.write("%s,%s" % (str(dataset.ids_test[i]), str(resultTransformed[i])))
        if i < len(resultTransformed) - 1:
            text_file.write("\n")

    text_file.close()
    return

# dataset = loadCongress()
# classifier = svm.LinearSVC(C=34)
# writeToFile(classifier, dataset, "output/congress_linearSVC_C34.csv", "ID,class\n")
#
# dataset = loadAmazon()
# classifier = ensemble.RandomForestClassifier(n_estimators=5000, max_features="sqrt")
# writeToFile(classifier, dataset, "output/amazon_randomForest.csv", "ID,class\n")


# dataset = loadAmazon()
# RunAmazonClassifiers(dataset.data_train, dataset.target)
## LinarSVC(dataset.data_train, dataset.target)
#ä randomForrest(dataset.data_train, dataset.target)

# dataset = loadKdd()
# RunKDDClassifiers(dataset.data_train, dataset.target)

# runAllClassifier(dataset.data_train, dataset.target)



dataset = loadACI()
classifier = tree.DecisionTreeClassifier(criterion="gini", min_weight_fraction_leaf=0.1, min_samples_leaf=20, max_depth=None)
writeToFile(classifier, dataset, "output/aci_decisionTree_gini_0.1_20_None.csv", "ID,moreThan50K\n")
# RunACIClassifiers(dataset.data_train, dataset.target)
# runAllClassifier(dataset.data_train, dataset.target)




