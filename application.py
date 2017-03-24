# Import statements
import numpy as np
import pandas as pd

from sklearn import datasets, svm, ensemble
from sklearn.utils import shuffle

from load import loadCongressTrain, loadCongressTest, loadAmazonTrain, loadAmazonTest, loadKddTrain
from util import getRandomState
from classifier import kNN, decisionTreeGini, decisionTreeEntropy, decisionTreePrePruning1, decisionTreePrePruning2, randomForrest, randomForrestParams, SVC, LinarSVC
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

def writeToFile(classifier, datasetTrain, datasetTest, filename, headers):

    classifier.fit(datasetTrain.data, datasetTrain.target)
    result = classifier.predict(datasetTest.data)
    resultTransformed = datasetTrain.le.inverse_transform(result)

    text_file = open(filename, "w")
    text_file.write(headers)
    for i in range(0, len(resultTransformed)):
        text_file.write("%s,%s" % (str(datasetTest.ids[i]), str(resultTransformed[i])))
        if (i < len(resultTransformed) - 1):
            text_file.write("\n")

    text_file.close()
    return

# datasetTrain = loadCongressTrain()
# datasetTest = loadCongressTest()
# classifier = svm.LinearSVC(C=34)
# writeToFile(classifier, datasetTrain, datasetTest, "output/congress_linearSVC_C34.csv", "ID,class\n")

# datasetTrain = loadAmazonTrain()
# datasetTest = loadAmazonTest()
# classifier = ensemble.RandomForestClassifier(n_estimators=5000, max_features="sqrt")
# writeToFile(classifier, datasetTrain, datasetTest, "output/amazon_randomForest.csv", "ID,class\n")


# dataset = loadAmazonTrain()
# RunAmazonClassifiers(dataset.data, dataset.target)
## LinarSVC(dataset.data, dataset.target)
#ä randomForrest(dataset.data, dataset.target)

# dataset = loadKddTrain()






