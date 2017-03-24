# Import statements
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from util import getRandomState
from classifier import kNN, decisionTreeGini, decisionTreeEntropy, decisionTreePrePruning1, decisionTreePrePruning2, randomForrest, SVC, LinarSVC
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

def loadIris():
    # load the IRIS dataset
    printHeaderDataset("IRIS")
    dataSet = datasets.load_iris()
    # Shuffle our input data
    data, target = shuffle(dataSet.data, dataSet.target, random_state=getRandomState())
    return [data, target]

def loadDigits():
    # load the DIGITS dataset
    printHeaderDataset("DIGITS")
    dataSet = datasets.load_digits()
    # Shuffle our input data
    data, target = shuffle(dataSet.data, dataSet.target, random_state=getRandomState())
    return [data, target]

def loadBreastCancer():
    # load the BEAST-CANCER dataset
    printHeaderDataset("BREAST-CANCER")
    dataSet = datasets.load_breast_cancer()

    # Shuffle our input data
    data, target = shuffle(dataSet.data, dataSet.target, random_state=getRandomState())
    return [data, target]



[data, target] = loadIris()
runAllClassifier(data, target)

[data, target] = loadDigits()
runAllClassifier(data, target)

[data, target] = loadBreastCancer()
runAllClassifier(data, target)




