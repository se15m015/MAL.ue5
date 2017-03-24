# Import statements
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
from util import getRandomState
from classifier import kNN, decisionTreeGini, decisionTreeEntropy, decisionTreePrePruning1, decisionTreePrePruning2, randomForrest, SVC, LinarSVC
from classifier import naiveBayes
from classifier import perceptron
from mlPrint import printHeaderDataset
import pickle

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
#
# from loadImages import loadImages
# [data, dataOpenCV_1D, dataOpenCV_2D, dataOpenCV_3D, target] = loadImages()
#
# printHeaderDataset("IMAGES")
#
# printHeaderDataset("data")
# runAllClassifier(np.asarray(data), target)
# printHeaderDataset("dataOpenCV_1D")
# runAllClassifier(np.asarray(dataOpenCV_1D), target)
# printHeaderDataset("dataOpenCV_2D")
# runAllClassifier(np.asarray(dataOpenCV_2D), target)
# printHeaderDataset("dataOpenCV_3D")
# runAllClassifier(np.asarray(dataOpenCV_3D), target)
#
#
#
#
# from loadAudio import loadAudio
# [data_bpm, data_bpm_statistics, data_chroma, data_mfcc, target] = loadAudio()
#
# printHeaderDataset("AUDIO")

#Achtung - Audio data_bpm kann nicht mit dem RandomForrest Classifier ausgeführt werden, da data_bpm nur ein Feature hat.
# RandomForrest muss bei diesem Datansatz auskommentiert werden.
# printHeaderDataset("data_bpm")
# runAllClassifier(np.asarray(data_bpm), target)
# printHeaderDataset("data_bpm_statistics")
# runAllClassifier(np.asarray(data_bpm_statistics), target)
# printHeaderDataset("data_chroma")
# runAllClassifier(np.asarray(data_chroma), target)
# printHeaderDataset("data_mfcc")
# runAllClassifier(np.asarray(data_mfcc), target)


