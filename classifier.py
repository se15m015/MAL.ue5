from sklearn import neighbors
from sklearn import tree
from sklearn import ensemble
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from split import holdout
from split import fold5
from mlPrint import printKNNHoldout
from mlPrint import printKNNFold
from mlPrint import printRandomTreeFold
from mlPrint import printHoldout
from mlPrint import printFold
from mlPrint import printHeader
from sklearn.neural_network import MLPClassifier
from util import getRandomState

def kNN(data, target):
    # parameters for k-NN
    n_neighbors = [1, 5, 13, 55]
    weights = ["uniform", "distance"]

    for k in n_neighbors:
        for weight in weights:
            # train the k-NN
            classifier = neighbors.KNeighborsClassifier(k, weights=weight)

            #Fold 5
            accSum, precisionSum, recallSum, time_trainSum, time_testSum = fold5(data, target, classifier)

            printKNNFold(k, weight, accSum, precisionSum, recallSum, time_trainSum, time_testSum)
    return

def kNNParams(data, target, k=5, weight="uniform"):
    # parameters for k-NN

    # train the k-NN
    classifier = neighbors.KNeighborsClassifier(k, weights=weight)

    #Fold 5
    accSum, precisionSum, recallSum, time_trainSum, time_testSum = fold5(data, target, classifier)

    printKNNFold(k, weight, accSum, precisionSum, recallSum, time_trainSum, time_testSum)
    return

def decisionTreeGini(data, target):
    decisionTree(data, target, "gini")
    return

def decisionTreeEntropy(data, target):
    decisionTree(data, target, "entropy")
    return

def decisionTreePrePruningTest(data, target):
    printHeader("minWeightFractionLeaf=0.3, minSamplesLeaf=2, maxDepth=5")
    decisionTree(data, target, criterion="gini", minWeightFractionLeaf=0)
    return

def decisionTreePrePruning1(data, target):
    printHeader("criterion=\"gini\", minWeightFractionLeaf=0.1, maxDepth=5")
    decisionTree(data, target, criterion="gini", minWeightFractionLeaf=0.1, maxDepth=5)
    return

def decisionTreePrePruning2(data, target):
    printHeader("criterion=\"gini\", minSamplesLeaf=20, maxDepth=5")
    decisionTree(data, target, criterion="gini", minSamplesLeaf=20, maxDepth=5)
    return

def decisionTree(data, target, criterion="gini", minWeightFractionLeaf=0, minSamplesLeaf=1, maxDepth=None):
    #minWeightFractionLeaf = min Anzahl in % der gesamt Samples pro Blattknoten
    #minSamplesLeaf = min Anzahl als absolut Wert pro Blattknoten
    classifier = tree.DecisionTreeClassifier(criterion=criterion, min_weight_fraction_leaf=minWeightFractionLeaf,
                                             min_samples_leaf=minSamplesLeaf, max_depth=maxDepth)

    #Fold 5
    acc, precision, recall, time_train, time_test = fold5(data, target, classifier)

    printHeader("Decision Tree", "Fold 5", criterion, str(minWeightFractionLeaf), str(minSamplesLeaf), str(maxDepth))
    printFold(acc, precision, recall, time_train, time_test)

    # tree.export_graphviz(classifier, out_file="treePNG/tree-cir_%s-mWFL_%s-mSL_%s-mD_%s.dot" % (criterion, minWeightFractionLeaf, minSamplesLeaf, maxDepth))
    return

def randomForrest(data, target):
    n_estimators = [20, 50]
    max_features = ["sqrt", "log2", 3]

    for e in n_estimators:
        for f in max_features:
            classifier = ensemble.RandomForestClassifier(n_estimators=e, max_features=f)

            # Fold 5
            accSum, precisionSum, recallSum, time_trainSum, time_testSum = fold5(data, target, classifier)

            printRandomTreeFold(e, f, accSum, precisionSum, recallSum, time_trainSum, time_testSum)
    return

def randomForrestParams(data, target,e = 100,f = "sqrt"):

    classifier = ensemble.RandomForestClassifier(n_estimators=e, max_features=f)

    # Fold 5
    accSum, precisionSum, recallSum, time_trainSum, time_testSum = fold5(data, target, classifier)

    printRandomTreeFold(e, f, accSum, precisionSum, recallSum, time_trainSum, time_testSum)
    return

def SVC(data, target):
    classifier = svm.SVC()

    # Fold 5
    acc, precision, recall, time_train, time_test = fold5(data, target, classifier)

    printHeader("SVC", "Fold 5")
    printFold(acc, precision, recall, time_train, time_test)
    return

def LinarSVC(data, target, penaltyC=1):
    classifier = svm.LinearSVC(C=penaltyC)

    # Fold 5
    acc, precision, recall, time_train, time_test = fold5(data, target, classifier)

    printHeader("LinarSVC", "Fold 5", "penaltyC: "+str(penaltyC))
    printFold(acc, precision, recall, time_train, time_test)

    return

def naiveBayes(data, target):
    classifier = GaussianNB()

    # Fold 5
    acc, precision, recall, time_train, time_test = fold5(data, target, classifier)

    printHeader("Naive Bayes", "Fold 5")
    printFold(acc, precision, recall, time_train, time_test)
    return

def perceptron(data, target):
    classifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = getRandomState())

    # Fold 5
    acc, precision, recall, time_train, time_test = fold5(data, target, classifier)

    printHeader("Perceptron", "Fold 5")
    printFold(acc, precision, recall, time_train, time_test)
    return
