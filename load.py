import csv
import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch

def loadCongressTrain():
    filename = 'data/congress/CongressionalVotingID.shuf.train.csv'

    df = pd.read_csv(filename, header=0)

    original_headers = list(df.columns.values)
    target_names = df.get_values()[:,1]
    feature_names = original_headers[2:]

    from sklearn import preprocessing
    leTarget = preprocessing.LabelEncoder()
    leTarget.fit(target_names)  # this basically finds all unique class names, and assigns them to the numbers
    # print("Found the following classes: " + str(list(le.classes_)))

    # now we transform our labels to integers
    target = leTarget.transform(target_names);
    # print("Transformed labels (first elements: " + str(target[0:1000]))

    ids = df._get_values[:, 0]
    data = df._get_values[:, 2:]

    for i in range(0, len(feature_names)):
        leFeatures = preprocessing.LabelEncoder()
        leFeatures.fit(data[:,i])
        data[:,i] = leFeatures.transform(data[:,i])

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 feature_names=feature_names,
                 le=leTarget,
                 ids=ids)

def loadCongressTest():
    filename = 'data/congress/CongressionalVotingID.shuf.test.csv'

    df = pd.read_csv(filename, header=0)

    original_headers = list(df.columns.values)
    feature_names = original_headers[1:]

    ids = df._get_values[:, 0]
    data = df._get_values[:, 1:]

    from sklearn import preprocessing
    leFeatures = preprocessing.LabelEncoder()
    leFeatures.fit(data[:,2])
    for i in range(0, len(feature_names)):
        data[:,i] = leFeatures.transform(data[:,i])

    return Bunch(data=data,
                 feature_names=feature_names,
                 ids=ids)

def loadAmazonTrain():
    filename = 'data/amazon/phpr1uf8OID.800.train.csv'

    df = pd.read_csv(filename, header=0)

    original_headers = list(df.columns.values)
    target_names = df.get_values()[:, -1]
    feature_names = original_headers[1:-1]

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(target_names)  # this basically finds all unique class names, and assigns them to the numbers
    # print("Found the following classes: " + str(list(le.classes_)))

    # now we transform our labels to integers
    target = le.transform(target_names);
    # print("Transformed labels (first elements: " + str(target[0:1000]))

    ids = df._get_values[:, 0]
    data = df._get_values[:, 1:-1]

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 feature_names=feature_names,
                 le=le,
                 ids=ids)


def loadAmazonTest():
    filename = 'data/amazon/phpr1uf8OID.700.test.csv'

    df = pd.read_csv(filename, header=0)

    original_headers = list(df.columns.values)
    feature_names = original_headers[1:]

    ids = df._get_values[:, 0]
    data = df._get_values[:, 1:]

    return Bunch(data=data,
                 feature_names=feature_names,
                 ids=ids)

def loadKddTrain():
    filename = 'data/kdd/cup98LRN.5k.train_binary.csv'

    df = pd.read_csv(filename, header=0)


    original_headers = list(df.columns.values)
    feature_names = list(df.columns.values)
    feature_names.remove("TARGET_B")
    feature_names.remove("CONTROLN")
    target_names = df["TARGET_B"]
    ids = df["CONTROLN"]

    df.drop("TARGET_B", axis=1, inplace=True)
    df.drop("CONTROLN", axis=1, inplace=True)

    data = df.get_values()
    for col in df:
        missingValuesCount = df[col].get_values().tolist().count(' ')
        if missingValuesCount > 50:
            print(col)
            df.drop(col, axis=1, inplace=True)
        # missingValuesCount = data[:, i].tolist().count(' ')
        # # print(original_headers[i])
        # if missingValuesCount > 50:
        #     df.drop(original_headers[i], axis=1, inplace=True)


    # for i in range(0, len(original_headers)):
    #     missingValuesCount = data[:, i].tolist().count(' ')
    #     # print(original_headers[i])
    #     if missingValuesCount > 50:
    #         df.drop(original_headers[i], axis=1, inplace=True)

    # data = df.get_values()
    # for i in range(0, len(feature_names)):
    #     missingValuesCount = data[:, i].tolist().count(' ')
    #     print(original_headers[i])
    #     print(missingValuesCount)



    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(target_names)  # this basically finds all unique class names, and assigns them to the numbers
    # print("Found the following classes: " + str(list(le.classes_)))

    # now we transform our labels to integers
    target = le.transform(target_names);
    # print("Transformed labels (first elements: " + str(target[0:1000]))

    ids = df._get_values[:, 0]
    data = df._get_values[:, 1:-1]

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 feature_names=feature_names,
                 le=le,
                 ids=ids)


# loadKddTrain()
# loadCongressTrain()
loadCongressTest()
