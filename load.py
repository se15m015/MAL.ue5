import csv
import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch

def loadCongress():
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

    ids_train = df._get_values[:, 0]
    data_train = df._get_values[:, 2:]

    for i in range(0, len(feature_names)):
        leFeatures = preprocessing.LabelEncoder()
        leFeatures.fit(data_train[:,i])
        data_train[:,i] = leFeatures.transform(data_train[:,i])

    filename = 'data/congress/CongressionalVotingID.shuf.test.csv'

    df = pd.read_csv(filename, header=0)

    # original_headers = list(df.columns.values)
    # feature_names = original_headers[1:]

    ids_test = df._get_values[:, 0]
    data_test = df._get_values[:, 1:]

    leFeatures = preprocessing.LabelEncoder()
    leFeatures.fit(data_test[:,2])
    for i in range(0, len(feature_names)):
        data_test[:,i] = leFeatures.transform(data_test[:,i])

    return Bunch(data_train=data_train,
                 data_test=data_test,
                 target=target,
                 target_names=target_names,
                 feature_names=feature_names,
                 le=leTarget,
                 ids_train=ids_train,
                 ids_test=ids_test)

def loadAmazon():
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

    ids_train = df._get_values[:, 0]
    data_train = df._get_values[:, 1:-1]

    filename = 'data/amazon/phpr1uf8OID.700.test.csv'

    df = pd.read_csv(filename, header=0)

    # original_headers = list(df.columns.values)
    # feature_names = original_headers[1:]

    ids_test = df._get_values[:, 0]
    data_test = df._get_values[:, 1:]

    return Bunch(data_train=data_train,
                 data_test=data_test,
                 target=target,
                 target_names=target_names,
                 feature_names=feature_names,
                 le=le,
                 ids_train=ids_train,
                 ids_test=ids_test)

def loadKdd():
    datasetTrain = loadKddFile('data/kdd/cup98LRN.5k.train_binary.csv', 'data/kdd/cup98LRN.10k.test_binary.csv')
    # datasetTest = loadKddFile('data/kdd/cup98LRN.10k.test_binary.csv')

    return datasetTrain
    # return Bunch(data_train=datasetTrain.data_train,
    #              data_test=datasetTest.data_train,
    #              target=datasetTrain.target,
    #              target_names=datasetTrain.target_names,
    #              feature_names=datasetTrain.feature_names,
    #              le=datasetTrain.le,
    #              ids_train=datasetTrain.ids_train,
    #              ids_test=datasetTest.ids_train)

def loadKddFile(filenameTrain, filenameTest):

    df_Train = pd.read_csv(filenameTrain, header=0,  dtype={'NOEXCH': str})
    df_Test = pd.read_csv(filenameTest, header=0,  dtype={'NOEXCH': str})

    numbers = df_Train.select_dtypes(exclude=['object'])
    # remove cols numbers
    for index in numbers:
        nan = np.count_nonzero(np.isnan(df_Train[index].get_values()))
        if nan > 200:
            df_Train.drop(index, axis=1, inplace=True)
            df_Test.drop(index, axis=1, inplace=True)


    numbers = df_Test.select_dtypes(exclude=['object'])
    for index in numbers:
        nan = np.count_nonzero(np.isnan(df_Test[index].get_values()))
        if nan > 200:
            df_Train.drop(index, axis=1, inplace=True)
            df_Test.drop(index, axis=1, inplace=True)


    objects = df_Train.select_dtypes(include=['object'])
    # remove cols objects
    for index in objects:
        missingCount = df_Train[index].get_values().tolist().count(' ')
        if missingCount > 200:
            if nan > 200:
                df_Train.drop(index, axis=1, inplace=True)
                df_Test.drop(index, axis=1, inplace=True)


    objects = df_Test.select_dtypes(include=['object'])
    # remove cols objects
    for index in objects:
        missingCount = df_Test[index].get_values().tolist().count(' ')
        if missingCount > 200:
            if nan > 200:
                df_Train.drop(index, axis=1, inplace=True)
                df_Test.drop(index, axis=1, inplace=True)



    numbers = df_Train.select_dtypes(exclude=['object'])
    objects = df_Train.select_dtypes(include=['object'])

    # remove rows for numbers
    for index in numbers:
        nan = np.count_nonzero(np.isnan(df_Train[index].get_values()))
        if nan > 1:
            df_Train = df_Train.drop(df_Train[np.isnan(df_Train[index])].index)

    # remove rows for objects
    for index in objects:
        missingCount = df_Train[index].get_values().tolist().count(' ')
        if missingCount > 1:
            df_Train = df_Train.drop(df_Train[df_Train[index] == ' '].index)


    original_headers = list(df_Train.columns.values)
    feature_names = list(df_Train.columns.values)
    feature_names.remove("TARGET_B")
    feature_names.remove("CONTROLN")
    target_names = df_Train["TARGET_B"]
    ids_train = df_Train["CONTROLN"]
    ids_test = df_Test["CONTROLN"]


    from sklearn import preprocessing
    leFeatures = preprocessing.LabelEncoder()

    df_Train.drop("TARGET_B", axis=1, inplace=True)
    df_Train.drop("TARGET_D", axis=1, inplace=True)
    df_Train.drop("CONTROLN", axis=1, inplace=True)
    # df.drop("NOEXCH", axis=1, inplace=True)

    df_Test.drop("TARGET_B", axis=1, inplace=True)
    df_Test.drop("CONTROLN", axis=1, inplace=True)

    for index in df_Train:
        leFeatures = preprocessing.LabelEncoder()
        leFeatures.fit(df_Train[index])
        df_Train[index] = leFeatures.transform(df_Train[index])

    for index in df_Test:
        leFeatures = preprocessing.LabelEncoder()
        leFeatures.fit(df_Test[index])
        df_Test[index] = leFeatures.transform(df_Test[index])

    data_train = df_Train.get_values()
    data_test = df_Test.get_values()

    le = preprocessing.LabelEncoder()
    le.fit(target_names)
    target = le.transform(target_names);


    return Bunch(data_train=data_train,
                 data_test=data_test,
                 target=target,
                 target_names=target_names,
                 feature_names=feature_names,
                 le=le,
                 ids_train=ids_train,
                 ids_test=ids_test)

def loadACI():
    # filename = 'data/aci/adult.csv'
    filename = 'data/aci/adult_ID.train.csv'

    df = pd.read_csv(filename, header=0)


    for index in df:
        missingCount = df[index].get_values().tolist().count('?')
        if missingCount > 1:
            df = df.drop(df[df[index] == '?'].index)


    feature_names = list(df.columns.values)
    target_names = df["moreThan50K"]
    feature_names.remove("moreThan50K")
    feature_names.remove("ID")

    ids_train = df["ID"]

    df.drop("moreThan50K", axis=1, inplace=True)
    df.drop("ID", axis=1, inplace=True)

    from sklearn import preprocessing
    leTarget = preprocessing.LabelEncoder()
    leTarget.fit(target_names)  # this basically finds all unique class names, and assigns them to the numbers
    # print("Found the following classes: " + str(list(le.classes_)))

    # now we transform our labels to integers
    target = leTarget.transform(target_names);
    # print("Transformed labels (first elements: " + str(target[0:1000]))


    data_train = df.get_values()
    for i in range(0, len(feature_names)):
        leFeatures = preprocessing.LabelEncoder()
        leFeatures.fit(data_train[:,i])
        data_train[:,i] = leFeatures.transform(data_train[:,i])


    filename = 'data/aci/adult_ID.test.csv'

    df = pd.read_csv(filename, header=0)

    # for index in df:
    #     missingCount = df[index].get_values().tolist().count('?')
    #     if missingCount > 1:
    #         df = df.drop(df[df[index] == '?'].index)

    ids_test = df["ID"]
    df.drop("ID", axis=1, inplace=True)

    data_test = df.get_values()
    for i in range(0, len(feature_names)):
        leFeatures = preprocessing.LabelEncoder()
        leFeatures.fit(data_test[:,i])
        data_test[:,i] = leFeatures.transform(data_test[:,i])

    return Bunch(data_train=data_train,
                 data_test=data_test,
                 target=target,
                 target_names=target_names,
                 feature_names=feature_names,
                 le=leTarget,
                 ids_train=ids_train,
                 ids_test=ids_test)

def loadKddFake():
    filename = 'data/kdd/cup98LRN.10k.test_binary.csv'
    df = pd.read_csv(filename, header=0)

    text_file = open("output/kdd_fake.csv", "w")
    text_file.write("CONTROLN,TARGET_B\n")

    for row in df.iterrows():
        if row[1]["TARGET_B"] == 0:
            text_file.write("%s,%s" % (str(row[1]["CONTROLN"]), str("False")))
        else:
            text_file.write("%s,%s" % (str(row[1]["CONTROLN"]), str("True")))
        text_file.write("\n")

    # row = next(df.iterrows())[1]
    # for i in range(0, len(resultTransformed)):
    #     text_file.write("%s,%s" % (str(dataset.ids_test[i]), str(resultTransformed[i])))
    #     if i < len(resultTransformed) - 1:
    #         text_file.write("\n")

    text_file.close()
    return

# loadKddFake()
# loadACI()
# loadKdd()
# loadCongressTrain()
# loadCongressTest()
