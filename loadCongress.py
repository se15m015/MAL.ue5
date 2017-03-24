import csv
import numpy as np
import pandas as pd

def loadCongress():
    congressFile = 'data/congress/CongressionalVotingID.shuf.train.csv'

    df = pd.read_csv(congressFile, header=0)

    original_headers = list(df.columns.values)
    feature_names = original_headers[2:]

    # df = df._get_values

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    col1 = df.columns[1]
    col2 = df.columns[2]
    encoded_series = df[df.columns[1]].apply(le.fit_transform)

    le.fit(target_names)  # this basically finds all unique class names, and assigns them to the numbers
    print("Found the following classes: " + str(list(le.classes_)))

    # now we transform our labels to integers
    target = le.transform(target_names);
    print("Transformed labels (first elements: " + str(target[0:150]))

    numeric_headers = list(df.columns.values)

    # create a numpy array with the numeric values for input into scikit-learn
    numpy_array = df.as_matrix()

    target_names = numpy_array[1]









    with open(congressFile) as csv_file:
        data_file = csv.reader(csv_file)
        first_line = next(data_file)
        n_samples = 218
        n_features = 16
        target_names = ['democrat','republican']
        feature_names = first_line[2:]
        data = np.empty((n_samples, n_features))

        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        le.fit(target_names)  # this basically finds all unique class names, and assigns them to the numbers
        print("Found the following classes: " + str(list(le.classes_)))

        # now we transform our labels to integers
        target = le.transform(target_names);
        print("Transformed labels (first elements: " + str(target[0:150]))


        target = np.empty((n_samples,), dtype=np.int)

        for count, value in enumerate(data_file):
            data[count] = np.asarray(value[:-1], dtype=np.float64)
            target[count] = np.asarray(value[-1], dtype=np.int)

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 feature_names=feature_names,
                 le=le)

loadCongress()