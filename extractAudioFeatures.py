import pickle

def extractAudioFeatures():
    # We need to construct our data set; unfortunately, we don't simply have a "loadGTZanDataSet()" function in SK-learn...
    # So we need to
    ## Download our data set & extract it (one-time effort)
    ## Run an audio feature extraction
    ## Create the create the ground truth (label assignment, target, ...)


    # path to our audio folder
    # For the first run, download the images from http://kronos.ifs.tuwien.ac.at/GTZANmp3_22khz.zip, and unzip them to your folder
    imagePath="/temp/MAL/data/GTZANmp3_22khz/"


    # Find all songs in that folder; there are like 1.000 different ways to do this in Python, we chose this one :-)
    import glob, os
    os.chdir(imagePath)
    fileNames = glob.glob("*/*.mp3")
    numberOfFiles=len(fileNames)
    targetLabels=[]

    print("Found " + str(numberOfFiles) + " files\n")

    # The first step - create the ground truth (label assignment, target, ...)
    # For that, iterate over the files, and obtain the class label for each file
    # Basically, the class name is in the full path name, so we simply use that
    for fileName in fileNames:
        pathSepIndex = fileName.index("/")
        targetLabels.append(fileName[:pathSepIndex])

    # sk-learn can only handle labels in numeric format - we have them as strings though...
    # Thus we use the LabelEncoder, which does a mapping to Integer numbers
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(targetLabels) # this basically finds all unique class names, and assigns them to the numbers
    print("Found the following classes: " + str(list(le.classes_)))

    # now we transform our labels to integers
    target = le.transform(targetLabels);
    print("Transformed labels (first elements: " + str(target[0:150]))

    # If we want to find again the label for an integer value, we can do something like this:
    # print list(le.inverse_transform([0, 18, 1]))

    print("... done label encoding")

    import matplotlib.pyplot as plt
    from librosa import display
    import librosa
    import numpy as np


    # Now we do the actual feature extraction

    import datetime
    from collections import deque
    #import progressbar

    import numpy as np
    import scipy.stats.stats as st


    # This is a helper function that computes the differences between adjacent array values
    def differences(seq):
        iterable = iter(seq)
        prev = next(iterable)
        for element in iterable:
            yield element - prev
            prev = element

    # This is a helper function that computes various statistical moments over a series of values, including mean, median, var, min, max, skewness and kurtosis (a total of 7 values)
    def statistics(numericList):
        return [np.mean(numericList), np.median(numericList), np.var(numericList), np.float64(st.skew(numericList)), np.float64(st.kurtosis(numericList)), np.min(numericList), np.max(numericList)]



    print("Extracting features using librosa" + " (" + str(datetime.datetime.now()) + ")")

    # compute some features based on BPMs, MFCCs, Chroma
    data_bpm=[]
    data_bpm_statistics=[]
    data_mfcc=[]
    data_chroma=[]

    # This takes a bit, so let's show it with a progress bar
    #with progressbar.ProgressBar(max_value=len(fileNames)) as bar:
    for indexSample, fileName in enumerate(fileNames):
        # Load the audio as a waveform `y`, store the sampling rate as `sr`
        y, sr = librosa.load(fileName)

        # run the default beat tracker
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        # from this, we simply use the tempo as BPM feature
        data_bpm.append([tempo])

        # Then we compute a few statistics on the beat timings
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        # from the timings, compute the time differences between the beats
        beat_intervals = np.array(deque(differences(beat_times)))

        # And from this, take some statistics
        # There might be a few files where the beat timings are not determined properly; we ignore them, resp. give them 0 values
        if len(beat_intervals) < 1:
            print("Errors with beat interval in file " + fileName + ", index " + str(indexSample) + ", using 0 values instead")
            data_bpm_statistics.append([tempo, 0, 0, 0, 0, 0, 0, 0])
        else:
            bpm_statisticsVector=[]
            bpm_statisticsVector.append(tempo) # we also include the raw value of tempo
            for stat in statistics(beat_intervals):  # in case the timings are ok, we actually compute the statistics
                bpm_statisticsVector.append(stat) # and append it to the vector, which finally has 1 + 7 features
            data_bpm_statistics.append(bpm_statisticsVector)

        # Next feature are MFCCs; we take 12 coefficients; for each coefficient, we have around 40 values per second
        mfccs=librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)
        mfccVector=[]
        for mfccCoefficient in mfccs: # we transform this time series by taking again statistics over the values
            mfccVector.append(statistics(mfccCoefficient))

        # Finally, this vector should have 12 * 7 features
        data_mfcc.append(np.array(mfccVector).flatten())


        # Last feature set - chroma (which is roughly similar to actual notes)
        chroma=librosa.feature.chroma_stft(y=y, sr=sr);
        chromaVector=[]
        for chr in chroma: # similar to before, we get a number of time-series
            chromaVector.append(statistics(chr)) # and we resolve that by taking statistics over the time series
        # Finally, this vector should be be 12 * 7 features
        data_chroma.append(np.array(chromaVector).flatten())

        #    bar.update(indexSample)

    print(".... done" + " (" + str(datetime.datetime.now()) + ")")

    with open('/temp/MAL/data/audio/data_bpm.pickel', 'wb') as f:
        pickle.dump(data_bpm, f)

    with open('/temp/MAL/data/audio/data_bpm_statistics.pickel', 'wb') as f:
        pickle.dump(data_bpm_statistics, f)

    with open('/temp/MAL/data/audio/data_chroma.pickel', 'wb') as f:
        pickle.dump(data_chroma, f)

    with open('/temp/MAL/data/audio/data_mfcc.pickel', 'wb') as f:
        pickle.dump(data_mfcc, f)

    with open('/temp/MAL/data/audio/target.pickel', 'wb') as f:
        pickle.dump(target, f)

    return [data_bpm, data_bpm_statistics, data_chroma, data_mfcc, target]

extractAudioFeatures()