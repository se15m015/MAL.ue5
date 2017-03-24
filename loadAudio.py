import pickle

def loadAudio():

    with open('/temp/MAL/data/audio/data_bpm.pickel', 'rb') as f:
        data_bpm = pickle.load(f)

    with open('/temp/MAL/data/audio/data_bpm_statistics.pickel', 'rb') as f:
        data_bpm_statistics = pickle.load(f)

    with open('/temp/MAL/data/audio/data_chroma.pickel', 'rb') as f:
        data_chroma = pickle.load(f)

    with open('/temp/MAL/data/audio/data_mfcc.pickel', 'rb') as f:
        data_mfcc = pickle.load(f)

    with open('/temp/MAL/data/audio/target.pickel', 'rb') as f:
        target = pickle.load(f)

    return [data_bpm, data_bpm_statistics, data_chroma, data_mfcc, target]