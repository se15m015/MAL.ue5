import pickle

def loadImages():
    with open('/temp/MAL/data/images/data.pickel', 'rb') as f:
        data = pickle.load(f)

    with open('/temp/MAL/data/images/dataOpenCV_1D.pickel', 'rb') as f:
        dataOpenCV_1D = pickle.load(f)

    with open('/temp/MAL/data/images/dataOpenCV_2D.pickel', 'rb') as f:
        dataOpenCV_2D = pickle.load(f)

    with open('/temp/MAL/data/images/dataOpenCV_3D.pickel', 'rb') as f:
        dataOpenCV_3D = pickle.load(f)

    with open('/temp/MAL/data/images/target.pickel', 'rb') as f:
        target = pickle.load(f)

    return [data, dataOpenCV_1D, dataOpenCV_2D, dataOpenCV_3D, target]

