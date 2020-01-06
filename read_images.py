import lmdb
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array


def read_images_from_lmdb(db_name, visualize):
    env = lmdb.open(db_name)
    txn = env.begin()
    cursor = txn.cursor()
    X = []
    y = []
    idxs = []
    for idx, (key, value) in enumerate(cursor):
        datum = caffe_pb2.Datum()
        datum.ParseFromString(value)
        X.append(np.array(datum_to_array(datum)))
        y.append(datum.label)
        idxs.append(idx)
    if visualize:
        print("Visualizing a few images...")
        for i in range(9):
            img = X[i]
            plt.subplot(3,3,i+1)
            plt.imshow(img)
            plt.title(y[i])
            plt.axis('off')
        plt.show()
    print(" ".join(["Reading from", db_name, "done!"]))
    return X, y, idxs


def main():
    read_images_from_lmdb("train.mdb", True)


if __name__ == '__main__':
    main()
