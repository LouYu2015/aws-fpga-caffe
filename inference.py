import lmdb
import caffe
import numpy as np
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array

net = caffe.Net("model_deploy.pototxt", "snapshot/solver_iter_100.caffemodel",0)

env = lmdb.open("train.mdb")
txn = env.begin()
cursor = txn.cursor()
X = []
y = []
idxs = []
for idx, (key, value) in enumerate(cursor):
    if idx > 10:
        break
    datum = caffe_pb2.Datum()
    datum.ParseFromString(value)
    X.append(np.array(datum_to_array(datum)).swapaxes(0, 2))
    y.append(datum.label)
    idxs.append(idx)
    print(net.forward_all(data=np.array(datum_to_array(datum))))

