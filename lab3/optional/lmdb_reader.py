# -*- coding: utf-8 -*-
import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt

# Source:

#   Read LevelDB/LMDB
#       http://research.beenfrog.com/code/2015/03/28/read-leveldb-lmdb-for-caffe-with-python.html

#   Plot image
#       http://www.pyimagesearch.com/2014/11/03/display-matplotlib-rgb-image/

#   Creating LMDB in python
#       http://deepdish.io/2015/04/28/creating-lmdb-in-python/

id_path = 1
visualize = False
lmdb_path = [
"/home/caffe/examples/mnist/mnist_train_lmdb/",
"/home/examples/imagenet/ilsvrc12_train_lmdb/"]

lmdb_env = lmdb.open(lmdb_path[id_path])
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

for key, value in lmdb_cursor:
    
    datum.ParseFromString(value)

    label = datum.label
    
    data = caffe.io.datum_to_array(datum)
    # OR
    # Read the datum.data
    #data = numpy.array(bytearray(datum.data))\
    #    .reshape(datum.channels, datum.height, datum.width)
    
    if id_path == 0 :
        im = data.astype(float)[0]
    elif id_path == 1:
        im = data.astype(np.uint8)
        # original (datum.channels, datum.height, datum.width)
        #CxHxW to HxWxC
        im = np.transpose(im, (1, 2, 0))

    #if label == 999:
    print "key: ",key
    print "label ", label

    #matplotlib.pyplot.imshow()
    #MxN – luminance (grayscale, float array only)
    #MxNx3 – RGB (float or uint8 array)
    #MxNx4 – RGBA (float or uint8 array)
    if visualize:
        plt.imshow(im)
        plt.show()
