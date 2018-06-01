import caffe
import numpy as np

BIN_MEAN_FILE = 'lmdb/data_lmdb/image_mean.binaryproto'
NPY_MEAN_FILE = 'lmdb/data_lmdb/image_mean.npy'

blob = caffe.proto.caffe_pb2.BlobProto()
data = open( BIN_MEAN_FILE , 'rb' ).read()
blob.ParseFromString(data)
arr = np.array( caffe.io.blobproto_to_array(blob) )
out = arr[0]
np.save( NPY_MEAN_FILE , out )
