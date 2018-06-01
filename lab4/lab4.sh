#!/bin/bash
#PBS -k o
#PBS -j oe
export PYTHONPATH="/home/doleinik/caffe/python"
export CAFFE_HOME="/home/doleinik/caffe"
export PATH="/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/$
export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"
PBS_O_WORKDIR=/home/student21m07/labs/lab4

for i in {1..10}
do
  python3 $PBS_O_WORKDIR/simple_model.py
done
