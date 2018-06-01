#!/bin/bash
#PBS -k o
#PBS -j oe
export PYTHONPATH="/home/doleinik/caffe/python"
export CAFFE_HOME="/home/doleinik/caffe"
export PATH="/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/home/caffe/build/tools:/usr/lib/jvm/java-8-oracle/bin:/usr/lib/jvm/java-8-oracle/db/bin:/usr/lib/jvm/java-8-oracle/jre/bin"
PBS_O_WORKDIR=/home/student21m07/labs/lab3
for i in {1..10}
do
	python $PBS_O_WORKDIR/classific.py
done
