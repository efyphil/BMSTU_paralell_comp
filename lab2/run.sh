#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python $DIR/img_to_text.py $DIR/vlados.jpg
nvcc $DIR/kernel1.cu -std=c++11 -arch=compute_37 -code=sm_37

qsub $DIR/script.sh

