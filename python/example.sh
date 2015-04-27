#!/bin/bash

./svm2classcuda.py -v 3 -g 0.7 -c 3.0 heart_scale
./grid.py  heart_scale
./easy.py  heart_scale
