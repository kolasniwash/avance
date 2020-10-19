#!/bin/bash

python3 preprocessing.py
python3 create_folds.py --n_folds 5
python3 train.py --model rf


