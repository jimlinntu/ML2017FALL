#!/bin/bash
python3 train.py JimResidualNetwork --train_filename $1 --n_epochs 100 --batch_size 32 
