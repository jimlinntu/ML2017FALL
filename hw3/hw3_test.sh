#!/bin/bash
python3 train.py JimResidualNetwork --test_filename $1 --testout $2 --batch_size 32 --timestamp "time: 20171114_2239" 
