#!/usr/bin/env bash

mkdir data/prediction_result
mkdir data/user_data
mkdir data/user_data/semantic
mkdir data/user_data/feature
mkdir data/user_data/models

python codes/train.py
