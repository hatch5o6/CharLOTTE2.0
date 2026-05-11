#!/bin/bash

# conda activate data_processing

source ../../.env
[ -z "$DATA_HOME" ] && echo "ERROR: DATA_HOME not set" && exit 1

python make_training_data.py $DATA_HOME