#!/bin/bash

# conda activate data_processing

# pip install editdistance
# pip install aksharamukha

source ../../.env
[ -z "$DATA_HOME" ] && echo "ERROR: DATA_HOME not set" && exit 1

cd data-cleaning-pipeline

# copy necessary config files
cp config/am-ET.yaml config/ti-ET.yaml

python ../clean_data.py $DATA_HOME