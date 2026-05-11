#!/bin/bash

# conda activate data_processing

source ../../.env
[ -z "$DATA_HOME" ] && echo "ERROR: DATA_HOME not set" && exit 1

# pip install "datasets<2.20"
# pip install pandas
# pip install opus_tools[all]

### download data ###
python download_data.py $DATA_HOME $HF_TOKEN
