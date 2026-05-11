#!/bin/bash


source ../../.env
[ -z "$DATA_HOME" ] && echo "ERROR: DATA_HOME not set" && exit 1

python romanizer_analysis.py $DATA_HOME