#!/bin/bash                                                                                                           
set -e          
uv sync
git clone https://github.com/clab/fast_align.git src/fast_align                                                                             
mkdir -p src/fast_align/build
cd src/fast_align/build && cmake .. && make