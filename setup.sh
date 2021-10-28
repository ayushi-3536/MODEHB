#!/bin/bash

# get the python version
v="$(python -V)"
echo $v
v="${v//[^0-9]/}"
v="${v:0:2}"

cd src

# install dependencies
if [ ! -d "$dependencies" ]; then
  mkdir dependencies
fi

cd dependencies

#To Test NASBenchmark-201
git clone --recurse-submodules https://github.com/D-X-Y/AutoDL-Projects.git XAutoDL
cd XAutoDL
python XAutoDL/setup.py install



