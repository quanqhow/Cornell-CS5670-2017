#!/bin/bash
# You need to change this to your downloaded project path !!!!
WORKING_PATH=/home/ubuntu/Cornell-CS5670-2017/Project5_CovNet/


set -x
sudo apt-get update
set -e

# install caffe's dependencies
sudo apt-get -y install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get -y install --no-install-recommends libboost-all-dev
sudo apt-get -y install libatlas-base-dev
sudo apt-get -y install libgflags-dev libgoogle-glog-dev liblmdb-dev

# download caffe and edit makefile
cd ~/

#if [[ ! -d ~/caffe ]]; then
git clone https://github.com/BVLC/caffe
#fi

cd ~/caffe
# pull out a specific commit (for consistency)
git fetch --all
git checkout master
git reset --hard f623d04c0e05b9c047cdeb5cbafc53d4ff0989bb
# copy & edit Makefile to use CPU only and build with python layers
cp Makefile.config.example Makefile.config
sed -i 's/# CPU_ONLY := 1/CPU_ONLY := 1/g' Makefile.config
sed -i 's/# WITH_PYTHON_LAYER := 1/WITH_PYTHON_LAYER := 1/g' Makefile.config

# make it!
make clean
make -j$(nproc) all
make -j$(nproc) test
make runtest

# download python requirements
# update pip
sudo python -m pip install --upgrade --force pip   
sudo pip install setuptools==33.1.1

cd $WORKING_PATH
for req in $(cat requirements.txt); do sudo pip install "$req"; done

cd $WORKING_PATH/../modules/ipython-4.0.0/
sudo python setup.py install

cd $WORKING_PATH
for req in $(cat requirements2.txt); do sudo pip install "$req"; done
#make pycaffe
cd ~/caffe
make -j$(nproc) pycaffe

# for the ipython notebooks
sudo pip install jupyter
# pytest
sudo pip install pytest pytest-sugar
# upgrade numpy
sudo pip install --upgrade numpy
sudo pip install pickleshare
sudo pip install simplegeneric

# export python path
export PYTHONPATH=/home/ubuntu/caffe/python:$PYTHONPATH

python $WORKING_PATH/test_student.py

