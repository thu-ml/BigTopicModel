#!/bin/bash

#get and enter current path
CUR=`dirname "$0"`
CUR=`cd "$CUR"; pwd`

# I should got some error test for each command
if [ ! -d "$CUR/third_party" ]; then
	# Control will enter here if $DIRECTORY doesn't exist.
	echo "$CUR/third_party doesn't exist"
	mkdir -p third_party
	pushd third_party
	DEP=$CUR/third_party
	if [ ! -d "$DEP/gflags" ]; then
		wget https://github.com/gflags/gflags/archive/v2.1.2.zip
		unzip v2.1.2.zip
		ln -s gflags-2.1.2 gflags
		pushd gflags
		mkdir -p build
		pushd build
		cmake ..
		make -j
		popd
		popd
	fi
	if [ ! -d "$DEP/glog" ]; then
		wget https://github.com/google/glog/archive/v0.3.4.zip
		unzip v0.3.4.zip
		ln -s glog-0.3.4 glog
		pushd glog
		mkdir -p build
		./configure --prefix=$PWD/build
		make -j install
		popd
	fi
	if [ ! -d "$DEP/googletest" ]; then
		wget https://github.com/google/googletest/archive/release-1.8.0.zip
		unzip release-1.8.0.zip
		ln -s googletest-release-1.8.0 googletest
	fi
	if [ ! -d "$DEP/boost" ]; then
		wget http://ml.cs.tsinghua.edu.cn/~jianfei/static/boost_1_62_0.tar.bz2
		tar jxf boost_1_62_0.tar.bz2
		ln -s boost_1_62_0 boost
		pushd boost
		mkdir -p build
		./bootstrap.sh --prefix=$PWD/build
		./b2 -j10 install
		popd
	fi
	if [ ! -d "$DEP/dSFMT" ]; then
		wget https://github.com/MersenneTwister-Lab/dSFMT/archive/c6bf8a8dab3710b7abd86c4b68d0e6b4aa5e6db1.zip
		unzip c6bf8a8dab3710b7abd86c4b68d0e6b4aa5e6db1.zip
		mv dSFMT-c6bf8a8dab3710b7abd86c4b68d0e6b4aa5e6db1 dSFMT
		rm c6bf8a8dab3710b7abd86c4b68d0e6b4aa5e6db1.zip
		mv $DEP/dSFMT/dSFMT.c $DEP/dSFMT/dSFMT.cpp
	fi
	if [ ! -d "$DEP/eigen3" ]; then
		wget http://bitbucket.org/eigen/eigen/get/3.2.10.zip
		unzip 3.2.10.zip
		mv eigen-eigen-b9cd8366d4e8 eigen3
	fi
	popd
else
	echo "$CUR/third_party already exists, please make sure the code is set up correctly!"
fi

exit 0
