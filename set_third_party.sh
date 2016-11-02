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
	if [ ! -d "$DEP/googletest" ]; then
		wget https://github.com/google/googletest/archive/release-1.8.0.zip
		unzip release-1.8.0.zip
		ln -s googletest-release-1.8.0 googletest
	fi
	popd
else
	echo "$CUR/third_party already exists, please make sure the code is set up correctlly!"
fi

exit 0
