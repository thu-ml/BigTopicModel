#!/bin/bash

#get and enter current path
CUR=`dirname "$0"`
CUR=`cd "$CUR"; pwd`

# TODO : do I need to checkout to a given version of build a special repo like petuum?
# the given version of BigMPI 7b84f352bddd48dbae630c1bd98fd04255602530

if [ ! -d "$CUR/third_party" ]; then
	# Control will enter here if $DIRECTORY doesn't exist.
	echo "$CUR/third_party doesn't exist"
	mkdir -p third_party
	pushd third_party
	DEP=$CUR/third_party
	if [ ! -d "$DEP/glog" ]; then
		wget https://github.com/google/glog/archive/master.zip; mv master.zip glog.zip
		unzip glog.zip
		mv glog-master glog
		pushd glog
		mkdir -p build
		pushd build
		cmake ..
		make
		popd
		popd
	fi
	if [ ! -d "$DEP/gflags" ]; then
		wget https://github.com/gflags/gflags/archive/master.zip; mv master.zip gflags.zip
		unzip gflags.zip
		mv gflags-master gflags
		pushd gflags
		mkdir -p build
		pushd build
		cmake ..
		make
		popd
		popd
	fi
	if [ ! -d "$DEP/BigMPI" ]; then
		wget https://github.com/jeffhammond/BigMPI/archive/master.zip; mv master.zip BigMPI.zip
		unzip BigMPI.zip
		mv BigMPI-master BigMPI 
		pushd BigMPI 
		cp $CUR/resource/bigmpi.patch ./
		cp $CUR/resource/CMakeLists.txt.BigMPI ./CMakeLists.txt
		pushd src
		patch -s -p1 < ../bigmpi.patch
		popd
		mkdir -p build
		pushd build
		cmake ..
		make
		popd
		popd
	fi
	popd
else
	echo "$CUR/third_party already exists, please make sure the code is set up correctlly!"
fi

exit 0
