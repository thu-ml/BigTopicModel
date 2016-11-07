#!/bin/bash

if [ ! -d "dSFMT" ]; then
	echo "Setting up dSFMT"
	wget https://github.com/MersenneTwister-Lab/dSFMT/archive/c6bf8a8dab3710b7abd86c4b68d0e6b4aa5e6db1.zip
	unzip c6bf8a8dab3710b7abd86c4b68d0e6b4aa5e6db1.zip
	mv ./dSFMT-c6bf8a8dab3710b7abd86c4b68d0e6b4aa5e6db1 ./dSFMT
	rm c6bf8a8dab3710b7abd86c4b68d0e6b4aa5e6db1.zip
fi

if [ ! -d "eigen3" ]; then
	echo "Setting up Eigen"
	wget http://bitbucket.org/eigen/eigen/get/3.2.10.zip
	unzip 3.2.10.zip
	mv eigen-eigen-b9cd8366d4e8 eigen3
fi
