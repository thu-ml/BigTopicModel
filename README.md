# BigTopicModel

Big Topic Model is a fast engine for running large-scale Topic Models. It uses a hybrid data and model parallel mechanism to accelerate and performs 3~5 times faster than the state-of-the-art ones on some general dataset. It supports a set of tpoic models including LDA, DTM, MedLDA and RTM.

## Requirement

Big Topic Model depends on several third-party libraries including google glog, gflags, dSFMT and eigen. Currently our tests compile and run mainly based on Intel libraries(Intel® Parallel Studio XE Professional Edition for C++ Linux* 2016), but other libraries like openmpi are practicable.

### Getting Intel software Toolkit

Go to http://software.intel.com to download the Intel software Toolkit. And then set the compilevars:

```
source /opt/intel/2016/compilers_and_libraries/linux/bin/compilervars.sh intel64
```

### Getting third party dependencies

Make sure you can access github.com. To get third-party dependencies, run:

```
$ ./set_third_party.sh
```

(You do not need to do this if you already have all the third-party dependencies.)

## Compilation

First, run build.sh under the root directory to generate release and debug folder.
```
$ ./build.sh
```
Second, using make to compile.
```
$ cd release/
$ make
```

## Data Preprocessing

### Setting the data directory

Set the data under root direcotry, run:
```
$ ln -sf  where/you/store/the/data/ data
```

### Preprocessing the data

To run on a cluster, data partition is necessary. Unlike other current cluster computing systems, Big Topic Model not only segments data by documents, but also by words, that is a two-dimensional partition. So the data we need is different from the traditional svm data, and we need to preprocess the data by distributing words randomly for each group before running on cluster.

Use split-input-data.sh in <kbd>src</kbd> folder to divide the raw data into groups of equal size for the cluster to format.

Now, we get a package of data for each machine. Use format.py in <kbd>src</kbd> folder to do the data partition according to the parallelism.

## Running the model

Modify the run.sh under <kbd>release</kbd> folder, to run your model.
```
$ cd release/
$ ./run.sh
```
