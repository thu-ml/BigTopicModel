# BigTopicModel

Big Topic Model is a fast engine for running large-scale Topic Models. It uses a hybrid data and model parallel mechanism to accelerate and performs 3~5 times faster than the state-of-the-art ones on some general dataset. It supports a set of tpoic models including LDA, DTM, MedLDA and RTM.

### Getting Intel software Toolkit (Intel® Parallel Studio XE Professional Edition for C++ Linux* 2016???)

Download the Intel software Toolkit and we will use it to compile and run BTM. Go to http://software.intel.com to find it. 
And then set the compilevars:

```
source /opt/intel/2016/compilers_and_libraries/linux/bin/compilervars.sh intel64
```

### Getting third party dependencies

Make sure you can access github.com. To get third-party dependencies, run:

```
$ ./set_third_party.sh
```

### Setting the data directory

Set the data under root direcotry
```
$ ln -sf  where/you/store/the/data/ data
```

### Preprocessing the data

To run on a cluster, data partition is necessary. Use the split-input-data.sh in <kbd>src</kbd> folder to distribute the data for the cluster to partition. 
Unlike other current cluster computing systems, Big Topic Model not only segments data by documents, but also by words, that is a two-dimensional partition. 
Read format.py in <kbd>src</kbd> folder and modify it to do the data partition according to the parallelism.

### Compilation

First, run build.sh under the root directory to generate release and debug folder.
```
$ ./build.sh
```
Second, using make to compile.
```
$ cd release/
$ make
```

### Running the model

Modify the run.sh under <kbd>release</kbd> folder, to run your model.
```
$ cd release/
$ ./run.sh
```
