#BigTopicModel

Big Topic Model is a fast engine for running large-scale Topic Models. 


###Got third party dependencies

Make sure you can access github.com , and run the the following command

```
$ ./set_third_party.sh
```

###Set the data directory

Set the data director under root direcotry
```
$ ln -sf  where/you/store/the/data/ data
```

###Compilation

First, run build.sh under the root directory to generate release and debug folder.
```
$ ./build.sh
```
Second, using make to compile.
```
$ cd release/
$ make
```

###Run the model

Modify the run.sh under <kbd>release</kbd> folder, to run your model.
```
$ cd release/
$ ./run.sh
```
