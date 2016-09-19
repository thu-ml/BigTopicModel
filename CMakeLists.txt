cmake_minimum_required (VERSION 2.6)
project (BigTopicModel)

# g++ doesn't support the -inline-factor option
# g++: error: unrecognized command line option ‘-inline-factor=400’
#set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -g -pthread -Ofast -march=native -inline-factor=400")
# on juncluster, cmake find intel mpi by default, so we need to specify mpicxx to get mpicxx of openmpi
# using intel mpi won't need set MPI_CXX_COMPILER
#set(MPI_CXX_COMPILER "mpicxx")
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -g -pthread -Ofast -march=native -inline-factor=400")

find_package(MPI)
find_package(ZLIB) 
find_package(OpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(${MPI_INCLUDE_PATH})
include_directories(${ZLIB_INCLUDE_PATH})
#include_directories(third_party/BigMPI/src)
include_directories(third_party/glog/src)
include_directories(third_party/gflags/build/include)

# TODO check MPI_Count and MPI_Aint is 8 bytes
#set(BIGMPI ${BigTopicModel_SOURCE_DIR}/third_party/BigMPI/build/src/libbigmpi.so)
set(GLOG ${BigTopicModel_SOURCE_DIR}/third_party/glog/build/libglog.a)
set(GFLAGS ${BigTopicModel_SOURCE_DIR}/third_party/gflags/build/lib/libgflags.a)

add_subdirectory(src/util)
add_subdirectory(src/engine)
add_subdirectory(src/model)
add_subdirectory(src/formatter)
add_subdirectory(src/benchmark)
#add_subdirectory(src/test)
configure_file(src/run.py run.py)
configure_file(src/run_tianhe.py run_tianhe.py)
configure_file(src/format.py format.py)

#add_library(glog STATIC IMPORTED)
#set_target_properties(glog PROPERTIES IMPORTED_LOCATION )

#enable_testing()
#add_test(unit_test ${PROJECT_BINARY_DIR}/src/test/unit_test)
