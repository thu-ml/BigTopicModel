cmake_minimum_required (VERSION 2.6)
project (BigTopicModel)

set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -g -pthread -Ofast -march=native -inline-factor=400")

find_package(OpenMP)
find_package(MPI)
find_package(ZLIB) 
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(${MPI_INCLUDE_PATH})
include_directories(${ZLIB_INCLUDE_PATH})
include_directories(third_party/include)

# TODO check MPI_Count and MPI_Aint is 8 bytes
set(THIRD_PARTY_LIBRARIES ${BigTopicModel_SOURCE_DIR}/third_party/lib/libbigmpi.so)

add_subdirectory(src/util)
add_subdirectory(src/engine)
add_subdirectory(src/model)
add_subdirectory(src/formatter)
add_subdirectory(src/benchmark)
#add_subdirectory(src/test)
configure_file(src/run.py run.py)
configure_file(src/run_tianhe.py run_tianhe.py)
configure_file(src/format.py format.py)


#enable_testing()
#add_test(unit_test ${PROJECT_BINARY_DIR}/src/test/unit_test)
