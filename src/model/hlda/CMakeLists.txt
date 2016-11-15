include_directories(${BigTopicModel_SOURCE_DIR}/src/util)
include_directories(${BigTopicModel_SOURCE_DIR}/src/model)
include_directories(${BigTopicModel_SOURCE_DIR}/src/engine)
include_directories(/opt/intel/mkl/include)
link_directories(/opt/intel/mkl/lib/intel64)

add_library(hlda_lib tree.cpp base_hlda.cpp document.cpp collapsed_sampling.cpp partially_collapsed_sampling.cpp parallel_tree.cpp distributed_tree.cpp)
add_executable(hlda main.cpp)
target_link_libraries(hlda hlda_lib util ${MPI_LIBRARIES} ${GLOG} ${GFLAGS} mkl_intel_lp64 mkl_core mkl_sequential m dl boost_thread boost_system)

add_executable(test_mpi test_mpi.cpp)
target_link_libraries(test_mpi util ${MPI_LIBRARIES} ${GLOG} ${GFLAGS} boost_thread boost_system)

add_executable(test2 test2.cpp)

add_executable(test_parallel_tree test_parallel_tree.cpp)
target_link_libraries(test_parallel_tree hlda_lib util ${MPI_LIBRARIES} ${GLOG} ${GFLAGS} mkl_intel_lp64 mkl_core mkl_sequential m dl boost_thread boost_system)
