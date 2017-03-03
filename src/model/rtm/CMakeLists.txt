include_directories(${BigTopicModel_SOURCE_DIR}/src/util)
include_directories(${BigTopicModel_SOURCE_DIR}/src/model)
include_directories(${BigTopicModel_SOURCE_DIR}/src/engine)
include_directories(${BigTopicModel_SOURCE_DIR}/third_party) 

add_executable(rtm main.cpp RTM.cpp ap.cpp cholesky.cpp cokus.cpp MVGaussian.cpp utils_rtm.cpp spdinverse.cpp PolyaGamma.cpp)
target_link_libraries(rtm util ${MPI_LIBRARIES} ${GLOG} ${GFLAGS})
