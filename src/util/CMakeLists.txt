include_directories(${BigTopicModel_SOURCE_DIR}/src/util)
include_directories(${BigTopicModel_SOURCE_DIR}/src/engine)
add_library(util corpus.cpp gzstream.cpp utils.cpp global_lock.cpp)
target_link_libraries(util ${GLOG})
