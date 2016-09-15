include_directories(${BigTopicModel_SOURCE_DIR}/src/util)
include_directories(${BigTopicModel_SOURCE_DIR}/src/engine)
add_library(util corpus.cpp gzstream.cpp)
target_link_libraries(util ${GLOG})
