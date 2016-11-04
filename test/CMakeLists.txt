project(Test)

include_directories(${BigTopicModel_SOURCE_DIR}/src/util)
include_directories(${BigTopicModel_SOURCE_DIR}/src/model)
include_directories(${BigTopicModel_SOURCE_DIR}/src/engine)
include_directories(${BigTopicModel_SOURCE_DIR}/src/model/hlda)

add_executable(run_test test_atomic.cpp test_parallel_tree.cpp)
target_link_libraries(run_test gtest gtest_main gmock hlda_lib util boost_thread boost_system)
