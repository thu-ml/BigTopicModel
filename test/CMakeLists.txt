project(Test)

include_directories(${BigTopicModel_SOURCE_DIR}/src/util)
include_directories(${BigTopicModel_SOURCE_DIR}/src/model)
include_directories(${BigTopicModel_SOURCE_DIR}/src/engine)

add_executable(run_test test_atomic.cpp)
target_link_libraries(run_test gtest gtest_main gmock)
