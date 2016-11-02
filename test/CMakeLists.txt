project(Test)

add_executable(run_test test_atomic.cpp)
target_link_libraries(run_test gtest gtest_main gmock)
