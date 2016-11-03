set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DHAVE_SSE2 -DDSFMT_MEXP=19937")  # for dSFMT

set(SOURCE_FILES dSFMT/dSFMT.cpp
    main.cpp lcorpus.cpp lcorpus.h random.h aliastable.h utils.h pdtm.cpp pdtm.h mpi_circular.h)

add_executable(dtm ${SOURCE_FILES})
target_link_libraries(dtm ${MPI_CXX_LIBRARIES} ${GLOG} ${GFLAGS})

add_executable(dtm_fmt dtm_formatter.cpp)
target_link_libraries(dtm_fmt ${GLOG} ${GFLAGS})
