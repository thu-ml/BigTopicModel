include_directories(${BigTopicModel_SOURCE_DIR}/src/util)
include_directories(${BigTopicModel_SOURCE_DIR}/src/model)
include_directories(${BigTopicModel_SOURCE_DIR}/src/engine)

add_executable(med_lda main.cpp MedLDA.cpp tron.cpp linear.cpp dscal.c dnrm2.c ddot.c daxpy.c)
target_link_libraries(med_lda util ${MPI_LIBRARIES} ${GLOG} ${GFLAGS})
