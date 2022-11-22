cmake_minimum_required(VERSION 2.8)
PROJECT( MotionInterpolation )
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE( MotionInterpolation MotionInterpolation.cpp )
TARGET_LINK_LIBRARIES( MotionInterpolation ${OpenCV_LIBS} )

find_package(OpenMP)


if(OpenMP_CXX_FOUND)
  target_link_libraries(${PROJECT_NAME}
    # OpenMP::OpenMP_CXX is an imported target with INTERFACE_**
    $<$<CONFIG:Release>:OpenMP::OpenMP_CXX> )
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# MPI
find_package(MPI REQUIRED)
target_link_libraries(MotionInterpolation PUBLIC MPI::MPI_CXX)