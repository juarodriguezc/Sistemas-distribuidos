cmake_minimum_required(VERSION 3.8)

PROJECT( MotionInterpolation LANGUAGES CXX CUDA )
enable_language(CUDA)
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE( MotionInterpolation MotionInterpolation.cu )
TARGET_LINK_LIBRARIES( MotionInterpolation ${OpenCV_LIBS} )

#if(OpenMP_CXX_FOUND)
#  target_link_libraries(${PROJECT_NAME}
#    # OpenMP::OpenMP_CXX is an imported target with INTERFACE_**
#    $<$<CONFIG:Release>:OpenMP::OpenMP_CXX> )
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#endif()