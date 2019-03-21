##### CMake entry point
cmake_minimum_required (VERSION 2.6)
project (mld_crf)

##### make release version
set(CMAKE_BUILD_TYPE Release)

set (CMAKE_CXX_STANDARD 11)

##### external library setting
# OpenCV
set( OPENCV_VER       310 )

if (WIN32)
  set( OPENCV_PATH "C:/programming/lib/opencv-master/build-test/install")
  set( OPENCV_INC_DIR "${OPENCV_PATH}/include" )
  set( OPENCV_LIB_DIR "${OPENCV_PATH}/x64/vc14/lib" )
endif (WIN32)

if (UNIX)
  find_package ( OpenCV REQUIRED )
  include_directories(${OpenCV_INCLUDE_DIR})
endif (UNIX)

set( OPENCV_LIB   
  optimized opencv_world${OPENCV_VER}      debug opencv_world${OPENCV_VER}d
  optimized opencv_xfeatures2d${OPENCV_VER}   debug opencv_xfeatures2d${OPENCV_VER}d
)

########################## NO CHANGES BEYOND THIS POINT ##########################


##### include & link

# main project 
include_directories(
  ${OPENCV_INC_DIR}
)
link_directories(
  ${OPENCV_LIB_DIR}
)

##### project

FILE(GLOB MLD_CRF_SRC_FILES "src/*.cpp" "src/*.c" "src/*.h")

##### build demo program

add_executable(mld_crf 
  ${MLD_CRF_SRC_FILES}
)

if (WIN32)
  target_link_libraries(mld_crf
    ${OPENCV_LIB}
  )
endif(WIN32)

if (UNIX)
  target_link_libraries(mld_crf
          ${OpenCV_LIBRARIES}
  )
endif(UNIX)
