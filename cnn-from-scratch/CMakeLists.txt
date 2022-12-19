# CMakeList.txt : CMake project for Opencl-ex1, include source and define
# project specific logic here.

cmake_minimum_required (VERSION 3.8)
project(test-cnn)
set (CMAKE_CXX_STANDARD 20)

#Adding Opencl libs and include files to the proj
find_package(OpenCL)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

#adding Opencl
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB cnn_src CONFIGURE_DEPENDS "src/*.h" "src/*.cpp")

if (WIN32)
  set(BOOST_INC "C:/local/boost_1_76_0_b1_rc2")
  set(BOOST_LIB "C:/local/boost_1_76_0_b1_rc2/lib64-msvc-14.2/")

  include_directories(${BOOST_INC})
  link_directories(${BOOST_LIB})

elseif(UNIX)
  message("OS: Linux")
  include_directories(${Boost_INCLUDE_DIR})
endif()

#compile files in COre and OpenCL external libs
file(GLOB CORE_SRC "Core/*.cpp" "Core/*.c")
file(GLOB OPENCL_SRC "OpenCL/*.cpp" "OpenCL/*.c")
if (WIN32)
  file(GLOB BOOST_SRC "${BOOST_LIB}/*.lib")
endif()

if(MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Add source to this project's executable.
add_executable (test-cnn ${cnn_src} ${CORE_SRC} ${OPENCL_SRC} )

target_include_directories (test-cnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} "CORE" "OPENCL" "src")
target_link_libraries (test-cnn ${OpenCL_LIBRARY} ${OpenCV_LIBS} )

# TODO: Add tests and install targets if needed.
