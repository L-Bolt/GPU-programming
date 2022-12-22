# # CMakeList.txt : CMake project for Opencl-ex1, include source and define
# # project specific logic here.
# # Tested to work on Linux, Windows and M1 macs.
# # Not tested for x86 macs. If you use the program with an x86 mac you might
# # have to edit this cmake file.

# cmake_minimum_required (VERSION 3.8)
# project(image_classifier)
# set (CMAKE_CXX_STANDARD 20)

# if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
#   set(MACOSX TRUE)
# endif()

# # Adding Opencl libs and include files to the proj
# find_package(OpenCL)
# include_directories(${OpenCL_INCLUDE_DIRS})
# link_directories(${OpenCL_LIBRARY})

# # Adding OpenCV
# find_package( OpenCV REQUIRED )
# include_directories( ${OpenCV_INCLUDE_DIRS} )

# # Adding boost
# find_package(Boost 1.56 REQUIRED)

# # Adding all source files.
# file(GLOB cnn_src CONFIGURE_DEPENDS "src/*.h" "src/*.cpp" "src/*.hpp")

# if (WIN32)
#   set(BOOST_INC "C:/local/boost_1_76_0_b1_rc2")
#   set(BOOST_LIB "C:/local/boost_1_76_0_b1_rc2/lib64-msvc-14.2/")
#   include_directories(${BOOST_INC})
#   link_directories(${BOOST_LIB})
# elseif(UNIX AND NOT MACOSX)
#   message("OS: Linux")
#   include_directories(${Boost_INCLUDE_DIR})
# elseif(MACOSX)
#   message("OS: MacOS")
#   include_directories("/opt/homebrew/Cellar/boost/1.80.0/include")
# endif()

# # Compile files in Core and OpenCL external libs
# if (WIN32)
#   file(GLOB BOOST_SRC "${BOOST_LIB}/*.lib")
# endif()

# file(GLOB CORE_SRC "lib/Core/*.cpp" "lib/Core/*.c")
# file(GLOB OPENCL_SRC "lib/OpenCL/OpenCL.hpp")

# if(MSVC)
#   add_definitions(-D_CRT_SECURE_NO_WARNINGS)
# endif()

# # Add source to this project's executable.
# add_executable (image_classifier ${cnn_src} ${CORE_SRC} ${OPENCL_SRC} )

# target_include_directories (image_classifier PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} "CORE" "OPENCL" "src")
# target_link_libraries (image_classifier ${OpenCL_LIBRARY} ${OpenCV_LIBS} )

# # TODO: Add tests and install targets if needed.




# CMakeList.txt : CMake project for Opencl-ex1, include source and define
# project specific logic here.
# Tested to work on Linux, Windows and M1 macs.
# Not tested for x86 macs. If you use the program with an x86 mac you might
# have to edit this cmake file.
cmake_minimum_required (VERSION 3.8)
project(image_classifier)
set (CMAKE_CXX_STANDARD 20)

#Adding Opencl libs and include files to the proj
find_package(OpenCL)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

find_package( OpenCV REQUIRED )
if (NOT MACOSX)
  include_directories( ${OpenCV_INCLUDE_DIRS} )
else()
  include_directories("/usr/local/include/opencv4")
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(MACOSX TRUE)
endif()

find_package(Boost 1.56 REQUIRED)

#adding the Boost libs to the proj
if (WIN32)
  set(BOOST_INC "C:/local/boost_1_76_0_b1_rc2")
  set(BOOST_LIB "C:/local/boost_1_76_0_b1_rc2/lib64-msvc-14.2/")

  include_directories(${BOOST_INC})
  link_directories(${BOOST_LIB})

elseif(UNIX AND NOT MACOSX)
  message("OS: Linux")
  include_directories(${Boost_INCLUDE_DIR})
elseif(MACOSX)
  message("OS: MacOS")
  include_directories("/opt/homebrew/Cellar/boost/1.80.0/include")
endif()

file(GLOB cnn_src CONFIGURE_DEPENDS "src/include/*.h" "src/*.cpp")

#compile files in COre and OpenCL external libs
file(GLOB CORE_SRC "lib/Core/*.cpp" "lib/Core/*.c")
file(GLOB OPENCL_SRC "lib/OpenCL/OpenCL.hpp")

if (WIN32)
  file(GLOB BOOST_SRC "${BOOST_LIB}/*.lib")
endif()

if (MSVC)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

# Add source to this project's executable.
add_executable (image_classifier ${cnn_src} ${CORE_SRC} ${OPENCL_SRC} )
target_include_directories (image_classifier PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} "CORE" "OPENCL" "src" "lib")
if (NOT MACOSX)
  target_link_libraries (image_classifier ${OpenCL_LIBRARY} ${OpenCV_LIBS} dl boost_system) #imagehlp)
else()
  target_link_libraries (image_classifier ${OpenCL_LIBRARY} ${OpenCV_LIBS}) #imagehlp)
endif()

# TODO: Add tests and install targets if needed.
