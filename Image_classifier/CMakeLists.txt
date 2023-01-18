# Tested to work on Linux and M1 macs.
# Not tested for x86 macs. If you use the program with an x86 mac you might
# have to edit this cmake file.
cmake_minimum_required (VERSION 3.8)
project(image_classifier)
set (CMAKE_CXX_STANDARD 20)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#Adding Opencl libs and include files to the proj
find_package(OpenCL 1.2 REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})

set(IMGUI_PATH  "lib/imgui")
file(GLOB IMGUI_GLOB
    ${IMGUI_PATH}/imgui.h
    ${IMGUI_PATH}/imgui.cpp
    ${IMGUI_PATH}/imconfig.h
    ${IMGUI_PATH}/imgui_demo.cpp
    ${IMGUI_PATH}/imgui_draw.cpp
    ${IMGUI_PATH}/imgui_internal.h
    ${IMGUI_PATH}/imstb_rectpack.h
    ${IMGUI_PATH}/imstb_textedit.h
    ${IMGUI_PATH}/imstb_truetype.h
    ${IMGUI_PATH}/imgui_tables.cpp
    ${IMGUI_PATH}/imgui_widgets.cpp

    # specific bindings...
    ${IMGUI_PATH}/backends/imgui_impl_glfw.h
    ${IMGUI_PATH}/backends/imgui_impl_glfw.cpp
    ${IMGUI_PATH}/backends/imgui_impl_opengl3.h
    ${IMGUI_PATH}/backends/imgui_impl_opengl3.cpp
    ${IMGUI_PATH}/backends/imgui_impl_opengl3_loader.cpp)
add_library("imgui" STATIC ${IMGUI_GLOB})
target_include_directories("imgui" PUBLIC ${IMGUI_PATH})
target_link_libraries("imgui" PRIVATE glfw)

find_package(OpenCV REQUIRED)
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
  include_directories("/opt/homebrew/Cellar/boost/1.81.0_1/include")
endif()

file(GLOB cnn_src CONFIGURE_DEPENDS "src/include/*.h" "src/*.cpp" "src/include/*.hpp")

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

# target_compile_options(image_classifier PUBLIC
#   $<$<NOT:$<CXX_COMPILER_ID:MSVC>>: -Wall -Werror>
# )

target_include_directories (image_classifier PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} imgui "src" "lib")
if (NOT MACOSX)
  target_link_libraries (image_classifier ${OPENGL_LIBRARIES} ${OpenCL_LIBRARY} ${OpenCV_LIBS} dl boost_system imgui) #imagehlp)
else()
  target_link_libraries (image_classifier ${OPENGL_LIBRARIES} ${OpenCL_LIBRARY} ${OpenCV_LIBS} boost_system imgui) #imagehlp)
endif()

# TODO: Add tests and install targets if needed.
