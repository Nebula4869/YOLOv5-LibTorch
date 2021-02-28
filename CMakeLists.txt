cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(YOLOv5LibTorch)
SET(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(OpenCV REQUIRED)

if (OpenCV_FOUND)
    # If the package has been found, several variables will
    # be set, you can find the full list with descriptions
    # in the OpenCVConfig.cmake file.
    # Print some message showing some of them
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}" \n)

else ()
    message(FATAL_ERROR "Could not locate OpenCV" \n)
endif()

set(Torch_DIR libtorch/share/cmake/Torch)
find_package(Torch PATHS ${Torch_DIR} NO_DEFAULT REQUIRED)
if (Torch_FOUND)
    message(STATUS "Torch library found!")
    message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}" \n)

else ()
    message(FATAL_ERROR "Could not locate Torch" \n)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_PREFIX_PATH /home/nebula/libtorch)
find_package(Torch REQUIRED)
add_executable(YOLOv5LibTorch src/YOLOv5LibTorch.cpp)

include_directories(
	${PROJECT_SOURCE_DIR}/include
	${OpenCV_INCLUDE_DIRS}
	${TORCH_INCLUDE_DIRS}
	)


target_link_libraries(YOLOv5LibTorch ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_property(TARGET YOLOv5LibTorch PROPERTY CXX_STANDARD 14)
