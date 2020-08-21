cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(YOLOv5LibTorch)
SET(CMAKE_BUILD_TYPE Release)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_PREFIX_PATH /home/nebula/libtorch)
find_package(Torch REQUIRED)
add_executable(YOLOv5LibTorch src/YOLOv5LibTorch.cpp)

include_directories(
${PROJECT_SOURCE_DIR}/include
${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(YOLOv5LibTorch ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_property(TARGET YOLOv5LibTorch PROPERTY CXX_STANDARD 14)
