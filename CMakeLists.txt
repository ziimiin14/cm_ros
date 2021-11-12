cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cm_ros)

add_compile_options(-std=c++14)

# Locate the cmake file of torchlib
# set(Torch_DIR "~/libtorch/share/cmake/Torch")
set(Torch_DIR "~/.local/lib/python3.6/site-packages/torch/share/cmake/Torch") ## torch directory in xavier nx board
#set(Torch_DIR "$ENV{TORCH_ROOT}/torch/share/cmake/Torch/")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  dvs_msgs
)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES ${PROJECT_NAME}
 CATKIN_DEPENDS roscpp rospy std_msgs dvs_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(${PROJECT_NAME}
  src/CM_Model.cpp
  src/CM_Opt.cpp
  src/cm_ros.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${TORCH_LIBRARIES}
)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(cm_node src/cm_node.cpp)


## Specify libraries to link a library or executable target against
target_link_libraries(cm_node
  ${TORCH_LIBRARIES}
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
)


