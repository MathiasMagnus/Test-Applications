cmake_minimum_required(VERSION 3.2.2)

project(SYCL-Test LANGUAGES CXX)

message(STATUS "COMPUTECPP_PACKAGE_ROOT_DIR: ${COMPUTECPP_PACKAGE_ROOT_DIR}")

# Include modules
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
include(FindComputeCpp)

add_executable(${PROJECT_NAME} Main.cpp)

set_target_properties(${PROJECT_NAME}
                      PROPERTIES CXX_STANDARD 14
                                 CXX_STANDARD_REQUIRED ON)

add_sycl_to_target(${PROJECT_NAME}
                   ${CMAKE_CURRENT_BINARY_DIR}
                   ${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp)