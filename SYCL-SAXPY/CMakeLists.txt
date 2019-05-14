cmake_minimum_required(VERSION 3.2.2)

project(SYCL-SAXPY LANGUAGES CXX)

# Include modules
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
find_package(TCLAP REQUIRED)
find_package(ComputeCpp REQUIRED)

add_executable(${PROJECT_NAME} Main.cpp
                               Options.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}
                                                   ${TCLAP_INCLUDE_PATH})

set_target_properties(${PROJECT_NAME}
                      PROPERTIES CXX_STANDARD 14
                                 CXX_STANDARD_REQUIRED ON)

add_sycl_to_target(TARGET ${PROJECT_NAME}
                   SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp)