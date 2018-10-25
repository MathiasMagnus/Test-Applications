cmake_minimum_required(VERSION 3.2.2)

project(SYCL-MultiCompilationUnit LANGUAGES CXX)

# Include modules
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
find_package(ComputeCpp)

add_executable(${PROJECT_NAME} Other.hpp
                               Other.cpp
                               Main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(${PROJECT_NAME}
                      PROPERTIES CXX_STANDARD 14
                                 CXX_STANDARD_REQUIRED ON)

add_sycl_to_target(TARGET ${PROJECT_NAME}
                   SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp
				           ${CMAKE_CURRENT_SOURCE_DIR}/Other.cpp)