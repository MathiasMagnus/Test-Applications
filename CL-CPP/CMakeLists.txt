# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.7)

# The name of our project
project (CL-CPP LANGUAGES CXX)

# Find dependent libraries
find_package (OpenCL REQUIRED)
find_package (Threads REQUIRED)

# Adding source code files according to configuration
set (Files_HDRS ${CMAKE_CURRENT_SOURCE_DIR}/include/CL-CPP.hpp)
set (Files_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/source/CL-CPP.cpp)
set (Files_KRNS ${CMAKE_CURRENT_SOURCE_DIR}/kernel/saxpy.cl)

# Generate the configuration file for application to locate kernel files
configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/include/CL-CPP-config.in.hpp"
                "${CMAKE_CURRENT_BINARY_DIR}/include/CL-CPP-config.hpp")

list (APPEND ${Files_HDRS} "${CMAKE_BINARY_DIR}/include/OpenCL-C++-API-config.hpp")

# Specify executable sources
add_executable (${PROJECT_NAME} ${Files_HDRS}
                                ${Files_SRCS}
                                ${Files_KRNS})

# Append our project's include directory to the "#include <>" paths
target_include_directories (${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include/
                                                    ${CMAKE_CURRENT_BINARY_DIR}/include/)

# Link dependant libraries
target_link_libraries (${PROJECT_NAME} PRIVATE OpenCL::OpenCL
                                               Threads::Threads)

# Specify strictly conforming required language standard
set_target_properties (${PROJECT_NAME} PROPERTIES CXX_STANDARD 14
                                                  CXX_STANDARD_REQUIRED ON
												  CXX_EXTENSIONS OFF)

# The implementation of device to host cl::copy does not comply with MSVC checked iterators
target_compile_definitions (${PROJECT_NAME} PRIVATE $<$<CXX_COMPILER_ID:MSVC>:_SCL_SECURE_NO_WARNINGS>)

# cl2.hpp contains attributes on basic types (cl_int, cl_double, etc.) that GCC does not like
target_compile_options (${PROJECT_NAME} PRIVATE $<$<CXX_COMPILER_ID:GNU>:"-Wno-ignored-attributes">)

# Create filters for IDEs
set_target_properties (${PROJECT_NAME} PROPERTIES FOLDER "OpenCL")
source_group ("Headers" FILES ${Files_HDRS})
source_group ("Sources" FILES ${Files_SRCS})
source_group ("Kernels" FILES ${Files_KRNS})