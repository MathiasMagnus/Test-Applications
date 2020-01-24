# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.7)

# The name of our project
project (CL-Include LANGUAGES CXX)

# Find dependent libraries
find_package (OpenCL REQUIRED)
find_package (Threads REQUIRED)

# Adding source code files according to configuration
set (Files_HDRS include/${PROJECT_NAME}.hpp
                include/Options.hpp)
set (Files_SRCS source/${PROJECT_NAME}.cpp
                source/Options.cpp)
set (Files_KRNS kernel/kernel.h.cl
                kernel/kernel.cl)

# Generate the configuration file for application to locate kernel files
set (Path_KRNS ${CMAKE_CURRENT_SOURCE_DIR}/kernel)
configure_file (${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}-config.in.hpp
                ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}-config.hpp)

list (APPEND ${Files_HDRS} ${CMAKE_CURRENT_BINARY_DIR}/include/${PROJECT_NAME}-config.hpp)

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

target_compile_definitions(${PROJECT_NAME}
   PRIVATE
      CL_TARGET_OPENCL_VERSION=120
)

target_compile_options(${PROJECT_NAME}
  PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:
      -Wall     # Turn on all warnings
      -Wextra   # Turn on even more warnings
      -pedantic # Turn on strict language conformance
      -Wno-ignored-attributes # Console blows up due to cl.h
    >
    $<$<CXX_COMPILER_ID:MSVC>:
      /W4          # Turn on all (sensible) warnings
      /permissive- # Turn on strict language conformance
      /EHsc        # Specify exception handling model
    >
)

# Create filters for IDEs
set_target_properties (${PROJECT_NAME} PROPERTIES FOLDER "OpenCL")
source_group ("Headers" FILES ${Files_HDRS})
source_group ("Sources" FILES ${Files_SRCS})
source_group ("Kernels" FILES ${Files_KRNS})