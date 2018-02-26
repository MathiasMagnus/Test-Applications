# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.7)

# The name of our project
project (clFFT_Test LANGUAGES CXX)

# Find dependent libraries
find_package(clFFT REQUIRED)
find_package(OpenCL REQUIRED)

# Adding source code files according to configuration
set (Files_HDRS ${PROJECT_SOURCE_DIR}/inc/Header.hpp)
set (Files_SRCS ${PROJECT_SOURCE_DIR}/src/Source.cpp)

# Specify executable sources
add_executable (${PROJECT_NAME} ${Files_HDRS} ${Files_SRCS})

# Append our project's include directory to the "#include <>" paths
target_include_directories (${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/inc/)

# Link dependant libraries
target_link_libraries(${PROJECT_NAME} PUBLIC OpenCL::OpenCL clFFT)

# Create filters for IDEs
set_target_properties (${PROJECT_NAME} PROPERTIES FOLDER "Test")
source_group ("Headers" FILES ${Files_HDRS})
source_group ("Sources" FILES ${Files_SRCS})