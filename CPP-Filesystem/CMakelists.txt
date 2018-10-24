# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.7)

# The name of our project
project (CPP-Filesystem LANGUAGES CXX)

# Replace default Warning Level 3 with 4
# (/Wall is pretty-much useless on MSVC, system headers are plagued with warnings)
#
# NOTE: Unfortunately, CMake defaults to W3 and plainly setting /W4 on targets
#       cuase MSVC to emit a warning
#
#   "Command line warning D9025: overriding '/W3' with '/W4'"
#
#       Hence for minimal console output we modify the default flags before target
#       definitions
#
if (MSVC)
  string(REGEX REPLACE "/W[0-9]" "/W4" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})	
endif (MSVC)

# Adding source code files according to configuration
set (Files_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/source/CPP-Filesystem.cpp)

# Specify executable sources
add_executable (${PROJECT_NAME} ${Files_SRCS})

# Specify strictly conforming required language standard
set_target_properties (${PROJECT_NAME} PROPERTIES CXX_STANDARD 17
                                                  CXX_STANDARD_REQUIRED ON
                                                  CXX_EXTENSIONS OFF)

# Create filters for IDEs
set_target_properties (${PROJECT_NAME} PROPERTIES FOLDER "C++")
source_group ("Sources" FILES ${Files_SRCS})