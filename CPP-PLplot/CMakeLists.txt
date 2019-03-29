# The minimum version of CMake necessary to build this project
cmake_minimum_required (VERSION 3.7)

# The name of our project
project (CPP-PLplot LANGUAGES CXX)

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

# Find dependencies
find_package(plplot CONFIG REQUIRED)

# Adding source code files according to configuration
set (Files_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/CPP-PLplot.cpp)

# Specify executable sources
add_executable (${PROJECT_NAME} ${Files_SRCS})

# Link to dependencies
target_link_libraries(${PROJECT_NAME} PRIVATE $<TARGET_NAME_IF_EXISTS:PLPLOT::plplotcxx>
                                              $<TARGET_NAME_IF_EXISTS:plplotcxx>)

#target_include_directories(${PROJECT_NAME} PRIVATE $<$<BOOL:${PLplot_FOUND}>:${PLplot_INCLUDE_DIRS}>)

# Specify strictly conforming required language standard
set_target_properties (${PROJECT_NAME} PROPERTIES CXX_STANDARD 14
                                                  CXX_STANDARD_REQUIRED ON
                                                  CXX_EXTENSIONS OFF)

# Create filters for IDEs
set_target_properties (${PROJECT_NAME} PROPERTIES FOLDER "C++")
source_group ("Sources" FILES ${Files_SRCS})