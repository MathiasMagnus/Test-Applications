cmake_minimum_required(VERSION 3.2.2)

project(SYCL-PlaceholderAccessor LANGUAGES CXX)

if (MSVC)
  string(REGEX REPLACE "/W[0-9]" "/W4" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})	
endif (MSVC)

# Include modules
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)
find_package(ComputeCpp)

add_executable(${PROJECT_NAME} Main.cpp)

set_target_properties(${PROJECT_NAME}
                      PROPERTIES CXX_STANDARD 14
                                 CXX_STANDARD_REQUIRED ON)

add_sycl_to_target(TARGET ${PROJECT_NAME}
                   SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp)