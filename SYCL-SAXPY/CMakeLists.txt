cmake_minimum_required(
   VERSION
      3.2.2
)

project(SYCL-SAXPY
   LANGUAGES
      CXX
)

set(CMAKE_MODULE_PATH
   ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
)

find_package(TCLAP REQUIRED)
find_package(ComputeCpp REQUIRED)

add_executable(${PROJECT_NAME}
   Main.cpp
   Options.cpp
)

set_target_properties(${PROJECT_NAME}
   PROPERTIES
      CXX_STANDARD 14
      CXX_STANDARD_REQUIRED ON
)

target_include_directories(${PROJECT_NAME}
   PRIVATE
      ${CMAKE_CURRENT_LIST_DIR}
      ${TCLAP_INCLUDE_PATH}
)

target_compile_definitions(${PROJECT_NAME}
   PRIVATE
      CL_TARGET_OPENCL_VERSION=120
)

target_compile_options(${PROJECT_NAME}
   PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:
         /W4          # Turn on all (sensible) warnings
         /permissive- # Turn on strict language conformance
         /EHsc        # Specify exception handling model
      >
      $<$<CXX_COMPILER_ID:Clang>:
         -Wall     # Turn on all warnings
         -Wextra   # Turn on even more warnings
         -pedantic # Turn on strict language conformance
      >
)

add_sycl_to_target(
   TARGET
      ${PROJECT_NAME}
   SOURCES
      ${CMAKE_CURRENT_SOURCE_DIR}/Main.cpp
)