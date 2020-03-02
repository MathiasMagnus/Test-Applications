cmake_minimum_required(
   VERSION
      3.8 # CUDA language support
)

project(HIP-SAXPY
   LANGUAGES
      CXX
)

set(CMAKE_MODULE_PATH
   ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
)

find_package(HIP REQUIRED)
find_package(TCLAP REQUIRED)

set(HOST_SOURCES
   ${CMAKE_CURRENT_SOURCE_DIR}/../CUDA-SAXPY/Main.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/../CUDA-SAXPY/CUDA.cpp
   ${CMAKE_CURRENT_SOURCE_DIR}/../CUDA-SAXPY/Options.cpp
)
set(DEV_SOURCES
   ${CMAKE_CURRENT_SOURCE_DIR}/../CUDA-SAXPY/Kernel.cu
)

set_source_files_properties(${DEV_SOURCES}
   PROPERTIES
      INCLUDE_DIRECTORIES
         ${CMAKE_CURRENT_LIST_DIR}/../CUDA-SAXPY
         ${HIP_INCLUDE_DIRECTORIES}
      COMPILE_OPTIONS
         -std=c++14
)

hip_add_executable(${PROJECT_NAME}
   ${HOST_SOURCES}
   ${DEV_SOURCES}
)

set_target_properties(${PROJECT_NAME}
   PROPERTIES
      CXX_STANDARD 14
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
)

target_include_directories(${PROJECT_NAME}
   PRIVATE
      ${CMAKE_CURRENT_LIST_DIR}/../CUDA-SAXPY
      ${HIP_INCLUDE_DIRECTORIES}
      ${TCLAP_INCLUDE_PATH}
)

target_compile_options(${PROJECT_NAME}
   PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:
         $<$<CXX_COMPILER_ID:MSVC>:
            /W4          # Turn on all (sensible) warnings
            /permissive- # Turn on strict language conformance
            /EHsc        # Specify exception handling model
         >
         $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:
            -Wall     # Turn on all warnings
            -Wextra   # Turn on even more warnings
            -pedantic # Turn on strict language conformance
         >
      >
)