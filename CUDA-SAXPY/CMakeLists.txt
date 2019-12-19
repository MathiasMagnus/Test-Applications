cmake_minimum_required(
   VERSION
      3.8 # CUDA language support
)

project(CUDA-SAXPY
   LANGUAGES
      CXX
      CUDA
)

set(CMAKE_MODULE_PATH
   ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
)

find_package(TCLAP REQUIRED)

add_executable(${PROJECT_NAME}
   Main.cpp
   CUDA.cpp
   Options.cpp
   Kernel.cu
)

set_target_properties(${PROJECT_NAME}
   PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_STANDARD 14
      CUDA_STANDARD_REQUIRED ON
      CUDA_EXTENSIONS OFF
      CXX_STANDARD 14
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
)

target_include_directories(${PROJECT_NAME}
   PRIVATE
      ${CMAKE_CURRENT_LIST_DIR}
      ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
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