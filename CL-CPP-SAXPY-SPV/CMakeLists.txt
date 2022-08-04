cmake_minimum_required(VERSION 3.7)

project(CL-CPP-SAXPY-SPV
  LANGUAGES CXX
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  find_package(TBB CONFIG REQUIRED)
endif()

if(NOT CMAKE_CXX_COMPILER_ID MATCHES Clang)
  find_program(CLANG_EXE
    clang
    REQUIRED
  )
else()
  set(CLANG_EXE "${CMAKE_CXX_COMPILER}")
endif()

set(Sources CL-CPP-SAXPY-SPV.cpp)
set(Kernels saxpy.cl)

source_group("Sources" FILES ${Sources})
source_group("Kernels" FILES ${Kernels})

set(Target_Name saxpy)

add_executable(${PROJECT_NAME}
  ${Sources}
  ${Kernels}
)

target_compile_features(${PROJECT_NAME}
  PRIVATE cxx_std_17
)

target_include_directories(${PROJECT_NAME}
  PRIVATE
    "${PROJECT_SOURCE_DIR}"
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    OpenCL::OpenCL
    Threads::Threads
)

target_compile_definitions(${PROJECT_NAME}
  PRIVATE
    CL_TARGET_OPENCL_VERSION=300
#   CL_HPP_TARGET_OPENCL_VERSION=300 # opencl.hpp is buggy with this conf
    CL_HPP_TARGET_OPENCL_VERSION=210
    CL_HPP_MINIMUM_OPENCL_VERSION=210
    CL_HPP_ENABLE_EXCEPTIONS
    CL_HPP_USE_IL_KHR
)

get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
set(TO_COMPILE "${CMAKE_CURRENT_SOURCE_DIR}/$<JOIN:${Kernels},;${CMAKE_CURRENT_SOURCE_DIR}/>")
set(COMPILE_TO "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<$<BOOL:${GENERATOR_IS_MULTI_CONFIG}>:$<CONFIG>>")
set(OUTPUT_AND_DEPENDS "${COMPILE_TO}/saxpy.spv")
add_custom_command(
  OUTPUT "${OUTPUT_AND_DEPENDS}"
  COMMAND "${CLANG_EXE}"
    ARGS
      -O0
      --target=spirv64
      #-cl-std=CLC++2021
      "${TO_COMPILE}"
      -o "${OUTPUT_AND_DEPENDS}"
  COMMENT "Building SPIR-V object for ${PROJECT_NAME}"
  COMMAND_EXPAND_LISTS
  DEPENDS ${Kernels}
)
add_custom_target(${PROJECT_NAME}-device-code
  DEPENDS "${OUTPUT_AND_DEPENDS}"
)
add_dependencies(${PROJECT_NAME}
  ${PROJECT_NAME}-device-code
)