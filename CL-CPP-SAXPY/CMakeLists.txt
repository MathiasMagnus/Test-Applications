cmake_minimum_required(VERSION 3.7)

project(CL-CPP-SAXPY
  LANGUAGES CXX
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  find_package(TBB CONFIG REQUIRED)
endif()

set(Sources CL-CPP-SAXPY.cpp)
set(Kernels saxpy.cl)

source_group("Sources" FILES ${Sources})
source_group("Kernels" FILES ${Files_KRNS})

add_executable(${PROJECT_NAME}
  ${Sources}
  ${Kernels}
)

target_compile_features(${PROJECT_NAME}
  PRIVATE cxx_std_17
)

target_include_directories(${PROJECT_NAME}
  PRIVATE
    "${TCLAP_INCLUDE_DIR}"
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    OpenCL::OpenCL
    Threads::Threads
    $<$<TARGET_EXISTS:TBB::tbb>:TBB::tbb>
)

target_compile_definitions(${PROJECT_NAME}
  PRIVATE
    CL_TARGET_OPENCL_VERSION=120
    CL_HPP_TARGET_OPENCL_VERSION=120
    CL_HPP_MINIMUM_OPENCL_VERSION=120
    CL_HPP_ENABLE_EXCEPTIONS
)

get_property(GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
set(TO_COPY "${CMAKE_CURRENT_SOURCE_DIR}/$<JOIN:${Kernels},;${CMAKE_CURRENT_SOURCE_DIR}/>")
set(COPY_TO "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<$<BOOL:${GENERATOR_IS_MULTI_CONFIG}>:$<CONFIG>>")
set(OUTPUT_AND_DEPENDS "${COPY_TO}/$<JOIN:${Kernels},;${COPY_TO}/>")
add_custom_command(
  OUTPUT "${OUTPUT_AND_DEPENDS}"
  COMMAND ${CMAKE_COMMAND}
    ARGS
      -E copy_if_different
      "${TO_COPY}"
      "${COPY_TO}"
  COMMENT "Copying CL kernels for ${PROJECT_NAME}"
  COMMAND_EXPAND_LISTS
  DEPENDS ${Kernels}
)
add_custom_target(${PROJECT_NAME}-device-code
  DEPENDS "${OUTPUT_AND_DEPENDS}"
)
add_dependencies(${PROJECT_NAME}
  ${PROJECT_NAME}-device-code
)