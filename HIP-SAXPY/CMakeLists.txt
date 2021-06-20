cmake_minimum_required(
  VERSION 3.8 # cxx_std_17
)

project(HIP-SAXPY
  LANGUAGES CXX
)

option(USE_HIP_CPU "Prefer HIP-CPU runtime instead of HW acceleration" OFF)

set(AMDGPU_TARGETS "gfx803" CACHE STRING "List of specific machine types for library to target")

if(USE_HIP_CPU)
  find_package(hip_cpu_rt REQUIRED)
else()
  find_package(hip REQUIRED CONFIG)
endif()

add_executable(${PROJECT_NAME} Main.cpp)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    $<$<BOOL:${USE_HIP_CPU}>:hip_cpu_rt::hip_cpu_rt>
    $<$<NOT:$<BOOL:${USE_HIP_CPU}>>:hip::device>
)

target_compile_features(${PROJECT_NAME}
  PRIVATE cxx_std_17
)