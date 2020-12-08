cmake_minimum_required(
  VERSION 3.8 # cxx_std_17
)

project(HIP-SAXPY
  LANGUAGES CXX
)

find_package(hip_cpu_rt REQUIRED)

add_executable(${PROJECT_NAME}
  Main.cpp
)

target_link_libraries(${PROJECT_NAME}
  hip_cpu_rt::hip_cpu_rt
)

target_compile_features(${PROJECT_NAME}
  PRIVATE cxx_std_17
)