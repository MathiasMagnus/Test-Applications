option(USE_HIP_CPU
  "Prefer HIP-CPU runtime instead of HW acceleration"
  ON
)

if(USE_HIP_CPU)
  cmake_minimum_required(
    VERSION 3.8 # cxx_std_17
  )
  project(HIP-SAXPY
    LANGUAGES CXX
  )
  find_package(hip_cpu_rt
    REQUIRED
  )
  find_package(Threads
    REQUIRED
  )
  find_package(TBB
    COMPONENTS tbb
    REQUIRED
  )
  add_executable(${PROJECT_NAME}
    Main.hip
  )
  set_source_files_properties(Main.hip
    PROPERTIES
      LANGUAGE CXX
  )
  target_link_libraries(${PROJECT_NAME}
    PRIVATE
      hip_cpu_rt::hip_cpu_rt
  )
else()
  cmake_minimum_required(
    VERSION 3.21 # HIP language support
  )
  project(HIP-SAXPY
    LANGUAGES HIP
  )
  add_executable(${PROJECT_NAME}
    Main.hip
  )
  target_compile_features(${PROJECT_NAME}
    PRIVATE hip_std_14
  )
endif()