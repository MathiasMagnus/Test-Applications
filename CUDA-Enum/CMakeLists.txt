cmake_minimum_required(
  VERSION
    3.8 # CUDA language support
)

project(CUDA-Enum
  LANGUAGES
    CXX
    CUDA
)

add_executable(${PROJECT_NAME}
  CUDA-Enum.cu
)

set_target_properties(${PROJECT_NAME}
  PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
)
