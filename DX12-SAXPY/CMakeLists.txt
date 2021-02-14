cmake_minimum_required(
  VERSION 3.16 # target_precompile_headers
)

project(DX12-SAXPY
  LANGUAGES CXX
)

set(CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
)

find_package(TCLAP REQUIRED)

add_executable(${PROJECT_NAME}
  Options.hpp
  Options.cpp
  DX12.hpp
  Main.cpp
)

set_target_properties(${PROJECT_NAME}
  PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${TCLAP_INCLUDE_PATH}
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    d3d12
    dxgi
    dxguid
    WindowsApp
)

target_precompile_headers(${PROJECT_NAME}
  PRIVATE
    # Windows includes
    [[<dxgi.h>]]
    # DirectX 12 includes
    [[<d3d12.h>]]
    # C++/WinRT includes
    [[<winrt/Windows.Graphics.h>]]
    # Standard C++ includes
    [[<iostream>]]
    [[<vector>]]
    [[<algorithm>]]
    [[<iterator>]]
)

target_compile_options(${PROJECT_NAME}
  PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:
      -Wall     # Turn on all warnings
      -Wextra   # Turn on even more warnings
      -pedantic # Turn on strict language conformance
    >
    $<$<CXX_COMPILER_ID:MSVC>:
      /W4          # Turn on all (sensible) warnings
      /permissive- # Turn on strict language conformance
      /EHsc        # Specify exception handling model
    >
)