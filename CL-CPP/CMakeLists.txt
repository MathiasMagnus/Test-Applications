cmake_minimum_required(
   VERSION
      3.7
)

project(CL-CPP
   LANGUAGES
      CXX
)

set(CMAKE_MODULE_PATH
   ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
)

find_package(TCLAP REQUIRED)
find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU
   OR(CMAKE_CXX_COMPILER_ID MATCHES GNU AND
      CMAKE_SYSTEM_NAME MATCHES Linux
   )
)
   find_package(TBB CONFIG REQUIRED)
endif()

set(Files_HDRS include/CL-CPP.hpp)
set(Files_SRCS source/CL-CPP.cpp)
set(Files_KRNS kernel/saxpy.cl)

get_filename_component(Folder_KRNS
   ${Files_KRNS}
   ABSOLUTE)

configure_file(
   ${CMAKE_CURRENT_SOURCE_DIR}/include/CL-CPP-config.in.hpp
   ${CMAKE_CURRENT_BINARY_DIR}/include/CL-CPP-config.hpp)

list(APPEND
   ${Files_HDRS}
   ${CMAKE_CURRENT_BINARY_DIR}/include/CL-CPP-config.hpp
)

add_executable(${PROJECT_NAME}
   ${Files_HDRS}
   ${Files_SRCS}
   ${Files_KRNS}
)

set_target_properties(${PROJECT_NAME}
   PROPERTIES
      CXX_STANDARD 17
      CXX_STANDARD_REQUIRED ON
      CXX_EXTENSIONS OFF
)

target_include_directories(${PROJECT_NAME}
   PRIVATE
      ${PROJECT_SOURCE_DIR}/include/
      ${CMAKE_CURRENT_BINARY_DIR}/include/
)

target_link_libraries(${PROJECT_NAME}
   PRIVATE
      OpenCL::OpenCL
      Threads::Threads
      #$<$<OR:$<CXX_COMPILER_ID:GNU>,$<AND:$<CXX_COMPILER_ID:Clang>,$<PLATFORM_ID:Linux>>>:
         TBB::tbb
      #>
)

target_compile_definitions(${PROJECT_NAME}
   PRIVATE
      CL_TARGET_OPENCL_VERSION=120
)

target_compile_options(${PROJECT_NAME}
  PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:
      -Wall     # Turn on all warnings
      -Wextra   # Turn on even more warnings
      -pedantic # Turn on strict language conformance
      -Wno-ignored-attributes # Console blows up due to cl.h
    >
    $<$<CXX_COMPILER_ID:MSVC>:
      /W4          # Turn on all (sensible) warnings
      /permissive- # Turn on strict language conformance
      /EHsc        # Specify exception handling model
    >
)

source_group("Headers" FILES ${Files_HDRS})
source_group("Sources" FILES ${Files_SRCS})
source_group("Kernels" FILES ${Files_KRNS})