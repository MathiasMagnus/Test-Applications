cmake_minimum_required(
  VERSION
    3.7
)

project(CL-CPP
  LANGUAGES CXX
)

set(CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
)

find_package(TCLAP REQUIRED)
find_package(OpenCL REQUIRED)
find_package(Threads REQUIRED)

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
  find_package(TBB CONFIG REQUIRED)
endif()

set(Files_HDRS CL-CPP.hpp)
set(Files_SRCS CL-CPP.cpp)
set(Files_KRNS saxpy.cl)

source_group("Headers" FILES ${Files_HDRS})
source_group("Sources" FILES ${Files_SRCS})
source_group("Kernels" FILES ${Files_KRNS})

add_executable(${PROJECT_NAME}
  ${Files_HDRS}
  ${Files_SRCS}
  ${Files_KRNS}
)

target_compile_features(${PROJECT_NAME}
  PRIVATE
    cxx_std_17
)

target_include_directories(${PROJECT_NAME}
  PRIVATE
    "${PROJECT_SOURCE_DIR}"
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

# Solution 1

#add_custom_command(
#  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/$<CONFIG>/${Files_KRNS}"
#  COMMAND ${CMAKE_COMMAND}
#  ARGS -E copy_if_different
#       $<JOIN:"${CMAKE_CURRENT_SOURCE_DIR}/${Files_KRNS}"," ">
#       "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/$<CONFIG>"
#  DEPENDS ${Files_KRNS}
#  COMMENT "Copying CL files of ${PROJECT_NAME}"
#  COMMAND_EXPAND_LISTS
#)
#
#add_custom_target(${PROJECT_NAME}-device-code
#  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/$<CONFIG>/${Files_KRNS}"
#)
#
#add_dependencies(${PROJECT_NAME}
#  ${PROJECT_NAME}-device-code
#)

# Solution 2 (doesn't trigger when kernel changes)

#add_custom_command(
#  TARGET ${PROJECT_NAME}
#  POST_BUILD
#  COMMAND ${CMAKE_COMMAND}
#    -E copy_if_different
#    $<JOIN:"${CMAKE_CURRENT_SOURCE_DIR}/${Files_KRNS}"," ">
#    $<TARGET_FILE_DIR:${PROJECT_NAME}>
#  #BYPRODUCTS $<JOIN:"$<TARGET_FILE_DIR:${PROJECT_NAME}>/$<CONFIG>/${Files_KRNS}",";">
#  BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${Files_KRNS}"
#  COMMAND_EXPAND_LISTS
#)

# Solution 3

#add_custom_command(
#  OUTPUT ${Files_KRNS}
#  COMMAND ${CMAKE_COMMAND}
#  ARGS -E copy_if_different
#       $<JOIN:${Files_KRNS}," ${CMAKE_CURRENT_SOURCE_DIR}/">
#       $<TARGET_FILE_DIR:${PROJECT_NAME}>
#  DEPENDS ${Files_KRNS}
#  COMMENT "Copying CL files of ${PROJECT_NAME}"
#  COMMAND_EXPAND_LISTS
#)
#
#add_custom_target(${PROJECT_NAME}-device-code
#  DEPENDS
#    "$<JOIN:${CMAKE_CURRENT_BINARY_DIR}/${Files_KRNS},;>"
#)
#
#add_dependencies(${PROJECT_NAME}
#  ${PROJECT_NAME}-device-code
#)

# Solution 4

function (add_custom_sources)
    set(options VERBATIM)
    set(oneValueArgs TARGET OUTPUT)
    set(multiValueArgs DEPENDS COMMAND)

    # Problem
    # -------
    # CMake doesn't support generator expressions for OUTPUT directories.
    # Thus, it's not gonna work:
    # add_custom_command(
    #    OUTPUT "$<TARGET_FILE_DIR:Main>/gen/file1"
    #    DEPENDS file1
    #    COMMAND blah
    # )
    # target_sources(Main "$<TARGET_FILE_DIR:Arena>/gen/file1")
    #
    # So there will be problems with Debug/Release configurations
    # in MS VS and Qt and Qt Creator and need to work it around.
    # 
    # The section below contains the code with exactly the same logic.
    #
    # Using
    # -----
    # add_custom_sources(
    #    TARGET Main
    #    OUTPUT "$<TARGET_FILE_DIR:Main>/gen/file1"
    #    DEPENDS file1
    #    COMMAND blah
    # )
    # TARGET - is the target where your generated source code/image/3d-model/anything should be added as a dependency.
    #
    # How it works
    # ------------
    # 1. Replacing generator expressions to something unique.
    #    I map such name "$<TARGET_FILE_DIR:Arena>/gen/file1"
    #    to somewhat like: "genexpr/TARGET_FILE_DIR_Arena/gen/file1.shadow"
    # 2. make all the subdirectories to SHADOW and the original file
    # 3. add_custom_command on shadow file. I add TOUCH command to
    #    create/update shadow file as well
    # 4. target_sources(SHADOW_FILE)

    # 1. Parsing args
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    if ( ${ARG_VERBATIM} )
        set(ARG_VERBATIM VERBATIM)
    else()
        unset(ARG_VERBATIM)
    endif()

    # 2. Getting SHADOW name
    string(REGEX REPLACE "\\$<(.+)>" "\\1" _no_genex_file ${ARG_OUTPUT} )
    string(REPLACE ":" "_" _no_genex_file ${_no_genex_file} )
    set(_no_genex_file "${_no_genex_file}.shadow")
    set(_no_genex_file "${CMAKE_BINARY_DIR}/genex/${_no_genex_file}")

    # 3. Making all the dirs
    get_filename_component(_no_genex_dir ${_no_genex_file} DIRECTORY)         # _from_root = "~/my/arena-shooter/client/res"
    get_filename_component(_output_dir ${ARG_OUTPUT} DIRECTORY)

    # 4. Running the command and TOUCHing SHADOW-file
    add_custom_command(
        OUTPUT ${_no_genex_file}
        DEPENDS ${ARG_DEPENDS}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_output_dir}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${_no_genex_dir}
        COMMAND ${CMAKE_COMMAND} -E touch ${_no_genex_file}
        COMMAND ${ARG_COMMAND}
        ${ARG_VERBATIM}
    )

    # 5. Adding the dependency, so both files (the original and the shadow)
    #    will be updated when the source file is changed
    target_sources(${ARG_TARGET} PUBLIC ${_no_genex_file})
endfunction()

add_custom_sources(
  TARGET ${PROJECT_NAME}
  OUTPUT "$<TARGET_FILE_DIR:${PROJECT_NAME}>/${Files_KRNS}"
  DEPENDS ${Files_KRNS}
  COMMAND ${CMAKE_COMMAND}
    -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/${Files_KRNS}"
    $<TARGET_FILE_DIR:${PROJECT_NAME}>
)