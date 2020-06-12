cmake_minimum_required(VERSION 3.13)
# Make sure to include CUDA, so that GVDB header files are processed correctly,
# and C, so that GLEW compiles correctly:
project(
  mrfmap
  VERSION 0.0.1
  DESCRIPTION
    "MRFMap: Online Probabilistic 3D Mapping using Forward Ray Sensor Models library"
  LANGUAGES CUDA CXX C)

# Check if this CMakeLists is being used directly or via add_subdirectory
set(MRFMAP_MASTER_PROJECT OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(MRFMAP_MASTER_PROJECT ON)
  set(default_build_type "Release")
  if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(
      STATUS
        "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE
        "${default_build_type}"
        CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
  endif()
endif()
# Allow for parent CMakeLists to populate the outgoing targets file
if(NOT MRFMAP_EXPORT_NAME)
  set(MRFMAP_EXPORT_NAME "${PROJECT_NAME}-targets")
endif()
message(STATUS "[MRFMap]:: Export name is ${MRFMAP_EXPORT_NAME}")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake")

# We are using FetchContent over git submodules and Add_ExternalProject as the
# most modern CMake variant of adding third party CMake libs.
include(FetchContent)

# EIGEN Compile and build Eigen if specific Eigen version is not provided
# set(EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}" CACHE FILEPATH "Path to
# Eigen 3.3.7")

find_package(Eigen3 3.3.7 QUIET)
if(NOT EIGEN3-FOUND)
  message(STATUS "[MRFMap]:: Cloning and configuring Eigen 3.3.7")
  FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.gz
    URL_HASH MD5=9e30f67e8531477de4117506fe44669b)
  if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
    set(EIGEN3_INCLUDE_DIR ${eigen_SOURCE_DIR})
    find_package(Eigen3 3.3.7 REQUIRED)
  endif()
endif()
message(
  STATUS "[MRFMap]:: Eigen3 include directory used is ${EIGEN3_INCLUDE_DIR}")

# OCTOMAP
message(STATUS "[MRFMap]:: Cloning and configuring octomap.")
FetchContent_Declare(
  octomap
  GIT_REPOSITORY https://github.com/icoderaven/octomap.git
  GIT_TAG master)
if(NOT octomap_POPULATED)
  FetchContent_Populate(octomap)
  option(BUILD_OCTOVIS_SUBPROJECT "" OFF)
  option(OCTOVIS_QT5 "" OFF)
  option(OCTOMAP_BUILD_ONLY_LIB "" ON)
  option(OCTOMAP_BUILD_STATIC "" OFF)
  set(OCTOMAP_EXPORT_NAME "${MRFMAP_EXPORT_NAME}")
  add_subdirectory(${octomap_SOURCE_DIR} ${octomap_BINARY_DIR})
endif()

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)

# CODETIMER
message(STATUS "[MRFMap]:: Cloning and configuring codetimer.")
FetchContent_Declare(
  codetimer
  GIT_REPOSITORY https://github.com/icoderaven/codetimer.git
  GIT_TAG master)
if(NOT codetimer_POPULATED)
  FetchContent_Populate(codetimer)
  set(CODETIMER_EXPORT_NAME "${MRFMAP_EXPORT_NAME}")
  add_subdirectory(${codetimer_SOURCE_DIR} ${codetimer_BINARY_DIR})
endif()

# PYBIND11 Add option to build python bindings
option(MRFMAP_BUILD_PYTHON_BINDINGS "Build Python bindings for MRFMap" ON)
if(MRFMAP_BUILD_PYTHON_BINDINGS)
  option(MRFMAP_USE_PYTHON2 "Use python2 for bindings" ON)
  # find_package(PythonInterp 2.7) set(PythonLibsNew_FIND_VERSION 2.7)
  # find_package(Python2 COMPONENTS Interpreter REQUIRED) set(PYTHON_INTERPRETER
  # "/usr/lib/python2.7") set(PYTHON_LIBRARY "/usr/lib/python2.7")
  # set(PYTHON_INCLUDE_DIR "/usr/include/python2.7") Include pybind11
  message(STATUS "[MRFMap]:: Cloning and configuring pybind11.")
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG master)
  if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    # Set pybind11 targets name
    set(PYBIND11_EXPORT_NAME "${MRFMAP_EXPORT_NAME}")
    option(PYBIND11_INSTALL "" ON)
    if(MRFMAP_USE_PYTHON2)
      find_package(PythonInterp 2.7)
    endif()
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
  endif()

endif()

# set(CUDA_SEPARABLE_COMPILATION TRUE)
find_package(CUDA)
message(STATUS "[MRFMap]:: Cloning and configuring gvdb-voxels.")
FetchContent_Declare(
  gvdb
  GIT_REPOSITORY https://github.com/icoderaven/gvdb-voxels.git
  GIT_TAG feature/upstream_refactor_merge)
if(NOT gvdb_POPULATED)
  FetchContent_Populate(gvdb)
  # Adding the custom build options in our fork
  option(GVDB_USE_DEPTH_IMAGE_BUFFER "" ON)
  option(GVDB_SKIP_DEPTH_BUFFER_TEST "" ON)
  option(GVDB_BUILD_SAMPLES "" OFF)
  set(GVDB_EXPORT_NAME "${MRFMAP_EXPORT_NAME}")
  add_subdirectory(${gvdb_SOURCE_DIR} ${gvdb_BINARY_DIR})
endif()

# PANGOLIN Add option for Pangolin visualisation, if enabled
option(MRFMAP_BUILD_PANGOLIN_VISUALISER
       "Build the integrated Pangolin based Visualiser" ON)
if(MRFMAP_BUILD_PANGOLIN_VISUALISER)
  message(STATUS "[MRFMap]:: Cloning and configuring pangolin.")
  FetchContent_Declare(
    pangolin
    GIT_REPOSITORY https://github.com/icoderaven/pangolin.git
    GIT_TAG master)
  if(NOT pangolin_POPULATED)
    FetchContent_Populate(pangolin)
    option(BUILD_TESTS "" OFF)
    option(BUILD_TOOLS "" OFF)
    option(BUILD_EXAMPLES "" OFF)
    option(BUILD_PANGOLIN_PYTHON "" OFF)
    option(BUILD_PANGOLIN_TOON "" OFF)
    option(BUILD_PYPANGOLIN_MODULE "" OFF)
    set(PANGOLIN_EXPORT_NAME "${MRFMAP_EXPORT_NAME}")
    add_subdirectory(${pangolin_SOURCE_DIR} ${pangolin_BINARY_DIR})
  endif()

  # At this point, we should have pangolin targets
endif()

# -march=native required because  otherwise Eigen gets compiled with different
# alignment options when using some other libraries... (cough, GTSAM) TODO: Make
# this cleaner using target_compile_options()
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -ffast-math  -fopenmp -O3 -fPIC -march=native")
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math  -fopenmp -O3 -fPIC -march=native")
set(CMAKE_CXX_FLAGS_DEBUG
    "${CMAKE_CXX_FLAGS_DEBUG}  -fopenmp -fPIC -march=native -g3")
message(STATUS "[MRFMap]:: Building in ${CMAKE_BUILD_TYPE} mode!")

if(CMAKE_BUILD_TYPE MATCHES Debug)
  message(STATUS "[MRFMap]:: Debug build.")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
  message(STATUS "[MRFMap]:: Release build.")
else()
  message(STATUS "[MRFMap]:: Some other build type.")
endif()

# Combine common system include directories that we end up using
set(COMMON_INCLUDE_DIRS $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>
                        ${YAML_CPP_INCLUDEDIR} ${OPENCV_INCLUDE_DIRS})

# message( "CUDA ROOT DIR IS ${CUDA_TOOLKIT_ROOT_DIR}, adding the sample common
cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto")
message(STATUS "[MRFMap]:: ARCH_FLAGS are ${ARCH_FLAGS}")
# Don't use gcc params
set(CUDA_PROPAGATE_HOST_FLAGS False)
# Set our NVCC flags
if(CMAKE_BUILD_TYPE MATCHES Debug)
  set(EXTRA_COMPILE_OPTIONS -G -g -lineinfo --restrict --expt-relaxed-constexpr)
else()
  set(EXTRA_COMPILE_OPTIONS --use_fast_math -lineinfo --restrict
                            --expt-relaxed-constexpr)
endif()
set(CUDA_VERBOSE_BUILD ON)

message(STATUS "[MRFMap]:: current source is ${CMAKE_CURRENT_SOURCE_DIR}")
set(KERNELS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/)
file(
  GLOB KERNELS_SOURCE_FILES
  RELATIVE ${KERNELS_SOURCE_DIR}
  src/cuda/*.cu)
message(STATUS "[MRFMap]:: kernel files found are ${KERNELS_SOURCE_FILES}")

# PTX files
set(PROJECT_NAME_PTX "GVDBKERNELS_PTX")
# Compile the source files into a PTX object library.
add_library(${PROJECT_NAME_PTX} OBJECT)
# add alias to match exported target when calling using add_subdirectory()
add_library(mrfmap::${PROJECT_NAME_PTX} ALIAS ${PROJECT_NAME_PTX})
target_sources(${PROJECT_NAME_PTX} PRIVATE src/cuda/gvdbkernels.cu)
list(APPEND DEFINED_TARGETS ${PROJECT_NAME_PTX})

# Make sure it compiles to PTX files.
set_target_properties(${PROJECT_NAME_PTX} PROPERTIES CUDA_PTX_COMPILATION ON)
# Enable the --use_fast_math CUDA compiler flag.
target_compile_options(${PROJECT_NAME_PTX} PRIVATE ${ARCH_FLAGS}
                                                   ${EXTRA_COMPILE_OPTIONS})
# Include gvdbPTX's include directories:
target_include_directories(
  ${PROJECT_NAME_PTX}
  PRIVATE include $<TARGET_PROPERTY:gvdbPTX,INTERFACE_INCLUDE_DIRECTORIES>
  # It also needs to include GVDB's include directories because it uses
  # gvdb_vec.h:
  PRIVATE $<TARGET_PROPERTY:gvdb,INTERFACE_INCLUDE_DIRECTORIES>)
# Copy the PTX files to the executable directory (this is where the dependencies
# on the PTX files are introduced):
add_custom_command(
  OUTPUT copiedPTX.stamp POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_INSTALL_PREFIX}/lib/
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_OBJECTS:gvdbPTX>
          $<TARGET_FILE_DIR:mrfmap>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_OBJECTS:gvdbPTX>
          ${CMAKE_INSTALL_PREFIX}/lib/
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_OBJECTS:${PROJECT_NAME_PTX}> $<TARGET_FILE_DIR:mrfmap>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_OBJECTS:${PROJECT_NAME_PTX}> ${CMAKE_INSTALL_PREFIX}/lib/
  COMMAND ${CMAKE_COMMAND} -E touch copiedPTX.stamp
  DEPENDS ${PROJECT_NAME_PTX} mrfmap
  COMMAND_EXPAND_LISTS)
add_custom_target(mrfmap_copy_ptx ALL DEPENDS copiedPTX.stamp)

add_library(
  mrfmap SHARED
  src/cuda/gvdbhelper.cu
  src/GVDBImage.cpp
  src/GVDBCamera.cpp
  src/GVDBInference.cpp
  src/GVDBMRFMap.cpp
  src/GVDBMapLikelihoodEstimator.cpp
  src/KeyframeSelector.cpp)
# add alias to match exported target when calling using add_subdirectory()
add_library(mrfmap::mrfmap ALIAS mrfmap)
target_include_directories(mrfmap
                           PRIVATE ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)
target_include_directories(mrfmap SYSTEM PUBLIC ${COMMON_INCLUDE_DIRS})

target_include_directories(
  mrfmap PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>)
target_link_libraries(
  mrfmap
  PUBLIC codetimer::codetimer
         gvdb::gvdb
         ${CUDA_CUDART_LIBRARY}
         ${CUDA_CUDA_LIBRARY}
         ${OpenCV_LIBRARIES}
         ${CUDA_cudadevrt_LIBRARY}
         ${YAML_CPP_LIBRARIES})
add_dependencies(mrfmap gvdbPTX)
foreach(_FILE ${KERNELS_SOURCE_FILES})
  string(REPLACE ".cu" ".ptx" _FILE_PTX ${_FILE})
  string(TOUPPER ${_FILE_PTX} _FILE_UPPER)
  string(REPLACE "." "_" _MACRO ${_FILE_UPPER})
  target_compile_definitions(
    mrfmap
    PUBLIC
      $<BUILD_INTERFACE:${_MACRO}="$<TARGET_FILE_DIR:mrfmap>/${_FILE_PTX}">
      $<INSTALL_INTERFACE:${_MACRO}=${CMAKE_INSTALL_PREFIX}/lib/${_FILE_PTX}>)
endforeach(_FILE)

# Make sure that we dont strip rpath during install since libgvdb isn't
# installed to libspace
set_target_properties(mrfmap PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
list(APPEND DEFINED_TARGETS mrfmap)

if(MRFMAP_BUILD_PANGOLIN_VISUALISER)
  add_library(pango_viewer src/Viewer.cpp src/GVDBOctomapWrapper.cpp)
  # add alias to match exported target when calling using add_subdirectory()
  add_library(mrfmap::pango_viewer ALIAS pango_viewer)
  target_link_libraries(
    pango_viewer PUBLIC pangolin octomap::octomap codetimer::codetimer
                        mrfmap::mrfmap gvdb::gvdb)
  target_include_directories(pango_viewer PRIVATE ${OCTOMAP_INCLUDE_DIRS})
  target_compile_definitions(
    pango_viewer PRIVATE SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/")
  set_target_properties(pango_viewer PROPERTIES INSTALL_RPATH_USE_LINK_PATH
                                                TRUE)
  list(APPEND DEFINED_TARGETS pango_viewer)

  add_executable(testMRFMapViewer src/testMRFMapViewer.cpp)
  target_link_libraries(testMRFMapViewer mrfmap::mrfmap mrfmap::pango_viewer)
  set_target_properties(testMRFMapViewer PROPERTIES INSTALL_RPATH_USE_LINK_PATH
                                                    TRUE)
  # Provide string paths to the test executable
  target_compile_definitions(
    testMRFMapViewer
    PRIVATE
      CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/coffin_world_640.yaml"
      DATA_PATH="${CMAKE_CURRENT_SOURCE_DIR}/data/")
  list(APPEND DEFINED_TARGETS testMRFMapViewer)
endif()

# Finally, add a python wrapper
if(MRFMAP_BUILD_PYTHON_BINDINGS)
  add_library(GVDBPyModules SHARED src/GVDBPyModules.cpp
                                   src/GVDBOctomapWrapper.cpp)
  target_link_libraries(
    GVDBPyModules octomap::octomap mrfmap::mrfmap pybind11::pybind11
    mrfmap::pango_viewer codetimer::codetimer)
  if(MRFMAP_BUILD_PANGOLIN_VISUALISER)
    target_link_libraries(GVDBPyModules pangolin)
  endif()
  set_target_properties(
    GVDBPyModules PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                             SUFFIX "${PYTHON_MODULE_EXTENSION}")
  set_target_properties(GVDBPyModules PROPERTIES INSTALL_RPATH_USE_LINK_PATH
                                                 TRUE)
  list(APPEND DEFINED_TARGETS GVDBPyModules)
endif()

install(DIRECTORY include/ DESTINATION include)

# cmake packaging
set(MRFMAP_CMAKECONFIG_INSTALL_DIR
    "share/cmake/${PROJECT_NAME}"
    CACHE STRING "install path for mrfmap-config.cmake")

write_basic_package_version_file(
  "mrfmap-config-version.cmake"
  VERSION ${PACKAGE_VERSION}
  COMPATIBILITY AnyNewerVersion)

# Generate targets file
install(
  TARGETS mrfmap pango_viewer GVDBPyModules
  EXPORT "${MRFMAP_EXPORT_NAME}"
  LIBRARY DESTINATION lib)

# Install the targets only if we're the Master project
if(MRFMAP_MASTER_PROJECT)
  install(
    EXPORT "${MRFMAP_EXPORT_NAME}"
    NAMESPACE "${PROJECT_NAME}::"
    DESTINATION ${MRFMAP_CMAKECONFIG_INSTALL_DIR})

  # This isn't sufficient, since dependencies also need our dependencies
  # (libcuckoo) This is provided in cmake/mrfmap-config.cmmake.
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mrfmap-config.cmake
                ${CMAKE_CURRENT_BINARY_DIR}/mrfmap-config-version.cmake
          DESTINATION ${MRFMAP_CMAKECONFIG_INSTALL_DIR})
endif()
