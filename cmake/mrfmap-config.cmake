#
# mrfmap-config.cmake
#

include(CMakeFindDependencyMacro)

# If this file hasn't been called before
if(NOT TARGET mrfmap::mrfmap)
  include("${CMAKE_CURRENT_LIST_DIR}/mrfmap-targets.cmake")
endif()
