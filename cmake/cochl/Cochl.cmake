# Cochl-specific build integration.

include_directories("${CMAKE_SOURCE_DIR}")

tvm_file_glob(GLOB_RECURSE COCHL_COMPILER_SRCS
  cochl/src/tir/*.cc
  cochl/src/tir/transform/*.cc
)

list(APPEND COMPILER_SRCS ${COCHL_COMPILER_SRCS})
