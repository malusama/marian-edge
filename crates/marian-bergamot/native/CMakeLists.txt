cmake_minimum_required(VERSION 3.16)
project(marian_mlx_bergamot_worker LANGUAGES C CXX)

if(NOT BERGAMOT_SOURCE)
  message(FATAL_ERROR "BERGAMOT_SOURCE must point to browsermt/bergamot-translator")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(COMPILE_CUDA OFF CACHE BOOL "" FORCE)
set(USE_MKL OFF CACHE BOOL "" FORCE)
set(USE_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(SSPLIT_COMPILE_LIBRARY_ONLY ON CACHE BOOL "" FORCE)
set(SSPLIT_USE_INTERNAL_PCRE2 ON CACHE BOOL "" FORCE)
set(COMPILE_TESTS OFF CACHE BOOL "" FORCE)
set(COMPILE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(COMPILE_PYTHON OFF CACHE BOOL "" FORCE)
set(GIT_SUBMODULE OFF CACHE BOOL "" FORCE)

# GCC 12 reports false uninitialized-value warnings from the AVX512 intrinsic
# wrappers when they are instantiated by Marian. Keep warnings-as-errors for
# everything else, matching the dependency's own GCC 12 workaround.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12
   AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
  add_compile_options(-Wno-uninitialized -Wno-maybe-uninitialized)
endif()

add_subdirectory("${BERGAMOT_SOURCE}" bergamot EXCLUDE_FROM_ALL)

add_executable(marian-mlx-bergamot-worker worker.cpp)
target_link_libraries(marian-mlx-bergamot-worker PRIVATE bergamot-translator)
install(TARGETS marian-mlx-bergamot-worker RUNTIME DESTINATION bin)
