cmake_minimum_required(VERSION 3.24)
project(weighted_gpu_source_snapshot LANGUAGES CXX CUDA)

# Keep project/ and overlay/ byte-identical to the compiled private experiment.
get_filename_component(REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
set(FROZEN_ROOT "${REPOSITORY_ROOT}/StiffGIPC" CACHE PATH "Native source used by the private replay")
set(OVERLAY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/overlay" CACHE PATH "Archived experiment overlay")
add_subdirectory(project)
