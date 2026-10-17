cmake_minimum_required(VERSION 3.24)
project(full_apply_graph LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
add_executable(full_apply_graph driver.cu "${FROZEN_ROOT}/gipc/statistics.cpp")
target_include_directories(full_apply_graph PRIVATE "${OVERLAY_ROOT}" "${FROZEN_ROOT}"
  "${CUDAToolkit_INCLUDE_DIRS}" "${CUDAToolkit_INCLUDE_DIRS}/cccl"
  "E:/university_class/my_includes/vcpkg/installed/x64-windows/include"
  "E:/university_class/my_includes/vcpkg/installed/x64-windows/include/eigen3"
  "E:/Anaconda/Library/include")
target_compile_definitions(full_apply_graph PRIVATE USE_SNK1 ADAPTIVE_KAPPA USE_FRICTION
  USE_QUADRATIC_BENDING CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING)
target_link_libraries(full_apply_graph PRIVATE CUDA::cublas)
target_compile_features(full_apply_graph PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(full_apply_graph PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RUNTIME_LIBRARY Static)
target_compile_options(full_apply_graph PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-D_WINDOWS;--use_fast_math;--extended-lambda;--expt-relaxed-constexpr;--default-stream=per-thread;-lineinfo;--ptxas-options=-allow-expensive-optimizations=true;-Xcompiler=/GR;-Xcompiler=/EHsc;-Xcompiler=/Zc:preprocessor>")
target_link_options(full_apply_graph PRIVATE "/NODEFAULTLIB:LIBCMT")
