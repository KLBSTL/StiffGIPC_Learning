cmake_minimum_required(VERSION 3.24)
project(workspace_apply LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
add_executable(workspace_apply driver.cu "${FROZEN_ROOT}/gipc/statistics.cpp")
target_include_directories(workspace_apply PRIVATE "${OVERLAY_ROOT}" "${FROZEN_ROOT}"
  "${CUDAToolkit_INCLUDE_DIRS}"
  "E:/university_class/my_includes/vcpkg/installed/x64-windows/include"
  "E:/university_class/my_includes/vcpkg/installed/x64-windows/include/eigen3"
  "E:/Anaconda/Library/include")
target_compile_definitions(workspace_apply PRIVATE USE_SNK1 ADAPTIVE_KAPPA USE_FRICTION
  USE_QUADRATIC_BENDING CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING)
target_link_libraries(workspace_apply PRIVATE CUDA::cublas)
target_compile_features(workspace_apply PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(workspace_apply PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RUNTIME_LIBRARY Static)
target_compile_options(workspace_apply PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:-D_WINDOWS;--use_fast_math;--extended-lambda;--expt-relaxed-constexpr;--default-stream=per-thread;-lineinfo;--ptxas-options=-allow-expensive-optimizations=true;-Xcompiler=/GR;-Xcompiler=/EHsc;-Xcompiler=/Zc:preprocessor>")
target_link_options(workspace_apply PRIVATE "/NODEFAULTLIB:LIBCMT")
