cmake_minimum_required(VERSION 3.24)
project(weighted_gpu LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
add_executable(weighted_gpu driver.cu "${FROZEN_ROOT}/gipc/statistics.cpp")
target_include_directories(weighted_gpu PRIVATE "${OVERLAY_ROOT}" "${CMAKE_CURRENT_SOURCE_DIR}" "${FROZEN_ROOT}" "${CUDAToolkit_INCLUDE_DIRS}"
  "E:/university_class/my_includes/vcpkg/installed/x64-windows/include"
  "E:/university_class/my_includes/vcpkg/installed/x64-windows/include/eigen3"
  "E:/Anaconda/Library/include")
target_compile_definitions(weighted_gpu PRIVATE USE_SNK1 ADAPTIVE_KAPPA USE_FRICTION USE_QUADRATIC_BENDING CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING)
target_link_libraries(weighted_gpu PRIVATE CUDA::cublas)
target_compile_features(weighted_gpu PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(weighted_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RUNTIME_LIBRARY Static)
target_compile_options(weighted_gpu PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-D_WINDOWS;--use_fast_math;--extended-lambda;--expt-relaxed-constexpr;--default-stream=per-thread;-lineinfo;--ptxas-options=-allow-expensive-optimizations=true;-Xcompiler=/GR;-Xcompiler=/EHsc;-Xcompiler=/Zc:preprocessor>")
target_link_options(weighted_gpu PRIVATE "/NODEFAULTLIB:LIBCMT")
