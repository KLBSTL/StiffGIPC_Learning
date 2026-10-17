cmake_minimum_required(VERSION 3.24)
project(cache_pcg LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
add_executable(cache_pcg driver.cu "${FROZEN_ROOT}/gipc/statistics.cpp")
target_include_directories(cache_pcg PRIVATE "${OVERLAY_ROOT}" "${CMAKE_CURRENT_SOURCE_DIR}" "${FROZEN_ROOT}" "${CUDAToolkit_INCLUDE_DIRS}"
  "E:/university_class/my_includes/vcpkg/installed/x64-windows/include"
  "E:/university_class/my_includes/vcpkg/installed/x64-windows/include/eigen3"
  "E:/Anaconda/Library/include")
target_compile_definitions(cache_pcg PRIVATE USE_SNK1 ADAPTIVE_KAPPA USE_FRICTION USE_QUADRATIC_BENDING CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING)
target_link_libraries(cache_pcg PRIVATE CUDA::cublas)
target_compile_features(cache_pcg PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(cache_pcg PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RUNTIME_LIBRARY Static)
target_compile_options(cache_pcg PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-D_WINDOWS;--use_fast_math;--extended-lambda;--expt-relaxed-constexpr;--default-stream=per-thread;-lineinfo;--ptxas-options=-allow-expensive-optimizations=true;-Xcompiler=/GR;-Xcompiler=/EHsc;-Xcompiler=/Zc:preprocessor>")
target_link_options(cache_pcg PRIVATE "/NODEFAULTLIB:LIBCMT")
