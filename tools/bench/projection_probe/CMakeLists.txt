cmake_minimum_required(VERSION 3.24)
project(projection_probe LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
if(NOT EXISTS "${FROZEN_NATIVE_DIR}/native_balanced_gpu.cuh")
  message(FATAL_ERROR "Verified frozen native header required")
endif()
add_executable(projection_probe probe.cu)
target_include_directories(projection_probe PRIVATE "${CUDAToolkit_INCLUDE_DIRS}" "${FROZEN_NATIVE_DIR}"
  "E:/university_class/my_includes/vcpkg/installed/x64-windows/include/eigen3"
  "E:/Anaconda/Library/include")
target_link_libraries(projection_probe PRIVATE CUDA::cublas)
target_compile_features(projection_probe PRIVATE cxx_std_17 cuda_std_17)
set_target_properties(projection_probe PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RUNTIME_LIBRARY Static)
target_compile_options(projection_probe PRIVATE
  "$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;--expt-relaxed-constexpr;--extended-lambda;--default-stream=per-thread;-lineinfo;--ptxas-options=-v;-Xcompiler=/EHsc;-Xcompiler=/Zc:preprocessor>")
