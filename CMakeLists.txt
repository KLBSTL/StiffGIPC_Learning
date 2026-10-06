cmake_minimum_required(VERSION 3.18)    
# support relative path, cuda_std_17, archive manipulation

# project meta
# ============

set(CMAKE_BUILD_TYPE "Release")
if(UNIX)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug")
    endif()
    set(CMAKE_CXX_FLAGS "-O3")
endif()

project(StiffGIPC LANGUAGES CXX CUDA)

# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find packages
enable_language(CUDA)

# Platform detection: MSVC vs GNU/Clang
if(MSVC)
    # Keep only the Windows define; host options (/GR /EHsc) are added below via
    # target_compile_options to avoid CMake merging them into a preprocessor define.
    set(CMAKE_CUDA_FLAGS "-D_WINDOWS")
else()
    set(CMAKE_CUDA_FLAGS "")
endif()
find_package(Eigen3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(CUDAToolkit REQUIRED)

# add external libraries
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()


add_executable(gipc)
target_compile_features(gipc PRIVATE cuda_std_17)
target_include_directories(gipc PRIVATE "StiffGIPC/")
file(GLOB_RECURSE GIPC_SOURCE CONFIGURE_DEPENDS 
    "StiffGIPC/*.cu" "StiffGIPC/*.cpp")
file(GLOB_RECURSE GIPC_HANDER CONFIGURE_DEPENDS 
    "StiffGIPC/*.h" "StiffGIPC/*.cuh" "StiffGIPC/*.hpp" "StiffGIPC/*.inl" "StiffGIPC/*.inc")
target_sources(gipc PRIVATE ${GIPC_SOURCE} ${GIPC_HANDER})

# Windows filenames are case insensitive: PCG_SOLVER.cu and pcg_solver.cu
# must not share the CUDA MSBuild default object name.
if(MSVC)
    foreach(_source IN LISTS GIPC_SOURCE)
        string(SHA256 _source_key "${_source}")
        string(SUBSTRING "${_source_key}" 0 16 _source_key)
        get_filename_component(_stem "${_source}" NAME_WE)
        if(_source MATCHES "\\.cu$")
            set_property(SOURCE "${_source}" TARGET_DIRECTORY gipc PROPERTY
                VS_SETTINGS "CompileOut=$(IntDir)${_stem}_${_source_key}.obj")
        else()
            set_property(SOURCE "${_source}" TARGET_DIRECTORY gipc PROPERTY
                VS_SETTINGS "ObjectFileName=$(IntDir)${_stem}_${_source_key}.obj")
        endif()
    endforeach()
endif()

# add assets and output directory, that we can easily access them in the code
target_compile_definitions(gipc PRIVATE GIPC_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Assets/")
target_compile_definitions(gipc PRIVATE GIPC_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Output/")

target_compile_definitions(gipc PRIVATE 
USE_SNK1 
ADAPTIVE_KAPPA 
USE_FRICTION 
USE_QUADRATIC_BENDING
CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING
)


find_package(CUDAToolkit REQUIRED)

# link cuda libraries
#include cuda directories
message(STATUS "CUDA library includes: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

target_include_directories(gipc PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13)
    target_include_directories(gipc PRIVATE ${CUDAToolkit_INCLUDE_DIRS}/cccl)
endif()

target_link_libraries(gipc PRIVATE cusparse cublas cusolver cudart_static)
target_link_libraries(gipc PRIVATE metis_partition CUDA::cudart)
target_link_libraries(gipc PRIVATE GLUT::GLUT GLEW::GLEW OpenGL::GLU Eigen3::Eigen)
target_link_libraries(gipc PRIVATE nlohmann_json::nlohmann_json)

if(MSVC)
    # Some external C dependencies still pull in the static CRT; gipc uses the
    # dynamic CRT, so drop the static one to avoid LNK4098 conflicts.
    target_link_options(gipc PRIVATE "/NODEFAULTLIB:LIBCMT")
endif()

target_compile_features(gipc PRIVATE cxx_std_17)

set_target_properties(gipc PROPERTIES
    CUDA_EXTENSIONS ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

target_compile_options(gipc
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        ${CMAKE_CUDA_FLAGS}
        --use_fast_math
        -lineinfo
        --ptxas-options=-allow-expensive-optimizations=true
        -Xcudafe
        "--display_error_number --diag_suppress=implicit_return_from_non_void_function --diag_suppress=2642"
    >
    PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        --default-stream=per-thread
        $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler=/GR>
        $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler=/EHsc>
        $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler=/Zc:preprocessor>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=-fPIC>
    >
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${GIPC_SOURCE} ${GIPC_HANDER})

add_subdirectory("MeshProcess")

add_executable(diag_fused_update_tests tests/diag_fused_update_tests.cu)
target_compile_features(diag_fused_update_tests PRIVATE cuda_std_17)
target_include_directories(diag_fused_update_tests PRIVATE StiffGIPC/)
target_link_libraries(diag_fused_update_tests PRIVATE Eigen3::Eigen CUDA::cudart)
# Project headers use <eigen3/Eigen/...>, so also expose Eigen's parent.
get_target_property(_eigen_includes Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
foreach(_eigen_include IN LISTS _eigen_includes)
    get_filename_component(_eigen_parent "${_eigen_include}" DIRECTORY)
    target_include_directories(diag_fused_update_tests PRIVATE "${_eigen_parent}")
endforeach()
target_compile_options(diag_fused_update_tests PRIVATE
    --use_fast_math --default-stream=per-thread --expt-relaxed-constexpr
    $<$<CXX_COMPILER_ID:MSVC>:-Xcompiler=/Zc:preprocessor>)
set_target_properties(diag_fused_update_tests PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
enable_testing()
add_test(NAME diag_fused_update_equivalence COMMAND diag_fused_update_tests)
add_test(NAME pcg_guard_boundaries COMMAND ${CMAKE_COMMAND} -E env
    "GIPC_PCG_GUARD_FIXTURE=${CMAKE_CURRENT_BINARY_DIR}/pcg_guard_fixture.json" $<TARGET_FILE:gipc>)

# Small invariant tests accompany the active IPC termination module. The
# official baseline is built separately from baseline/CMakeLists.txt.
add_executable(ipc_budget_test tests/ipc_budget_test.cpp)
target_compile_features(ipc_budget_test PRIVATE cxx_std_17)
add_test(NAME ipc_budget_invariants COMMAND ipc_budget_test)
add_executable(ipc_residual_controller_test tests/ipc_residual_controller_test.cpp)
target_compile_features(ipc_residual_controller_test PRIVATE cxx_std_17)
target_include_directories(ipc_residual_controller_test PRIVATE StiffGIPC/)
target_link_libraries(ipc_residual_controller_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME ipc_compensated_controller COMMAND ipc_residual_controller_test)
add_test(NAME contact_pool_equivalence COMMAND ${CMAKE_COMMAND} -E env
    "GIPC_CONTACT_POOL=1" "GIPC_CONTACT_POOL_VALIDATE=1"
    "GIPC_CONTACT_POOL_FIXTURE=${CMAKE_CURRENT_BINARY_DIR}/contact_pool_fixture.json" $<TARGET_FILE:gipc>)

# Ensure external C libraries use the same dynamic runtime as the executable
# to avoid LIBCMT/MSVCRT conflicts (MSVC only).
if(MSVC)
    foreach(_tgt GKlib metis metis_partition)
        if(TARGET ${_tgt})
            set_property(TARGET ${_tgt} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        endif()
    endforeach()
endif()
