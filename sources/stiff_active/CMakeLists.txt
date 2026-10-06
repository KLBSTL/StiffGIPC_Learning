cmake_minimum_required(VERSION 3.22)
project(StiffGIPCActive LANGUAGES CXX CUDA)

# The historical project supplies dependencies and all existing compiler flags.
# Its build tree is entirely separate from every frozen historical build.
get_filename_component(GIPC_WORKSPACE "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(GIPC_BASE "${GIPC_WORKSPACE}/sources/stiff_perf_v50")
set(GIPC_V54 "${GIPC_WORKSPACE}/sources/stiff_perf_v54/StiffGIPC")
set(GIPC_ACTIVE "${CMAKE_CURRENT_SOURCE_DIR}/StiffGIPC")
add_subdirectory("${GIPC_BASE}" "${CMAKE_CURRENT_BINARY_DIR}/inherited-v50")

get_target_property(_original_sources gipc SOURCES)
set(_effective_sources)
set(_logical_names)
set(_source_map "logical_path\tsource_path\tlayer\tobject_relative_path\n")
foreach(_source IN LISTS _original_sources)
    file(RELATIVE_PATH _logical "${GIPC_BASE}/StiffGIPC" "${_source}")
    if(_logical MATCHES "^\\.\\.")
        message(FATAL_ERROR "Unexpected inherited gipc source: ${_source}")
    endif()
    set(_layer "v50")
    if(_logical STREQUAL "solver/toi_solver.cu")
        set(_source "${GIPC_V54}/${_logical}")
        set(_layer "v54")
    endif()
    if(EXISTS "${GIPC_ACTIVE}/${_logical}")
        set(_source "${GIPC_ACTIVE}/${_logical}")
        set(_layer "active")
    endif()
    list(APPEND _logical_names "${_logical}")
    list(APPEND _effective_sources "${_source}")
endforeach()
file(GLOB_RECURSE _active_sources CONFIGURE_DEPENDS
    "${GIPC_ACTIVE}/*.cu" "${GIPC_ACTIVE}/*.cpp"
    "${GIPC_ACTIVE}/*.h" "${GIPC_ACTIVE}/*.cuh"
    "${GIPC_ACTIVE}/*.hpp" "${GIPC_ACTIVE}/*.inl" "${GIPC_ACTIVE}/*.inc")
foreach(_source IN LISTS _active_sources)
    file(RELATIVE_PATH _logical "${GIPC_ACTIVE}" "${_source}")
    if(NOT _logical IN_LIST _logical_names)
        list(APPEND _logical_names "${_logical}")
        list(APPEND _effective_sources "${_source}")
    endif()
endforeach()
set_property(TARGET gipc PROPERTY SOURCES "${_effective_sources}")
target_include_directories(gipc BEFORE PRIVATE "${GIPC_ACTIVE}" "${GIPC_V54}")
set_target_properties(gipc PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

# Source identity is part of each object's name. Different overlay providers
# cannot silently inherit the same basename object from an earlier provider.
foreach(_source IN LISTS _effective_sources)
    if(_source MATCHES "/stiff_active/StiffGIPC/")
        set(_layer "active")
        file(RELATIVE_PATH _logical "${GIPC_ACTIVE}" "${_source}")
    elseif(_source MATCHES "/stiff_perf_v54/StiffGIPC/")
        set(_layer "v54")
        file(RELATIVE_PATH _logical "${GIPC_V54}" "${_source}")
    else()
        set(_layer "v50")
        file(RELATIVE_PATH _logical "${GIPC_BASE}/StiffGIPC" "${_source}")
    endif()
    set(_object "")
    if(_source MATCHES "\\.(cu|cpp)$")
        # Preserve quoted sibling includes without copying inherited headers.
        # New active sibling headers win, then the corresponding frozen folder.
        if(_source MATCHES "\\.cu$")
            get_filename_component(_logical_dir "${_logical}" DIRECTORY)
            set_property(SOURCE "${_source}" TARGET_DIRECTORY gipc PROPERTY
                INCLUDE_DIRECTORIES "${GIPC_ACTIVE}/${_logical_dir};${GIPC_V54}/${_logical_dir};${GIPC_BASE}/StiffGIPC/${_logical_dir}")
        endif()
        string(SHA256 _source_key "${_source}")
        string(SUBSTRING "${_source_key}" 0 16 _source_key)
        get_filename_component(_stem "${_source}" NAME_WE)
        set(_object "inherited-v50/gipc.dir/$<CONFIG>/${_stem}_${_source_key}.obj")
        if(_source MATCHES "\\.cu$")
            set_property(SOURCE "${_source}" TARGET_DIRECTORY gipc PROPERTY
                VS_SETTINGS "CompileOut=$(IntDir)${_stem}_${_source_key}.obj")
        else()
            set_property(SOURCE "${_source}" TARGET_DIRECTORY gipc PROPERTY
                VS_SETTINGS "ObjectFileName=$(IntDir)${_stem}_${_source_key}.obj")
        endif()
    endif()
    string(APPEND _source_map "${_logical}\t${_source}\t${_layer}\t${_object}\n")
endforeach()
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/effective_sources_$<CONFIG>.tsv"
    CONTENT "${_source_map}")

# Keep root-level tests discoverable without changing the historical CMake file.
enable_testing()
add_executable(ipc_budget_test tests/ipc_budget_test.cpp)
target_compile_features(ipc_budget_test PRIVATE cxx_std_17)
add_test(NAME ipc_budget_invariants COMMAND ipc_budget_test)
find_package(nlohmann_json REQUIRED)
add_executable(ipc_residual_controller_test tests/ipc_residual_controller_test.cpp)
target_compile_features(ipc_residual_controller_test PRIVATE cxx_std_17)
target_include_directories(ipc_residual_controller_test PRIVATE "${GIPC_ACTIVE}" "${GIPC_BASE}/StiffGIPC")
target_link_libraries(ipc_residual_controller_test PRIVATE nlohmann_json::nlohmann_json)
add_test(NAME ipc_compensated_controller COMMAND ipc_residual_controller_test)
add_test(NAME active_pcg_guard_boundaries COMMAND ${CMAKE_COMMAND} -E env
    "GIPC_PCG_GUARD_FIXTURE=${CMAKE_CURRENT_BINARY_DIR}/pcg_guard_fixture.json" $<TARGET_FILE:gipc>)
