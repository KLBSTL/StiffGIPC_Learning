# Build-only adapter. Every native TU and solver/material byte stays frozen.
cmake_minimum_required(VERSION 3.19)
project(StiffM0FrozenBaseline LANGUAGES CXX CUDA)
get_filename_component(M0_FROZEN_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../baseline" ABSOLUTE)
add_subdirectory("${M0_FROZEN_ROOT}" frozen_baseline)
get_target_property(M0_SOURCES gipc SOURCES)
foreach(M0_SOURCE IN LISTS M0_SOURCES)
    string(SHA256 M0_KEY "${M0_SOURCE}")
    string(SUBSTRING "${M0_KEY}" 0 16 M0_KEY)
    get_filename_component(M0_STEM "${M0_SOURCE}" NAME_WE)
    if(M0_SOURCE MATCHES "\\.cu$")
        set_property(SOURCE "${M0_SOURCE}" TARGET_DIRECTORY gipc PROPERTY
            VS_SETTINGS "CompileOut=$(IntDir)${M0_STEM}_${M0_KEY}.obj")
    elseif(M0_SOURCE MATCHES "\\.cpp$")
        set_property(SOURCE "${M0_SOURCE}" TARGET_DIRECTORY gipc PROPERTY
            VS_SETTINGS "ObjectFileName=$(IntDir)${M0_STEM}_${M0_KEY}.obj")
    endif()
endforeach()
