cmake_minimum_required(VERSION 3.20)
project(stiff_toi_validator LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
set(TI ../../references/tight_inclusion)
add_executable(validate_path validate_path.cpp
    ${TI}/tight_inclusion/inclusion_ccd.cpp
    ${TI}/tight_inclusion/interval_root_finder.cpp
    ${TI}/tight_inclusion/avx.cpp)
target_include_directories(validate_path PRIVATE ${TI})
target_link_libraries(validate_path PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)
target_compile_definitions(validate_path PRIVATE TIGHT_INCLUSION_DOUBLE NOMINMAX)
add_executable(diagnose_first_path diagnose_first_path.cpp
    ${TI}/tight_inclusion/inclusion_ccd.cpp
    ${TI}/tight_inclusion/interval_root_finder.cpp
    ${TI}/tight_inclusion/avx.cpp)
target_include_directories(diagnose_first_path PRIVATE ${TI})
target_link_libraries(diagnose_first_path PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json)
target_compile_definitions(diagnose_first_path PRIVATE TIGHT_INCLUSION_DOUBLE NOMINMAX)
if(MSVC)
    target_compile_options(validate_path PRIVATE /FI${CMAKE_CURRENT_SOURCE_DIR}/compat.h)
    target_compile_options(diagnose_first_path PRIVATE /FI${CMAKE_CURRENT_SOURCE_DIR}/compat.h)
else()
    target_compile_options(validate_path PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/compat.h)
    target_compile_options(diagnose_first_path PRIVATE -include ${CMAKE_CURRENT_SOURCE_DIR}/compat.h)
endif()
