cmake_minimum_required(VERSION 3.24)
project(weighted_check_contract LANGUAGES CXX)
add_executable(weighted_check_schedule_test weighted_check_schedule_test.cpp)
target_compile_features(weighted_check_schedule_test PRIVATE cxx_std_17)
target_include_directories(weighted_check_schedule_test PRIVATE "${FROZEN_ROOT}"
  "E:/university_class/my_includes/vcpkg/installed/x64-windows/include/eigen3")
enable_testing()
add_test(NAME weighted_check_schedule COMMAND weighted_check_schedule_test)
