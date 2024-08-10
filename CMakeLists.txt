cmake_minimum_required(VERSION 3.5)
project(micropp CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-pedantic-errors -Wall -Wextra -Werror -Wshadow -g3 -fsanitize=address -fsanitize=undefined")

add_executable(micropp src/main.cpp src/value.cpp)
add_executable(micro_test tests/doctest_main.cpp tests/test_value.cpp src/value.cpp)

target_include_directories(micro_test PRIVATE src)