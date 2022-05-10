cmake_minimum_required(VERSION 3.16)
project(moldyn)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O2 -mavx2 -mavx -mfma")
add_executable(moldyn  moldyn.c)
target_link_libraries(moldyn m)

set(CMAKE_C_FLAGS "-O2 -mavx2 -mavx -mfma")
add_executable(moldyn_avx  moldyn_avx.c)
target_link_libraries(moldyn_avx m)

set(CMAKE_C_FLAGS "-O2 -mavx2 -mavx -mfma")
add_executable(moldyn_test  moldyn_test.c)
target_link_libraries(moldyn_test m)

set(CMAKE_C_FLAGS "-O2 -mavx2 -mavx -mfma")
add_executable(moldyn_test1  moldyn_test1.c)
target_link_libraries(moldyn_test1 m)

set(CMAKE_C_FLAGS "-O2 -mavx2 -mavx -mfma")
add_executable(moldyn_test2  moldyn_test2.c)
target_link_libraries(moldyn_test2 m)
