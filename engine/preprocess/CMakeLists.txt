# SHIELD-V native preprocessing: libshield_v_prep.so + tools.
#   cmake -S engine/preprocess -B build/prep -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build/prep
cmake_minimum_required(VERSION 3.16)
project(shield_v_prep LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(Threads REQUIRED)

set(SHV_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(shv_bin2c bin2c.cpp)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/cascade_data.cpp
  COMMAND shv_bin2c ${CMAKE_CURRENT_SOURCE_DIR}/data/haarcascade_frontalface_default.xml
          ${CMAKE_CURRENT_BINARY_DIR}/cascade_data.cpp shv_embedded_cascade
  DEPENDS shv_bin2c ${CMAKE_CURRENT_SOURCE_DIR}/data/haarcascade_frontalface_default.xml)

add_library(shield_v_prep SHARED
  cpu.cpp pillow.cpp cvresize.cpp color.cpp haar.cpp haar_simd.cpp geometry.cpp canvas.cpp window.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/cascade_data.cpp)
target_include_directories(shield_v_prep PUBLIC ${SHV_ROOT}/engine/include)
# Portable baseline (x86-64); SIMD kernels opt in with function target attributes + runtime dispatch.
# -ffp-contract=off: no FMA contraction anywhere (the references round every float operation separately).
target_compile_options(shield_v_prep PRIVATE -O3 -ffp-contract=off -fno-fast-math -Wall -Wextra
                       -Wno-unused-parameter -fvisibility=hidden)
target_compile_definitions(shield_v_prep PRIVATE "SHV_API=__attribute__((visibility(\"default\")))")
set_target_properties(shield_v_prep PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(shield_v_prep PRIVATE Threads::Threads)

add_executable(prep_cli ${SHV_ROOT}/tools/prep_cli.cpp)
target_compile_options(prep_cli PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(prep_cli PRIVATE shield_v_prep Threads::Threads)

add_executable(prep_bench ${SHV_ROOT}/tools/prep_bench.cpp)
target_compile_options(prep_bench PRIVATE -O2 -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(prep_bench PRIVATE shield_v_prep Threads::Threads)

