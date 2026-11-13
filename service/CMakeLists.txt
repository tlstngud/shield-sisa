# SHIELD-AV service core: build/service/libshield_av.so + demo, tests and benchmarks.
#   cmake -S service -B build/service -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build/service
# Links the engine libraries built by ../build.sh:
#   build/audio/libshield.so (SHIELD-A audio engine, engine/audio; header engine/audio/include/shield.h)
#   build/prep/libshield_v_prep.so, build/cpu/libshield_v_cpu.so, build/cuda/libshield_v_cuda.so (engine/include)
cmake_minimum_required(VERSION 3.16)
project(shield_av CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
get_filename_component(WS ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)
set(BUILD_ROOT ${WS}/build)

add_library(shield_audio SHARED IMPORTED)
set_target_properties(shield_audio PROPERTIES IMPORTED_LOCATION ${BUILD_ROOT}/audio/libshield.so)
add_library(shv_prep SHARED IMPORTED)
set_target_properties(shv_prep PROPERTIES IMPORTED_LOCATION ${BUILD_ROOT}/prep/libshield_v_prep.so)
add_library(shv_cpu SHARED IMPORTED)
set_target_properties(shv_cpu PROPERTIES IMPORTED_LOCATION ${BUILD_ROOT}/cpu/libshield_v_cpu.so)
add_library(shv_cuda SHARED IMPORTED)
set_target_properties(shv_cuda PROPERTIES IMPORTED_LOCATION ${BUILD_ROOT}/cuda/libshield_v_cuda.so)

add_library(shield_av SHARED src/shield_av.cpp)
target_include_directories(shield_av PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                           PRIVATE ${WS}/engine/audio/include ${WS}/engine/include)
target_compile_options(shield_av PRIVATE -fvisibility=hidden -Wall -Wextra -pthread)
target_link_libraries(shield_av PRIVATE shield_audio shv_prep shv_cpu shv_cuda pthread)
set_target_properties(shield_av PROPERTIES
  BUILD_RPATH "$ORIGIN/../audio:$ORIGIN/../prep:$ORIGIN/../cpu:$ORIGIN/../cuda"
  INSTALL_RPATH "$ORIGIN/../audio:$ORIGIN/../prep:$ORIGIN/../cpu:$ORIGIN/../cuda"
  SKIP_BUILD_RPATH OFF BUILD_WITH_INSTALL_RPATH ON)

function(shav_tool name src)
  if(NOT EXISTS ${src})
    return()
  endif()
  add_executable(${name} ${src})
  target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${WS}/engine/include
                             ${WS}/engine/audio/include ${WS}/bench ${WS}/tools)
  target_compile_options(${name} PRIVATE -Wall -Wextra -pthread)
  target_link_libraries(${name} PRIVATE shield_av shv_prep shield_audio pthread z ${ARGN})
  set_target_properties(${name} PROPERTIES
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../audio:$ORIGIN/../prep:$ORIGIN/../cpu:$ORIGIN/../cuda")
endfunction()

shav_tool(demo_av ${WS}/tools/demo_av.cpp)
shav_tool(service_video_parity ${WS}/bench/service_video_parity.cpp z)
shav_tool(service_contracts ${WS}/bench/service_contracts.cpp z)
shav_tool(service_load ${WS}/bench/service_load.cpp z)
shav_tool(service_audio_parity ${WS}/bench/service_audio_parity.cpp z)
# edge / lifetime suite (edge-case, lifetime and regression tests)
shav_tool(verify_service_edge ${WS}/bench/verify_service_edge.cpp z)
