# SHIELD-V CUDA engine: build/cuda/libshield_v_cuda.so (+ experimental variant and bench tools).
#   ../../build.sh (CUDA 12+/13 nvcc, CUDA_ARCH default 86, Release)
cmake_minimum_required(VERSION 3.24)
project(shield_v_cuda LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

set(SHV_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(NUMERIC_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/common.cuh ${CMAKE_CURRENT_SOURCE_DIR}/gemm.cuh ${CMAKE_CURRENT_SOURCE_DIR}/gemm.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/kernels.cuh ${CMAKE_CURRENT_SOURCE_DIR}/cnn.cu ${CMAKE_CURRENT_SOURCE_DIR}/fused.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/seq.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/engine.hpp ${CMAKE_CURRENT_SOURCE_DIR}/engine.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/model.hpp ${CMAKE_CURRENT_SOURCE_DIR}/model.cpp ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt)
set(REV_HDR ${CMAKE_CURRENT_BINARY_DIR}/gen/shv_cuda_revision.h)
add_custom_command(OUTPUT ${REV_HDR}
  COMMAND ${CMAKE_COMMAND} "-DOUT=${REV_HDR}" "-DSRCS=${NUMERIC_SRCS}" -P ${CMAKE_CURRENT_SOURCE_DIR}/gen_revision.cmake
  DEPENDS ${NUMERIC_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/gen_revision.cmake
  COMMENT "compute revision" VERBATIM)
add_custom_target(shv_cuda_revision DEPENDS ${REV_HDR})

# No fast math anywhere. -fmad=false: separate multiply/add expressions round like the reference (explicit fmaf()
# in the GEMM / convolution / recurrence inner loops still fuse).
set(SHV_CUDA_FLAGS -O3 -fmad=false --extended-lambda -Xptxas=-O3 -lineinfo)
add_library(shv_cuda_core OBJECT gemm.cu cnn.cu fused.cu seq.cu engine.cu model.cpp)
target_include_directories(shv_cuda_core PUBLIC ${SHV_ROOT}/engine/include ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(shv_cuda_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${SHV_CUDA_FLAGS} -Xcompiler=-fvisibility=hidden>
  $<$<COMPILE_LANGUAGE:CXX>:-O3 -ffp-contract=off -fno-fast-math -Wall -Wextra -fvisibility=hidden>)

foreach(variant shield_v_cuda shield_v_cuda_experimental)
  add_library(${variant} SHARED api.cu $<TARGET_OBJECTS:shv_cuda_core>)
  add_dependencies(${variant} shv_cuda_revision)
  target_include_directories(${variant} PUBLIC ${SHV_ROOT}/engine/include PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                             ${CMAKE_CURRENT_BINARY_DIR}/gen)
  target_compile_options(${variant} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${SHV_CUDA_FLAGS} -Xcompiler=-fvisibility=hidden>)
  target_link_libraries(${variant} PRIVATE Threads::Threads)
  set_target_properties(${variant} PROPERTIES CUDA_VISIBILITY_PRESET hidden CXX_VISIBILITY_PRESET hidden)
endforeach()
target_compile_definitions(shield_v_cuda_experimental PRIVATE SHV_CUDA_EXPERIMENTAL=1)

# ---------------------------------------------------------------- bench / verification tools (bench/gpu_*)
# gpu_dump: internal engine, writes intermediates of one sample for the oracle fixture comparison.
add_executable(gpu_dump ${SHV_ROOT}/bench/gpu_dump.cu $<TARGET_OBJECTS:shv_cuda_core>)
target_include_directories(gpu_dump PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${SHV_ROOT}/engine/include ${SHV_ROOT}/bench)
target_compile_options(gpu_dump PRIVATE ${SHV_CUDA_FLAGS})
target_link_libraries(gpu_dump PRIVATE ZLIB::ZLIB Threads::Threads)

# gpu_cli: logits for the 840 reference crops through the public API (experimental lib: uncertified modes allowed).
add_executable(gpu_cli ${SHV_ROOT}/bench/gpu_cli.cpp)
target_include_directories(gpu_cli PRIVATE ${SHV_ROOT}/engine/include ${SHV_ROOT}/bench)
target_compile_options(gpu_cli PRIVATE -O2 -Wall -Wextra)
target_link_libraries(gpu_cli PRIVATE shield_v_cuda_experimental ZLIB::ZLIB Threads::Threads)
set_target_properties(gpu_cli PROPERTIES BUILD_RPATH ${CMAKE_CURRENT_BINARY_DIR})

# gpu_bench: latency / throughput / live-update benchmarks + API tests through the public API (experimental lib so
# that a build can be measured before its certificates exist; certificate enforcement is tested by gpu_certcheck).
add_executable(gpu_bench ${SHV_ROOT}/bench/gpu_bench.cpp)
target_include_directories(gpu_bench PRIVATE ${SHV_ROOT}/engine/include ${SHV_ROOT}/bench)
target_compile_options(gpu_bench PRIVATE -O2 -Wall -Wextra)
target_link_libraries(gpu_bench PRIVATE shield_v_cuda_experimental ZLIB::ZLIB Threads::Threads)
set_target_properties(gpu_bench PROPERTIES BUILD_RPATH ${CMAKE_CURRENT_BINARY_DIR})

# gpu_kbench: model-only timings of the internal engine (GPU-resident inputs, CUDA events).
add_executable(gpu_kbench ${SHV_ROOT}/bench/gpu_kbench.cu $<TARGET_OBJECTS:shv_cuda_core>)
target_include_directories(gpu_kbench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${SHV_ROOT}/engine/include ${SHV_ROOT}/bench)
target_compile_options(gpu_kbench PRIVATE ${SHV_CUDA_FLAGS})
target_link_libraries(gpu_kbench PRIVATE ZLIB::ZLIB Threads::Threads)

# gpu_gemm_bench: engine GEMMs vs cuBLAS on the model's shapes (cuBLAS used only here, for reference).
find_package(CUDAToolkit REQUIRED)
add_executable(gpu_gemm_bench ${SHV_ROOT}/bench/gpu_gemm_bench.cu ${CMAKE_CURRENT_SOURCE_DIR}/gemm.cu)
target_include_directories(gpu_gemm_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(gpu_gemm_bench PRIVATE ${SHV_CUDA_FLAGS})
target_link_libraries(gpu_gemm_bench PRIVATE CUDA::cublas)

# gpu_certcheck: certificate enforcement of the shipped (strict) library.
add_executable(gpu_certcheck ${SHV_ROOT}/bench/gpu_certcheck.cpp)
target_include_directories(gpu_certcheck PRIVATE ${SHV_ROOT}/engine/include)
target_link_libraries(gpu_certcheck PRIVATE shield_v_cuda)
set_target_properties(gpu_certcheck PROPERTIES BUILD_RPATH ${CMAKE_CURRENT_BINARY_DIR})
