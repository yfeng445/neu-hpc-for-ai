cmake_minimum_required(VERSION 3.27)
project(dsmoe_bridge LANGUAGES CXX CUDA)

# 安静一点
set(CMAKE_RULE_MESSAGES OFF)
set(CMAKE_MESSAGE_LOG_LEVEL WARNING)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 80)

# 仅编译桥接 + demo
add_library(dsmoe_bridge STATIC
  ${CMAKE_SOURCE_DIR}/moe_fwd.cpp
)

target_include_directories(dsmoe_bridge PUBLIC
  ${CMAKE_SOURCE_DIR}                        # for "moe_fwd.h"
  ${CMAKE_SOURCE_DIR}/../flashmoe/csrc/include
)

# CUDA Runtime
find_package(CUDAToolkit REQUIRED)
target_link_libraries(dsmoe_bridge PUBLIC CUDA::cudart)
# 注意：不要强依赖 CUDA::cudadevrt（本项目不需要）
# if(TARGET CUDA::cudadevrt)
#   target_link_libraries(dsmoe_bridge PUBLIC CUDA::cudadevrt)
# endif()

# NVSHMEM：先静默查找，失败则显式指定路径后再 Required
find_package(NVSHMEM CONFIG QUIET)
if(NOT NVSHMEM_FOUND)
  if(NOT DEFINED NVSHMEM_DIR)
    set(NVSHMEM_DIR "/opt/nvshmem/lib/cmake/nvshmem")
  endif()
  find_package(NVSHMEM CONFIG REQUIRED)
endif()

target_link_libraries(dsmoe_bridge PUBLIC NVSHMEM::nvshmem NVSHMEM::nvshmem_host)

# RPATH（运行时无需额外导出 LD_LIBRARY_PATH）
set_target_properties(dsmoe_bridge PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  BUILD_RPATH   "/opt/nvidia-mathdx/lib;/opt/nvshmem/lib"
  INSTALL_RPATH "/opt/nvidia-mathdx/lib;/opt/nvshmem/lib"
)

# Demo 可执行文件
add_executable(dsmoe_demo ${CMAKE_SOURCE_DIR}/main_moe.cpp)
target_link_libraries(dsmoe_demo PRIVATE dsmoe_bridge)
set_target_properties(dsmoe_demo PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "/workspace/out/bin"
  BUILD_RPATH   "/opt/nvidia-mathdx/lib;/opt/nvshmem/lib"
  INSTALL_RPATH "/opt/nvidia-mathdx/lib;/opt/nvshmem/lib"
)
