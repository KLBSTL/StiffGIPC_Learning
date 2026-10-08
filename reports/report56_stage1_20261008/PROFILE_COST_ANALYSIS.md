# Stage-1 Nsight 单帧成本分析

仅覆盖 sphere 第49帧与 fixed 第40帧的 capture。GPU 时间为裁剪到唯一 completed physical-frame NVTX 窗口后的区间并集；CPU NVTX 是包含子范围的包络。两者、等待 API 和 Graph 包络均不能相加。结果不构成整场收益或 5% 晋级门槛证据。

| Capture | CPU frame ms | GPU union 含 whole Graph ms | whole Graph ms | Graph nodes ms | 原规则 unknown ms |
|---|---:|---:|---:|---:|---:|
| sphere / f49 | 268.789 | 168.713 | 0.000 | 31.179 | 56.437 |
| fixed / f40 | 165.181 | 115.633 | 37.044 | 0.000 | 31.264 |

两份 resolved 配置均为 `conditional_graph` / chunk=1、legacy MAS，FullCCD refit、batched energy、energy reuse、discrete BVH refit 均关闭。因此 `swept_bvh_build_or_refit` 是范围名字，本次实际配置走 build。sphere 采用 node trace，fixed 采用 whole-Graph trace：0 表示该表没有记录该粒度，不能表示 Graph 或其算子没有执行。

| 符号类别：GPU union ms | sphere f49 | fixed f40 |
|---|---:|---:|
| collision.discrete_query | 50.908665 | 19.515101 |
| collision.swept_query | 7.240881 | 7.362674 |
| collision.safety_edge_triangle | 20.939380 | 8.450385 |
| collision.accd_self_narrow | 4.194935 | 2.165564 |
| collision.bvh | 3.049146 | 2.760252 |
| energy.producer | 2.918901 | 1.727677 |
| matrix.convert | 3.235067 | 2.814140 |
| mas.numeric_scatter | 1.448669 | 2.183711 |
| mas.numeric_reduce | 0.314847 | 0.316192 |
| mas.inverse_prepare | 7.630674 | 6.885586 |
| spmv | 8.342157 | Graph 内部不可见 |

| 实际 kernel：每格为 launch次数 / GPU union ms | sphere f49 | fixed f40 |
|---|---:|---:|
| `_selfQuery_vf` | 22 / 23.089074 | 7 / 5.574421 |
| `_selfQuery_ee` | 22 / 27.775719 | 7 / 13.915752 |
| `_GroundCollisionDetect` | 22 / 0.043872 | 7 / 0.024928 |
| `_selfQuery_vf_ccd` | 14 / 2.425020 | 7 / 1.694300 |
| `_selfQuery_ee_ccd` | 14 / 4.815861 | 7 / 5.668374 |
| `_edgeTriIntersectionQuery` | 22 / 20.939380 | 7 / 8.450385 |
| `_cub_reduct_self_step` | 28 / 4.194935 | 12 / 2.165564 |

`collision.bvh` 仅含原符号规则识别的建树核函数，未包含被规则留作 unknown 的 radix/scan；下表由唯一 runtime correlation 的 BVH owner 给出包括库核函数的完整 launch 集合。两种口径有包含关系。

| 包含库核函数的 GPU launch 集合 / 归约 | sphere f49 ms | fixed f40 ms |
|---|---:|---:|
| bvh.discrete | 4.794292 | 2.578076 |
| bvh.swept | 3.095451 | 2.650620 |
| matrix.convert_including_library_kernels | 4.807225 | 4.447097 |
| mas.prepare | 10.110188 | 9.827280 |
| energy.all_owned_kernels | 3.367572 | 1.907068 |
| energy.final_reduce_owned | 0.448671 | 0.179391 |
| energy.owned_gpu_copies | 0.274752 | 0.104512 |

| CPU NVTX 包络（包含子范围，不可相加） | sphere f49 ms | fixed f40 ms |
|---|---:|---:|
| collision.discrete_bvh_build | 14.880397 | 6.738048 |
| collision.discrete_query | 54.959948 | 20.932665 |
| collision.swept_bvh_build_or_refit | 9.080272 | 5.900532 |
| collision.swept_query | 8.712266 | 9.373344 |
| collision.self_ccd | 6.073499 | 2.892191 |
| ipc.line_search | 113.386599 | 49.533075 |
| ipc.energy_evaluation | 17.349664 | 9.846705 |
| ipc.energy_scalar_reduce_and_readback | 11.189917 | 6.230604 |
| linear.matrix_convert | 11.127744 | 6.897861 |
| mas.prepare | 7.391366 | 7.120698 |
| mas.hierarchy | 6.789256 | 6.736245 |
| graph.entry | 66.383442 | 51.390930 |
| graph.capture_and_instantiate | 2.567544 | 1.877336 |
| graph.replay | 0.557938 | 0.519367 |
| graph.initial_readback | 8.354261 | 9.221917 |
| graph.final_readback | 52.300469 | 37.908468 |

能量 scalar/readback 的 CPU 包络远大于其 GPU final-reduce kernel 并集。这些同步读取会等待前面的 GPU 工作；差值不能全归为 final sum、PCIe 传输或可消除的 CPU 成本。`energy.final_reduce_owned` 包含 FEM scalar owner 和 ABD evaluation owner 下的 CUB reduce，排除了 Graph/CCD/PCG 的其他 CUB reduce；GPU copy 单列，仍不与上述 CPU 包络相加。


## sphere f49：归因细节

FullCCD query 生成 swept AABB 候选；真正 self ACCD 在 `_cub_reduct_self_step` 内。DCD query 已包含距离分类。edge-triangle 是线搜索安全查询，保留为独立类别。

| outside-Graph BVH launch owner | GPU union ms |
|---|---:|
| collision.discrete_bvh_build | 4.794292 |
| collision.swept_bvh_build_or_refit | 3.095451 |

| 唯一 runtime + native NVTX 的 energy owner | GPU union ms |
|---|---:|
| ipc.energy_production | 2.786677 |
| ipc.energy_scalar_reduce_and_readback | 0.320448 |
| ipc.energy_evaluation | 0.260447 |

| Graph 内部：仅依实际符号分类 | GPU union ms |
|---|---:|
| abd.local_action | 3.650713 |
| mas.local_action | 2.400639 |
| mas.prolong | 1.368896 |
| mas.restrict | 2.238973 |
| memory.graph_internal_kernel | 2.696057 |
| pcg.control_and_validation | 3.200505 |
| pcg.dot_partial | 2.505984 |
| pcg.vector_and_condition | 1.284798 |
| pcg.vector_update | 1.100383 |
| reduce.cub_unspecified_operand | 2.390365 |
| spmv | 8.342157 |

Graph nodes 为11736次活动、652次 restrict 算子活动；节点首尾跨度跨越该帧多次 replay 与外部求解/碰撞，不能当成单一 Graph replay 用时。上述 CUB 只按算法识别，未从 CPU `pcg.reduce_device` 或 `mas` 范围归因内部节点。

| 原 unknown 符号（最高15项） | 补充类别 | GPU union ms |
|---|---|---:|
| `_edgeTriIntersectionQuery` | collision.safety_edge_triangle | 20.939380 |
| `DeviceRadixSortOnesweepKernel<…>` | library.cub_radix_sort_unspecified_operand | 6.885521 |
| `gipc::<unnamed>::cal_abd_system_preconditioner_kernel2` | abd.preconditioner_prepare | 4.564694 |
| `_cub_reduct_self_step` | collision.accd_self_narrow | 4.194935 |
| `_calculate_triangle_fem_gradient_hessian` | fem.assembly | 3.693175 |
| `_calFrictionHessian` | contact.friction_assembly | 2.447803 |
| `gipc::write_barrier_hessian` | abd.contact_assembly | 2.411354 |
| `memset32` | memory.graph_internal_kernel | 1.906236 |
| `_get_triangleFEMEnergy_Reduction_3D` | energy.producer | 1.278236 |
| `fast_segmental_reduce_ptr_kernel<…>` | library.segmented_matrix_reduce_unspecified_owner | 0.964639 |
| `_calculate_quad_bending_gradient_hessian` | fem.assembly | 0.955293 |
| `memcpy32_post` | memory.graph_internal_kernel | 0.789821 |
| `_getBarrierEnergy_Reduction_3D` | energy.producer | 0.639423 |
| `DeviceRadixSortHistogramKernel<…>` | library.cub_radix_sort_unspecified_operand | 0.550816 |
| `_getFrictionEnergy_Reduction_3D` | energy.producer | 0.387647 |

| 查询/ACCD kernel | 资源字段的实际取值 |
|---|---|
| `_cub_reduct_self_step` | `[{"registersPerThread": 178, "staticSharedMemory": 80, "dynamicSharedMemory": 0, "localMemoryPerThread": 0, "localMemoryTotal": 138936320, "sharedMemoryExecuted": 8192, "sharedMemoryLimitConfig": 0}]` |
| `_selfQuery_vf_ccd` | `[{"registersPerThread": 40, "staticSharedMemory": 0, "dynamicSharedMemory": 0, "localMemoryPerThread": 0, "localMemoryTotal": 85196800, "sharedMemoryExecuted": 8192, "sharedMemoryLimitConfig": 0}]` |
| `_selfQuery_ee_ccd` | `[{"registersPerThread": 40, "staticSharedMemory": 0, "dynamicSharedMemory": 0, "localMemoryPerThread": 0, "localMemoryTotal": 85196800, "sharedMemoryExecuted": 8192, "sharedMemoryLimitConfig": 0}]` |
| `_edgeTriIntersectionQuery` | `[{"registersPerThread": 58, "staticSharedMemory": 0, "dynamicSharedMemory": 0, "localMemoryPerThread": 0, "localMemoryTotal": 117964800, "sharedMemoryExecuted": 8192, "sharedMemoryLimitConfig": 0}]` |
| `_selfQuery_vf` | `[{"registersPerThread": 162, "staticSharedMemory": 0, "dynamicSharedMemory": 0, "localMemoryPerThread": 0, "localMemoryTotal": 111411200, "sharedMemoryExecuted": 8192, "sharedMemoryLimitConfig": 0}]` |
| `_selfQuery_ee` | `[{"registersPerThread": 138, "staticSharedMemory": 0, "dynamicSharedMemory": 0, "localMemoryPerThread": 0, "localMemoryTotal": 90439680, "sharedMemoryExecuted": 8192, "sharedMemoryLimitConfig": 0}]` |
| `_GroundCollisionDetect` | `[{"registersPerThread": 38, "staticSharedMemory": 0, "dynamicSharedMemory": 0, "localMemoryPerThread": 0, "localMemoryTotal": 68157440, "sharedMemoryExecuted": 8192, "sharedMemoryLimitConfig": 0}]` |

## fixed f40：归因细节

FullCCD query 生成 swept AABB 候选；真正 self ACCD 在 `_cub_reduct_self_step` 内。DCD query 已包含距离分类。edge-triangle 是线搜索安全查询，保留为独立类别。

| outside-Graph BVH launch owner | GPU union ms |
|---|---:|
| collision.discrete_bvh_build | 2.578076 |
| collision.swept_bvh_build_or_refit | 2.650620 |

| 唯一 runtime + native NVTX 的 energy owner | GPU union ms |
|---|---:|
| ipc.energy_production | 1.676381 |
| ipc.energy_scalar_reduce_and_readback | 0.129087 |
| ipc.energy_evaluation | 0.101600 |

此 capture 仅有7条 whole Graph activity，内部 SpMV、MAS apply、PCG reduce/control 耗时不可分解。

| 原 unknown 符号（最高15项） | 补充类别 | GPU union ms |
|---|---|---:|
| `_edgeTriIntersectionQuery` | collision.safety_edge_triangle | 8.450385 |
| `_calculate_triangle_fem_gradient_hessian` | fem.assembly | 5.895348 |
| `DeviceRadixSortOnesweepKernel<…>` | library.cub_radix_sort_unspecified_operand | 3.342458 |
| `gipc::<unnamed>::cal_abd_system_preconditioner_kernel2` | abd.preconditioner_prepare | 2.265277 |
| `_cub_reduct_self_step` | collision.accd_self_narrow | 2.165564 |
| `_calculate_fem_gradient` | fem.assembly | 1.683101 |
| `gipc::write_barrier_hessian` | abd.contact_assembly | 1.184063 |
| `_get_triangleFEMEnergy_Reduction_3D` | energy.producer | 0.963871 |
| `_calculate_quad_bending_gradient_hessian` | fem.assembly | 0.935231 |
| `_calFrictionHessian` | contact.friction_assembly | 0.882655 |
| `fast_segmental_reduce_ptr_kernel<…>` | library.segmented_matrix_reduce_unspecified_owner | 0.748575 |
| `_getQuadBendingEnergy_Reduction` | energy.producer | 0.285984 |
| `_getBarrierEnergy_Reduction_3D` | energy.producer | 0.228319 |
| `DeviceRadixSortHistogramKernel<…>` | library.cub_radix_sort_unspecified_operand | 0.223584 |
| `_getFrictionEnergy_Reduction_3D` | energy.producer | 0.151167 |

| 查询/ACCD kernel | 资源字段的实际取值 |
|---|---|
| `_cub_reduct_self_step` | `[{"registersPerThread": 178, "staticSharedMemory": 80, "dynamicSharedMemory": 0, "localMemoryPerThread": 0, "localMemoryTotal": 138936320, "sharedMemoryExecuted": 8192, "sharedMemoryLimitConfig": 0}]` |
| `_selfQuery_vf_ccd` | `[{"registersPerThread": 40, "staticSharedMemory": 0, "dynamicSharedMemory": 0, "localMemoryPerThread": 0, "localMemoryTotal": 85196800, "sharedMemoryExecuted": 8192, "sharedMemoryLimitConfig": 0}]` |
| `_selfQuery_ee_ccd` | `[{"registersPerThread": 40, "staticSharedMemory": 0, "dynamicSharedMemory": 0, "localMemoryPerThread": 0, "localMemoryTotal": 85196800, "sharedMemoryExecuted": 8192, "sharedMemoryLimitConfig": 0}]` |
| `_edgeTriIntersectionQuery` | `[{"registersPerThread": 58, "staticSharedMemory": 0, "dynamicSharedMemory": 0, "localMemoryPerThread": 0, "localMemoryTotal": 117964800, "sharedMemoryExecuted": 8192, "sharedMemoryLimitConfig": 0}]` |
| `_selfQuery_vf` | `[{"registersPerThread": 162, "staticSharedMemory": 0, "dynamicSharedMemory": 0, "localMemoryPerThread": 0, "localMemoryTotal": 111411200, "sharedMemoryExecuted": 8192, "sharedMemoryLimitConfig": 0}]` |
| `_selfQuery_ee` | `[{"registersPerThread": 138, "staticSharedMemory": 0, "dynamicSharedMemory": 0, "localMemoryPerThread": 0, "localMemoryTotal": 90439680, "sharedMemoryExecuted": 8192, "sharedMemoryLimitConfig": 0}]` |
| `_GroundCollisionDetect` | `[{"registersPerThread": 38, "staticSharedMemory": 0, "dynamicSharedMemory": 0, "localMemoryPerThread": 0, "localMemoryTotal": 68157440, "sharedMemoryExecuted": 8192, "sharedMemoryLimitConfig": 0}]` |

## 源码语义锚点

以下当前源码锚点用于检查符号语义；capture 的二进制身份由各文件夹 build manifest / capture validation 保留，不能用当前源码行号替代二进制身份。完整原符号、所有原规则 unknown 条目、补充分类与源文件 SHA256 均在 JSON。补充后两份 kernel 均归类，通用库 operand 与 Graph 内部 CPU owner 仍明确保留为未确定。

- DCD VF / narrow distance classification：[源代码](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/collision/mlbvh.cu:1265)
- FullCCD VF / swept candidate output：[源代码](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/collision/mlbvh.cu:1454)
- DCD EE / narrow distance classification：[源代码](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/collision/mlbvh.cu:1632)
- FullCCD EE / swept candidate output：[源代码](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/collision/mlbvh.cu:1814)
- self ACCD point-triangle branch：[源代码](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:10319)
- self ACCD edge-edge branch：[源代码](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:10333)
- edge-triangle safety traversal：[源代码](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:8167)
- energy production owner：[源代码](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:10784)
- energy final reduce/readback owner：[源代码](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:10890)
- tetrahedral FEM gradient：[源代码](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/fem/femEnergy.cu:2264)
- ABD generalized-to-Cartesian move direction：[源代码](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/abd_system/abd_system_function/cal_x_from_q.cu:28)

## 范围与复现

原 `analyze_profile` 规则与 `report_profile_attribution.analyze(folder)` 结果完整保存在 JSON；补充分类另存，不修改 tools。Graph 内部只按实际 kernel 符号分类，绝不使用 replay 的 CPU 范围归因。通用 CUB reduce 的 operand 仍未由符号确定；能量 final reduce 使用唯一 runtime correlation 的 owner 数据，不能把全部 CUB reduce 算为能量归约。

查询源码有编译期固定大小的局部 `uint32_t stack[65]`（260字节），安全 edge-triangle 为 `stack[64]`（256字节）。资源列确实存在：各查询/ACCD 的 `localMemoryPerThread` 均为0，`localMemoryTotal` 非零；这组 launch 描述无法确认数组放置位置，也未提供动态分配、spill load/store、local-memory transactions、occupancy 证据。DCD VF/EE 的 registersPerThread=162/138 与 FullCCD=40 可作为后续资源检查线索，不能据此证明 register spill 或 GPU 瓶颈。节点访问与成本因果仍需独立测量。

CPU/GPU 包络、Graph 内部节点与 whole Graph 均存在包含关系；跨度中的 gaps 不等同于可消除开销。单个 capture 未证明整场成本份额、整场5%收益或相对 Stiff >2×。

复现：`E:/Anaconda/envs/DL/python.exe reports/report56_stage1_20261008/analyze_profile_costs.py`。脚本以 SQLite `mode=ro` / `query_only` 打开，核对前后身份，输出本 JSON 与 Markdown。
