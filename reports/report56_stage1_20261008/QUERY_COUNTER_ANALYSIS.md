# 单查询动态计数检查

真实 GPU 捕获与从零 120 帧运行已完成。原 collector 将 wide CSV 当作 long CSV，离线解析出现 `KeyError: Metric Name`；原失败回执保留，本脚本只修正离线读取，未重跑 GPU。

采样：四组件组合、落球 L、f49、第一个匹配的离散 EE 查询；仅一个 kernel，不是整场速度组。

| 指标 | 观测 | 单位 |
|---|---:|---|
| gpu__time_duration.sum | 1.18925e+06 | ns |
| l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum | 1.2138e+06 | sector |
| l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum | 615043 | sector |
| sm__warps_active.avg.pct_of_peak_sustained_active | 12.57 | % |
| smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct | 21.98 | % |

| launch | 观测 | 单位 |
|---|---:|---|
| launch__registers_per_thread | 138 | register/thread |
| launch__registers_per_thread_allocated | 144 | register/thread |
| launch__waves_per_multiprocessor | 0.9 |  |
| launch__sm_count | 40 | SM |
| launch__grid_size | 36 |  |
| launch__block_size | 256 |  |
| profiler__replayer_passes | 1 | pass |

该次 kernel 的测量时长为 1.189248ms。数据证明实际存在 local-memory 访问，不能证明其全部来自 BVH 栈；堆栈、窄相调用、编译临时量与网格不足同时存在。

Dynamic local-memory traffic includes stack, calls and compiler temporaries; no attribution solely to traversal stack.
One EE kernel does not measure all VF/EE queries, narrow-phase share or whole-run removable cost.
Occupancy also depends on 36 blocks over 40 SMs; fewer registers alone cannot fill an underpopulated grid.
Local-sector counts and long scoreboard stalls do not prove a stackless speedup or >2x result.
Original producer parser failure is preserved. This is offline format repair, with no GPU retry.

复现：`E:/Anaconda/envs/DL/python.exe reports/report56_stage1_20261008/analyze_query_counters.py`。原始捕获只允许一次，脚本不会修改 result/evidence 或重启模拟。
