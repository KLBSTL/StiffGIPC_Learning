# Windows 本机有限诊断

只新增适配层，复用当前仓库冻结的 `tools/bench` 配置/指标和 `tools/diagnostic` 的 Stiff 配置能力验证，不导入历史项目 Python 代码。不会构建、清理、连接远端或自动进入下一轮。

计划共 13 次：fixed59 为 off1/base1、base2/off2、on1/off3/base3 三轮；hang51 为 off/on、on/off、off/on 三轮。均从零开始，dt=.01、minimum=6、Newton/cumulative=.01、rho=1e-4，单次硬预算 120 秒。显存预算保持 min(75% 初始空闲、空闲−1536MiB)，至少 1024MiB；低于 768MiB、磁盘储备低于 1GiB、确证外来 GPU 计算进程或监控失败会停止本次自有进程。启动要求磁盘至少 4GiB。TCC 保持两个 GPU 负载样本不超过 5%、无外来计算 PID；WDDM 的 N/A/权限不足进程显存仅记为未知桌面或计算负载，不能假造外来 CUDA 证据。WDDM 可在较高桌面负载下作资源受保护诊断；另一 gipc 或实际正数计算显存的外来 PID 仍阻断。所有负载样本原样记录，不终止外来进程；全部时间仅诊断，不能认证独占负载或加速比。

活动程序独立核对 Release 的全部源码→唯一对象→实际链接，兼容相对 XML 路径和 MSBuild 最后生效属性；CUDA 实际命令接受 `-c`、`--compile`、`-x cu`，不接受 device-link。还核对构建前源码快照、CL command、编译器版本/哈希、应用 DLL。原 Stiff 保留原 perf_v34 程序身份、源码、Assets 和构建证据；它没有 resolved config、真实速度或 breakdown 计数出口，不构造这些数据。程序相邻 DLL 已绑定，但没有宣称冻结全部 Windows/驱动 DLL。

资源核验要求全部 active Assets 在基线存在且逐项 SHA 相同。仅允许基线额外保留 `sorted_mesh/` 下 `cipc_table_sorted.16.{obj,part}`、`cloth_high_sorted.16.{obj,part}`、`cube_sorted.16.{msh,part}`、`high_mat_sorted.16.{msh,part}` 这八个旧缓存；seal 会根据当前13次计划逐项核对场景 `stiff_mesh` 及其 `.16` 排序缓存引用，拒绝引用白名单差异的场景，并记录八项完整哈希和忽略理由。其他额外文件、active-only、任何公共文件变化均拒绝；旧缓存不删除，初态、拓扑、质量材料比较不放宽。

以下命令在新仓库根目录执行，仅主任务负责启动 GPU。先完成当前构建，再 seal；seal 会拒绝正在变化的构建产物。默认不覆盖已有输出。

```powershell
& 'E:/Anaconda/envs/DL/python.exe' tools/local/contracts_test.py -v
& 'E:/Anaconda/envs/DL/python.exe' tools/local/windows_runner.py seal --build-dir build/local_step2_v3_20261006 --exe build/local_step2_v3_20261006/Release/gipc.exe --build-log build/local_step2_v3_20261006/build.log --base-root E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929 --prior-guard reports/autodl_step1/guards_analysis.json --output build/local_step2_v3_20261006/local_diagnostic_seal.json
& 'E:/Anaconda/envs/DL/python.exe' tools/local/windows_runner.py run --seal build/local_step2_v3_20261006/local_diagnostic_seal.json --session local_step2_20261006 --stage fixed_r1
```

每轮生成 `runs/<session>/<stage>/analysis.json`、batch 和 receipt，先核对材料、硬失败与输入身份，再由主任务决定是否继续。r2/r3 还必须明确传入上一轮实际分析 SHA，例如：

```powershell
& 'E:/Anaconda/envs/DL/python.exe' tools/local/windows_runner.py run --seal build/local_step2_v3_20261006/local_diagnostic_seal.json --session local_step2_20261006 --stage fixed_r2 --reviewed-previous-sha <已审阅的fixed_r1分析SHA256>
```

`hang_r1` 是独立系列首轮，`hang_r2/r3` 同样要求上一轮回执与完整原始证据未改变。任何硬失败阻止续轮。原质量阈值超界保留为本次因果诊断的待解释结果，不能用新增重复范围替换阈值。原 `allow_next_round=false` guard 不改写；新诊断始终 `allow_performance_screen=false`，不认证速度/质量、不自动启动长测。完整 raw、实际位置/速度与小报告均留在本机，不删除或上传。
