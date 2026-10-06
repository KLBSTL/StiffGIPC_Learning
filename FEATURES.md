# 融合实现与运行入口

## 版本和目录

官方基线固定为 `bb2849a7b292099581907937860d96ecfdf42588`。`sources/stiff_base` 只有共同的输入、批量运行、记录及退出清理适配；算法修改在 `sources/stiff_fused`。源码清单为 `IMPLEMENTATION_MANIFEST.json`，每次运行另存二进制 SHA-256、配置及场景预估。旧版本源码包保存于 `bundles/frozen`，AutoDL 各版本包分别保存于任务独立目录。

## 四组

| 组 | 接触 | PCG | v3 A 优化 |
| --- | --- | --- | --- |
| base | 原 IPC | host | 关闭 |
| base_graph | 原 IPC | conditional Graph | 开启 |
| base_toi | TOI/AL | host | 开启 |
| base_toi_graph | TOI/AL | conditional Graph | 开启 |

`base_graph` 和 `base_toi_graph` 中的 Graph 组均包含所迁移的 A 组合。用 `--suite 0` 可以只比较 Graph PCG；用 `fused_host --suite 0` 可检查移植后的普通 IPC 路径。

## 开关

- `GIPC_CONTACT_BACKEND=ipc|toi_al`；默认 `ipc`。
- `GIPC_PCG_EXECUTION=host|conditional_graph`；默认 `host`。
- `GIPC_ACCEL_SUITE=1` 开启 A；也可分别设置 `GIPC_MAS_STATIC_TOPOLOGY`、`GIPC_CCD_BVH_REFIT`、`GIPC_BATCHED_ENERGY`、`GIPC_ENERGY_REUSE` 为 `0|1`。
- 能量复用只用于 IPC。TOI 的 slack、乘子和线性化发生变化时，不沿用旧能量。
- `GIPC_TOI_EARLY_TERMINATION=1`：无 AL 活动约束、完整 CCD 接受 alpha=1 且位移达到原 base 门槛时退出；接触阶段保留累计 TOI 判据。设置为 `0` 做原 Kmin=6 消融。
- `GIPC_TOI_FILTER_RATIO=0.2`：活动集筛选的 ACCD 保守率，默认沿用原实现。修改只影响筛选，最终安全 CCD 保持 0.2；0.01 的球布料诊断更慢，未采用为默认。
- `GIPC_TOI_MU_MODE=diagonal|mass`：默认 `diagonal`；可选 `mass` 按 FEM 顶点质量和 ABD 刚体总质量为每个接触设罚刚度，可用 `--mu-mode mass` 运行。v18 的 bunny 100 帧完成，但物理偏差和四面体翻转未通过验收，不能作为默认修复。
- `GIPC_TOI_PERSIST_CONTACTS=1`：可选跨帧保留接触和摩擦基，单独消融延后了旧 bunny 失败，但尚未通过 100 帧质量验收。
- `GIPC_TOI_STALL_WINDOW` 与 `GIPC_TOI_STALL_HOLD_DELTA`：停滞升级诊断开关；过早升刚度可能触发 PCG 上限，保留默认策略。
- `GIPC_AUDIT_GRAPH=1`：同一 A/b/preconditioner 上重复 host 求解；记录 Graph/host 与 host/host 的差异。
- `GIPC_AUDIT_PCG=1`：记录实际欧氏残差。`pcg_tol` 是旧 rho 比门槛，不应解读为欧氏残差门槛。
- `GIPC_PCG_REPLAY_FROM_FRAME=N`（`--pcg-replay-from-frame N`）：默认关闭；从指定帧的第一个线性系统开始，固定同一已组装 A/b 和预条件器，从零初值重复同执行路径两次，Graph 另对照 host 两次。记录解差、迭代数和真残差，核对 b 未变并逐字节恢复原解；所有运行时间标为诊断。
- `GIPC_AUDIT_REFIT=1`：每帧比较 refit 与重建的完整 CCD 候选集合。
- `GIPC_AUDIT_ENERGY=1`：核对批量/原能量及复用能量。
- `GIPC_TRACE_SUBSTEPS`：输出真实已接受路径供独立 CCD 检查。内部 trial 允许穿透，不作为安全输出。
- `GIPC_TOI_ROBUST_VELOCITY_TOL`：可选的公开 Robust 大方向保护，单位 m/s；单独消融，默认未开启。
- `GIPC_TOI_ROBUST_VELOCITY_STOP=1` / `--robust-velocity-stop`：需同时设置 `--robust-velocity-tol`；允许已通过能量检查且方向速度低于阈值的内层求解退出，后续完整 CCD 与 safe 状态验证不变。默认关闭；v19 AutoDL 100 帧未测出加速，见 `reports/TOI_INNER_STOP_V19_AUDIT.md`。
- `GIPC_PCG_PRECONDITIONER=mas|diag`：fused 二进制的诊断入口，不作为速度公平比较的默认设置。
- `GIPC_TOI_MU_SCALE`：罚刚度初值乘数，默认 1；×0.001 仍未完成 100 帧兔子测试。
- v12 根据编译材料选择体积约束策略：Stable NH1 与原 base 一致，默认不额外限制 FEM 翻转。`--safe-injectivity` / `GIPC_TOI_SAFE_INJECTIVITY=1` 显式开启严格体积消融；此时使用归一化三次体积限步并复核实际 safe 路径。`GIPC_TOI_VOLUME_BOUND=legacy` 仅供旧风险复现。
- CPU 验证器 `--stable-nh1` 只改变 FEM 的正体积条件，继续报告翻转并保持 ABD 方向、连续 CCD 和地面检查。正式使用前核对实际编译材料。
- `--build-tag v5|v18` 选择保留的本机构建目录名；当前源码版本以每次运行保存的 source_digest 为准。

审计、逐步导出和详细 profiling 运行的时间只用于诊断。一般计时包含图捕获/重建，状态导出和 JSON 文件序列化在计时窗口外。

## 本机历史命令

在本任务目录中运行：

```powershell
& 'E:/Anaconda/envs/DL/python.exe' tools/run_local.py --arm base_toi_graph --scene cloth_sphere7_l --steps 100 --suite 1 --trace --name my_run
& 'E:/Anaconda/envs/DL/python.exe' tools/run_matrix.py --config configs/local_matrix_v4.json
```

场景包括五类布料、兔子–布料–兔子、盒子布料，各 L/M/H。`sphere7` 是旧场景编号 7 的单固定球场景。每个运行目录必须使用新名称；既有结果不会覆盖。用户随后要求本轮测试改在本机，v28 按该最新要求使用 Windows CUDA 13.0 / SM86；AutoDL 已先完成空间清理。

v28/v29 `--state-audit-from-frame N` 在 TOI 外层 0/1 及首次有活动接触外层的首个内层方向抽样，核对相同物理状态重复组装的转换后 A/b，以及能量采样后的顶点、q、safe/trial、接触 slack/乘子和摩擦快照。该选项默认关闭，启用时计时标为诊断。能量诊断结束后直接复制保存的顶点和 q，避免用 `step_forward(0)` 重新映射 ABD 顶点。

## AutoDL

任务路径为 `/root/stiff_toi_cudagraph_20260929`，CUDA 12.8，SM89；v17 和 v18 使用独立子目录。源码包验证使用 `python3 tools/verify_upload.py`；构建入口 `bash tools/autodl_build.sh`。运行器不依赖 NumPy；远端 `/root/miniconda3/bin/python` 已有 NumPy，可在 AutoDL 做物理量分析。

```bash
python3 tools/run_local.py --platform autodl --arm base_toi_graph --scene cloth_sphere7_l --steps 100 --suite 1 --trace --name my_run
python3 tools/run_matrix.py --config configs/autodl_matrix.json
```

远端性能运行器检查其他计算进程并拒绝并行测量。`--quality-only` 仅用于明确标为诊断的正确性运行；这些时间不得加入速度报告。

## 当前证据与限制

模块有限差分和 v17 盒子布料 100 帧 TOI 接受路径检查通过；AutoDL 四组盒子布料 100 帧已完成，但 Graph 长轨迹等价门槛未通过，正式质量匹配加速比 N/A。v18 可选质量罚参数单独完成 bunny 100 帧，但比 base 末态位置差约 4.14%，出现 48 个翻转四面体，独立表面 CCD 有 72 个保守标记。加跨帧活动集后 host/Graph 均完成 100 帧，host 末态无翻转，host dt/4 对 base dt/4 位置差约 1.034%，但 1404 个接受子步仍有 2 个表面 CCD 保守标记，Graph 等价也未通过。完整数据见 `reports/2026-09-30_COMPARISON_AND_GATES.md`。

本移植保留官方材料、PSD 投影和边界处理；没有移植论文框架全部功能，勘误正文未取得。ABD 广义坐标的初始 mu 是工程扩展，尚无完整物理精度验收。不得据此声明理论有限终止或所有场景的质量匹配加速。
