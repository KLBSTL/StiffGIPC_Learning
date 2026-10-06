# v47 持续大方向事件与同系统准确解

日期：2026-10-04。本轮按用户要求在本机执行，沿用 v46 的收窄方向。结论：已获得真正持续大方向的 A/b 和实际解；一份异常系统的高精度 CPU 解仍有巨大方向，暂不把修改 PCG 停止准则作为主线。完整兔子轨迹、质量与性能仍未通过。

## 实现与范围

- 独立 `sources/stiff_perf_v47`，保留 v45 冻结版本。默认关闭 `GIPC_DIRECTION_EVENT_DIR`；启用后从物理帧 30 开始，同一 outer 内方向速度连续超过 5 m/s 达到 8 或 32 次时保存，进程最多 4 份快照。方向速度是试探 Newton 位移的最大坐标绝对值除以 dt，不能称为物体真实速度。
- 保存装配 A/b、实际广义解、物理方向、trial/safe 顶点、ABD q/dq、接触结构及摘要、能量分项、PCG 统计。线搜索前 metadata 先提交，完成后单独写 `_line_search.json`，避免超时覆盖已有证据。
- 捕获及后续能量读取前后比较受保护物理状态字节；本轮 8 份快照均通过。全程最多 4 份限制在小场景和兔子分别得到验证。小场景开关差异通过 1e-8 物理门槛，但不逐位一致。
- 不改原生 inner 退出、PCG 预算/容差、CCD、安全推进和材料。未导出完整 MAS；快照不是完整可重启状态。没有认证开关在整条接触轨迹上等价，也没有用诊断运行认证性能。

## 本机验证

RTX 3070 Laptop 8 GiB / driver 581.57 / CUDA 13 / VS2022 Release / SM86。Python 使用 `E:\Anaconda\envs\DL\python.exe`，NumPy 2.0.2、SciPy 1.17.1。

| 项目 | 结果 |
|---|---|
| Release 初次及最终增量构建 | 成功；初次沙箱 SDK 权限失败日志保留，授权后构建通过 |
| CTest | 2/2 通过 |
| GPU 组件检查 | 6 组检查全部通过，原始字段见 `V47_components.json` |
| 悬挂布料 3 帧 off/on | 均完成；最大位置差 2.01488e-15，相对 L2 1.18668e-16；逐位不一致 |
| 夹具事件检查 | 4 份，CPU A/b/x 残差与 GPU 一致、能量求和与物理步符号检查通过 |
| 小场景导出接受路径 CPU CCD | 共 10 段，零标记 |
| 兔子连续 40 帧目标 | 完成 34 帧，第 35 帧在 180.219 秒预算超时 |
| 兔子 CPU CCD | 127 段已导出接受路径/桥接，零碰撞标记；FEM 翻转观察 239 次，ABD 0 |

小场景仅为触发夹具，将阈值显式设为 1e-12、连续 1/2 次。兔子保持正式阈值 5 m/s、连续 8/32 次。兔子最低采样 GPU 空闲 4460 MiB，未触发显存预算。事件已覆盖持续异常，因此没有追加 host 或整场景参数扫描。

完成帧包含 174 次返回 PCG；未完成第 35 帧另有 170 次返回 PCG，最大 2593 次迭代，均未报上限/breakdown。最后 outer=6、inner=40 的 PCG 未返回，结果未知，不能算通过。最新已接受 outer 的 alpha=0.00142291、剩余 beta=0.72316687，推进仍很慢。最后完成的 inner 连续大方向计数为 40、方向速度约 17662 m/s。

第 34 帧已接受端点 FEM 最小 J=-0.465528，5 个非正四面体。CCD 零标记仅针对导出接受路径，不等于物理质量通过；Stable NH1 当前允许 FEM 翻转。下表更加极端的 J 属于试探构形，不能混同于已接受端点。

## 捕获事件

帧号从 1 开始；outer/inner 从 0 开始。4 份都在第 35 帧、inner=7、连续第 8 次超阈值时捕获。

| 事件 | outer | 方向速度 m/s | CPU 真相对残差 | 线搜索 r | 二次模型预测 ΔE | 实际 ΔE | trial 最小 J |
|---|---:|---:|---:|---:|---:|---:|---:|
| e1 | 2 | 83.3826 | 0.00455804 | 0.25 | -0.115499 | -0.107256 | -59.8262 |
| e2 | 3 | 48.5612 | 0.00276114 | 0.0625 | -0.00511973 | -0.00428753 | -28.7230 |
| e3 | 4 | 336.855 | 0.00131913 | 0.125 | -0.955140 | -0.0797950 | -254.988 |
| e4 | 5 | 5632.76 | 0.00724123 | 0.015625 | -120.200 | -5.13289 | -3344.51 |

4 个捕获 outer 后来都完成安全推进；最终超时发生在 outer=6，因 4 份预算已用尽，没有该 outer 的 A/b。这是事件覆盖限制，不能声称已保存最终卡住的那个线性系统。连续 32 次触发逻辑本次未取得快照：前 4 个连续 8 次事件已耗尽预算。

e4 线搜索前总能量 2283.412，其中 AL 接触 2254.717（约 98.7%）、FEM 28.674。接受 r=1/64 后，AL 接触减少约 68.506，FEM 增加约 63.355，总能量仅减少 5.133，远小于二次模型预测下降 120.200。此证据支持优先检查接触子问题与材料响应，但能量占比不等于梯度/方向贡献，更不能单凭此认定某段代码有错。

## 同一个 A/b 的准确参考

选择方向最大的 e4，以 CPU SciPy 稀疏直接 LU 解同一导出系统，独立进程 120 秒上限，实际求解 8.89 秒、进程 10.312 秒。没有改变仿真或重新装配 A/b。

- CPU 真相对残差 **2.70708e-13**，通过预设 1e-8 门槛；无秩警告，解全部有限。
- 原 PCG 的 FEM 最大方向速度 **5632.7636 m/s**，准确解为 **5555.0880 m/s**，只减少 **1.379%**。
- 全广义解相对差 **9.2018%**；PCG 误差并非可以忽略，但该异常系统的大方向在准确线性求解后仍然存在。
- 已核对原广义解的 FEM 部分与实际物理方向一致。参考解的 `-b·x` 约 -7773.505，`xᵀAx` 约 7773.505。

因此，这份 A/b 不支持“只收紧 PCG 就能消除大方向”。这是固定线性系统的一次单因素检验，不是把准确方向送入非线性线搜索后的完成性对照，也不是物理真值。它不能排除之前的线性误差改变了后续接触、轨迹和当前 A/b。

## 下一步收窄计划

1. 保持原生算法与预算，针对同一持续大方向事件增加分项梯度和局部能量方向导数检查，分别核对惯性、材料、AL 接触、摩擦；先判断装配梯度与能量一致性，再解释投影 Hessian 的模型误差。
2. 优先保存首次能量/方向突然放大前后的接触锚点、乘子、gamma、slack 更新关系；当前 `max_abs_penalty_mu=0` 是字段观测，不能据此认定全局 mu 未生效。必须沿实际使用路径核实。
3. 如需要实际/准确方向的能量比较，在同一活跃状态、固定接触集合上做有限步长的只读探测，逐次恢复并验证受保护状态；不使用尚未认证等价的帧检查点重启作为因果前提。
4. 只有上述证据指出具体问题后才做最小算法修复，再回到从 0 连续完成性与质量测试。完整 100 帧和性能重复继续待办；不添加任意方向截断、正体积限制或提高预算来掩盖问题。

## 身份、命令与证据

292 个源码文件与两二进制最终核验通过；v45 原冻结源码和二进制亦再次核验保持。v47 source digest：`529275a05406c183a910c7189f1af49c7d13f46a3d45c732e7bd1e5ce44a08e6`；gipc.exe SHA256：`c1b094338379948e5845e3393be92b266f8eab2268c5cac89de101a80ab69d4e`。辅助脚本哈希保存在 `FINAL_V47_20261004.json`。三次运行原始数据合计约 227 MiB，事件受限保存；本轮未清理旧证据。

在任务根目录执行，`python` 指上述 DL 解释器：

```text
cmake --build builds/local-v47 --config Release --target gipc diag_fused_update_tests --parallel 2
ctest --test-dir builds/local-v47 -C Release --output-on-failure
python tools/benchmark_events_v47.py smoke
python tools/analyze_events_v47.py v47_smoke_on
python tools/analyze_local_v47.py v47_smoke_off v47_smoke_on --ccd --output reports/VERIFICATION_V47_SMOKE.json
python tools/benchmark_events_v47.py graph
python tools/analyze_events_v47.py v47_bunny_graph
python tools/analyze_local_v47.py v47_bunny_graph --ccd --output reports/VERIFICATION_V47_BUNNY.json
python tools/run_reference_event_v47.py
python tools/summarize_v47.py
```

组件测试使用 `GIPC_VALIDATE_COMPONENTS=<绝对 reports/V47_components.json>` 启动 gipc.exe。运行/分析工具拒绝覆盖已有同名结果，复跑需新名字或新一代目录。首次小场景分析从错误 cwd 写报告失败，改在任务根运行后成功；没有改验证门槛。

主要证据：`V47_ctest.log`、`V47_components.json`、`V47_SMOKE_COMPARISON.json`、`VERIFICATION_V47_SMOKE.json`、`VERIFICATION_V47_BUNNY.json`、`EVENTS_v47_smoke_on_CPU.json`、`EVENTS_v47_bunny_graph_CPU.json`、`REFERENCE_V47_E4.json`、`REFERENCE_V47_E4_PROCESS.json`、`FINAL_V47_20261004.json`。事件本体在 `runs/local/v47_bunny_graph/events/`。
