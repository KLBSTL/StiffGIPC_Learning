# IPC 轻量成本观测与当前组合复核（2026-10-05）

本轮完成了轻量观测实现、12 次本机有界运行、6 份 Nsight 节点追踪和 8 项既有 GPU 回归。没有新增求解或碰撞优化 kernel，没有新的整场加速比，也没有推广默认配置。结果用于决定优化对象，不能替代材料质量验收。

## 1. 实现及身份：已验证

- 活动入口仍为 `sources/stiff_active`、`builds/active`、`tools/active`。
- 新增显式诊断参数 `cost_events=false`：保留 NVTX、CPU scope 和上下文编号，不为成本 scope 创建/记录 CUDA event，也不在 flush 时增加流同步。正式运行关闭成本诊断时行为不变；原事件诊断模式默认仍为 true。
- `requested_config` 和 `resolved_config.cost_observation` 同时记录请求与实际模式；轻量模式的全部 `gpu_interval_ms` 为 null，不能把 CPU 区间当作 GPU 执行时间。
- IPC 仍为 `legacy/.01/min_updates=6`，PCG rho 为 `1e-4`，legacy MAS restriction 为 `atomic`。此前封存的 ordered 路径没有重新启用。
- Release 编译 37 个单元、7 个对象变化；源码、对象和工程包含依赖均与 manifest 一致。当前程序 SHA256：`d41c9553ab50ff95bcdfb293dca638c625ea278449d1fbfa2ae9448a9b3ea710`。
- 12 项配置测试、58 个 subtest 通过；8/8 既有 GPU fixture 通过。fixture 的范围不等于新增求解器的全部边界验收，本轮没有新增求解器。

构建记录：`builds/active/provenance/20261005T122542_074629Z_ipc_light_cost_20261005/manifest.json`。程序身份和交付核对见 [验证回执](ipc_light_cost_20261005_verification.json)。

## 2. 预声明预算与实际覆盖：已验证

配置：[公共运行计划](../../configs/active/ipc_light_cost_20261005.json)；[协议](ipc_light_cost_20261005_protocol.json)；[批次结果](ipc_light_cost_20261005_batch.json)。全部从零推进、dt=.01、单次上限 120 秒，GPU 串行，保留既有显存/磁盘保留线。

| 组 | 运行数 | 覆盖 |
|---|---:|---|
| Graph / 当前组合，两个布料 | 4 | 推进 3 帧，仅追踪 f1 |
| 当前组合 off / CPU-NVTX / events | 6 | 两场景各一次 3 帧；固定兔子反向顺序 |
| 悬挂 Graph / 当前组合 | 2 | 推进 43 帧，仅追踪 f41 |

12/12 完成，没有触发本轮资源保留线、PCG 上限或 breakdown。当前组合指 swept FullCCD refit、批量能量及接受能量复用，均显式开启；完整安全 CCD 保留。

校正历史窗口：旧 Nsight `hang_graph_nsys_v2` 实际捕获 f41、`fixed_bunny_graph_nsys_v2` 捕获 f39，并非初始化帧。查明后，在本轮任何 GPU 运行之前将两次悬挂晚窗写入协议。固定兔子晚窗和混合场景此前资源失败仍保留，本轮未重试；初始固定兔子追踪不能替代晚窗覆盖。

## 3. 观测短前缀：已验证，但不是长轨迹等价

off / CPU-NVTX / events 的 PCG 逐方向次数、接受次数、退出原因及活动对工作量全部一致。位置最大分量差小于预声明 `1e-9 m`；不是逐位一致，也没有材料质量认证。

| 场景 | CPU-NVTX vs off 最大位置差 m | events vs off 最大位置差 m | off / CPU / events 的 3 帧求解秒数 |
|---|---:|---:|---|
| 悬挂 | 6.234e-11 | 3.883e-11 | .109060 / .112605 / .130269 |
| 固定兔子 | 3.259e-12 | 1.502e-11 | .213780 / .221732 / .229643 |

最大接受 alpha 差约 `9.93e-9`。CPU 模式观测成本约 +3.25%/+3.72%，事件模式约 +19.45%/+7.42%；每模式只有一次短窗，这些是诊断开销观察，不是生产加速比或置信结论。真实速度导出在所有本轮场景运行中均关闭。

Nsight 中仍有生产代码原生 event，不能称“零 event”。六份轻量追踪的 create/record 计数分别为悬挂 f1 的 26/22、固定兔子 f1 的 20/16、悬挂 f41 的 104/104，与原生每 Newton 轮 6 个 event 加每帧 2 个 event 的时序相符；没有新增成本 scope event。

## 4. 真实 GPU 活动与等待：已验证

由 Nsight SQLite 直接核对 kernel/copy/memset，区间裁剪到物理帧。CPU memcpy API 时间包括等待先前设备工作完成，不能视为搬运带宽成本，也不能与 GPU 时间相加。

| 场景 / 配置 / 帧 | CPU NVTX 帧 ms | GPU 活动并集 ms | GPU copy 总 ms | copy / CPU 帧 |
|---|---:|---:|---:|---:|
| 悬挂 Graph f1 | 81.067 | 9.225 | .165472 | .2041% |
| 悬挂组合 f1 | 66.096 | 8.052 | .137280 | .2077% |
| 固定兔子 Graph f1 | 90.207 | 26.182 | .295167 | .3272% |
| 固定兔子组合 f1 | 94.939 | 26.666 | .220000 | .2317% |
| 悬挂 Graph f41 | 391.632 | 121.721 | 1.035931 | .2645% |
| 悬挂组合 f41 | 321.014 | 115.255 | .815932 | .2542% |

例如组合悬挂 f41 的 memcpy CPU API 总时间为 **245.744 ms**，实际设备复制仅 **.816 ms**。D2H 408 次、7836 字节、最大单次 80 字节；H2D 17 次、68 字节；D2D 253 次、约 30.70 MB。优先优化复制带宽的依据不足；标量读回的依赖链仍需另行证明可安全消除。

同帧 GPU 活动跨度 320.129 ms、并集 115.255 ms、间隙 204.874 ms。间隙同时包含提交、分析器、共享桌面调度和真实依赖，不是可直接兑现的 CPU 开销。7820 条 Graph 节点没有 runtime correlation，保持未归属，不能强行算入 MAS/SpMV 子阶段。

初始追踪有冷启动开销：profiled 三帧 solver_seconds 约 3.4 秒，独立无追踪控制只有 .1–.23 秒。NVTX 帧也不覆盖全部分析器激活时间。不能从 profiled 总时间计算生产加速比。

## 5. 当前组合的阶段定位：支持，但尚非可推广收益

以下为组合悬挂 f41 的 **inclusive GPU-projected NVTX 区间**，含区间内部等待/间隙。父子嵌套，不能相加得到完整 wall-clock，也不能与上表实际 GPU activity 求和混用。

| 范围 | projected ms | 解释 |
|---|---:|---|
| 物理帧 | 320.129 | 参照区间 |
| 完整线性阶段 | 117.648 | 含准备、矩阵转换及 PCG，不是单个 kernel |
| 梯度/Hessian 装配 | 66.035 | 关联 GPU 工作可能跨过 CPU 子区间结束 |
| 线搜索 | 68.005 | 包含离散碰撞及能量子范围 |
| 普通离散 BVH 构建 | 16.825 | 不能与所属父范围相加 |
| 离散查询 | 29.022 | 同上 |
| swept BVH 构建/refit | 2.047 | 现有 refit 已开启 |
| swept 查询 | 9.399 | 完整 CCD 查询 |
| self CCD | 15.982 | 安全检测保留 |
| 能量评估 | 3.172 | 已有批量/复用路径 |

Graph-only 的对应 swept 构建为 30.335 ms、能量为 12.004 ms，支持已有组件已处理这两项成本。两追踪不是同状态实验：f41 两者均 17 个方向，但 PCG 分别 458/460，beta 也不同；每臂一次追踪，不能将表中差值宣传为组件加速比。

MAS 融合、Hessian 符号拆分和普通 BVH refit 的启动门槛仍没有齐全证据。尤其 Graph 内核不能用图外一次初始化作用的时间代替全部重放作用时间。

## 6. 有界候选判断：待验证，未实现

只保留一个具体假设：普通离散 EE 查询中，以子树最大**原始边 ID**提前排除必被 `obj_idx <= self_eid` 叶过滤拒绝的子树。代码审查支持该静态谓词安全；Morton 排名不能替代原始 ID；FullCCD 不受影响。

真正实施需冻结同一 GPU 树，逐项核对 pair 多重集合、类型及 MatIndex，覆盖增长/overflow，计入元数据构建和读取。重载位置后重新建树不是同树验证。

组合悬挂 f41 的 `_selfQuery_ee` 实际 kernel 总时间 17.329 ms，约诊断帧 5.398%；组合固定兔子 f1 为 5.452 ms、约 5.743%。按这些诊断比例，要净省整帧 5%，EE 本身需净减约 **92.6% / 87.1%**，尚无证据达到。比例受追踪影响，不是正式性能上限。没有同树净收益和代表窗证据，因此不写新 kernel、不扩大参数网格。

**下一步边界：**只有在剩余有限预算内能获得同状态遍历/准备成本，并支持整场至少 5% 潜力，才进入这一候选；否则停止此分支，提交当前瓶颈和未达阶段目标结论。继续增加观测并不构成加速目标的完成。

## 7. 质量、默认与历史边界

- 原质量协议 SHA 保持 `1cfb9045d6988fa606b8bb76afdd71e20661ef602c84296c49cdda9a0d07d78f`。历史失败决定和数值比较容差均未修改。
- 本轮证明短前缀工作量与位置兼容，不证明长轨迹、材料质量或“两个方法都达标”。此前固定兔子基线和候选的材料越界仍有效。
- 未做新 100/300 帧、AutoDL、七组配对或质量审计；没有新默认执行组合。
- 实验索引按追加方式登记 12 次运行、fixture、诊断交付及待验证假设，保留此前 381 项原数据/决定；保存旧条目与决定的摘要核验。

## 8. 可复现入口和证据

在工作目录使用 `E:/Anaconda/envs/DL/python.exe`：

```text
tools/active/test_contracts.py
tools/active/build.py build --label ipc_light_cost_20261005 --jobs 2
tools/active/batch.py --plan configs/active/ipc_light_cost_20261005.json
tools/active/fixtures.py --name ipc_light_cost_20261005_fixtures
tools/active/export_ipc_light_cost.py
tools/active/analyze_ipc_light_delivery.py
tools/active/verify_ipc_light_delivery.py
```

运行/导出/分析采用不可覆盖输出。重现时用新的运行名和输出名，不直接覆写本轮证据。SQLite 分析器可独立使用 `analyze_ipc_light_cost.py --sqlite INPUT --output NEW_JSON --top 0`，只读输入。

[主分析 v2](ipc_light_cost_20261005_analysis_v2.json)绑定六份 sqlite/nsys/config/程序身份；[GPU 回归](../../runs/active/ipc_light_cost_20261005_fixtures/result.json)及[验证回执](ipc_light_cost_20261005_verification.json)保存核对结果。v1 分析保留；v2 修正了选择帧/组件推断和 top20 截断遗漏问题，不需要重跑 GPU。

首次手工 SQLite 导出因 PowerShell `--output` 参数拼装失败，日志保留；改用结构化 subprocess argv 后六份导出成功。最初分析的 `input` schema 字段错误也已修正。两者属于 CPU 工具错误，没有将失败删除或把重新分析作为新增 GPU 观测。
