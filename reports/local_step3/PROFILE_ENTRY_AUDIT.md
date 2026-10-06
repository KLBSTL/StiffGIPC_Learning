# 本机接触帧成本追踪入口审查（2026-10-06）

本次只读核查源码、配置与已安装工具，执行了 `nsys --version` 和 `nsys profile --help`，未启动 profiling/GPU、未修改原生源码、构建或实验工具。

**结论：新 standalone 原生程序已有所需的 cost/NVTX 入口，无需再编译；但当前 Windows 分轮 CLI 没有可直接安全启动 Nsight 的入口。** 可另建有限诊断任务，保留原 13 次分轮协议、程序身份及资源门禁。不能把 profiler 直接塞进现有 `Popen` 后仍宣称原单进程监管有效。

## 1. 已确认的入口和身份

- 当前程序：`build/local_step3_20261006/Release/gipc.exe`，10,838,528 字节，SHA-256 `b2ad19884fc01b81827d560f9f1984346b5488851995c2fd495f7427eac0f812`，来自 `build/local_step3_20261006/local_diagnostic_seal.json`。
- Nsight CLI：`C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.3.2/target-windows-x64/nsys.exe`；本机返回版本 `2025.3.2.474-253236389321v0`。本次仅查询帮助，不认为实际 capture 已验证。
- 原生 `StiffGIPC/gipc/cost_trace.h:30–66`：`GIPC_COST_TRACE` 开启，`GIPC_COST_FRAMES` 选择帧，`GIPC_COST_EVENTS=0|1` 选择是否插入 CUDA events，`GIPC_COST_OPERATOR_PROBE=0` 保持正常工作量。
- `StiffGIPC/core/GIPC.cu:11504–11505` 的 `ipc.physical_frame` 是整物理帧 NVTX 区间。只有所选帧产生 CostScope，因此可从零推进而只捕获第 57 帧。
- `tools/bench/config.py:260–263` 已将 `diagnostics=['cost']`、`cost_frames`、`cost_events` 映射到原生环境。`profile` 只是配置值，真正启动 profiler 仍需要 launcher。
- NVTX 是否实际编进程序由 `__has_include(<nvtx3/nvToolsExt.h>)` 决定；未来必须检查 cost.jsonl 的 `nvtx_available=true`，不能仅凭源代码存在 NVTX 宏就认定 capture 成功。

## 2. 最小配置

优先复用已完成 `fixed_off` 的完整 expanded_config，只把下面诊断字段写入**新的独立诊断任务**；其余材料、legacy 停止、PCG rho、执行组件、状态/速度输出全部继承原配置。新输出目录不得覆盖旧 runs/receipt。

```json
{
  "scene": "cloth_fixed_bunny_l",
  "steps": 59,
  "timeout_seconds": 120,
  "diagnostics": ["cost"],
  "cost_frames": "57",
  "cost_events": false,
  "profile": "node",
  "contact_pool_validate": false
}
```

这是从零运行 59 帧、仅观察接触后的第 57 帧；不是检查点续算，也不是缩短原性能验收时长。pool on/off 按被研究的既有臂选择，一次诊断只对应一个臂，不打开同状态 validation、operator probe 或 fixed study。

两种互补模式不能混为同一性能结果：

| 模式 | 配置差量 | 能回答的问题 |
|---|---|---|
| 轻量 NVTX/CPU 时间线 | `cost_events=false, profile=graph` | 物理帧与整个 Graph 的时间线、图外准备/等待；看不到 Graph 内各 kernel |
| kernel 分解 | `cost_events=false, profile=node` | Graph 内 SpMV/MAS/归约等实际 GPU 节点成本；采集开销更高 |
| 无 Nsight 的 CUDA event 分项 | `cost_events=true, profile=none` | 图外分项及整体包络；含额外 event 和 flush 同步，不能分解捕获图内部 |

如果当前目标是解释线性阶段约 45% 的内部构成，单个 `node` 帧最直接；`graph` 模式只能给整体包络。无需同时开启 event 模式，也无需重型算子探针。

新 launcher 所需 Nsight 命令形状如下（**参考参数，不是已执行命令；环境必须由公共 config 构造并清除 ambient GIPC 变量**）：

```text
<nsys.exe> profile
  --trace=cuda,nvtx --sample=none --cpuctxsw=none
  --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0
  --capture-range=nvtx --nvtx-capture=ipc.physical_frame
  --capture-range-end=stop --cuda-graph-trace=node
  --output=<new_output>/nsight
  <sealed Step3 gipc.exe>
```

本机帮助确认 `capture-range-end=stop` 只停止第一次所选区间的采集，目标程序继续运行；因此 `cost_frames='57'` 与单帧捕获匹配。不能把多个帧范围加进去仍声称 `stop` 捕获了每一帧。`--duration=120` 是采集时长，不能替代从启动时刻计算的 120 秒外部总预算。

## 3. 当前工具的边界与资源门禁

`tools/local/local_plan.py:13–22` 固定 `diagnostics=[]` 和 `profile='none'`；`windows_runner.py` 的 CLI 只接受既定 stage，不接受任意 cost config。`execute()` 虽可生成 cost 环境，但始终直接启动 `gipc.exe`，并将 `heavy_diagnostics` 写为 false，不能不加诊断记录就作为正式 profiling 入口。

更关键的是 `windows_runner.py:33–37,50–53,103–119`：stop_owned 只持有直接程序 handle，外部 GPU PID 检查只认可这一个 PID。换成 nsys 后，真正的子进程 gipc 会被误判为 foreign；只结束 nsys 也不能保证子程序已退出。必须由独立薄 launcher 记录本次 profiler 与其子程序身份、拥有并终止本次进程树，不能通过全局按名称杀进程解决。

历史 `E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/tools/active/run.py:105–110` 有 Nsight 参数参考，但它绑定旧 overlay 的 manifest/路径及旧锁。不能原样调用它为新 standalone 制造身份正确的运行证据。

独立诊断可以保持原资源约束：使用同一个 `runs/.local_gpu.lock`；启动前保存实际 sealed binary/source/DLL 与配置；从启动起 120 秒 deadline；显存预算固定为 `min(.75*free_before, free_before-1536)` MiB，至少 1024 MiB；运行中 free 至少 768 MiB；磁盘启动前至少 4 GiB、运行中至少 1 GiB；保持串行。profiler 自身开销也计入这些约束，超限即中止，不延长预算或把前缀当完整追踪。

因此：**原生无需改，原测试协议无需改；现成公开 Windows CLI 不能直接做这次 profiling，需要一个独立诊断入口及新的有限任务记录。** 此前已发生显存资源中止，不能预先保证带 profiler 一定能到第 57 帧。根任务最新状态是悬挂反序第二轮在第 40 帧触发显存预算，后续 GPU 已停止；本审查只作为资源恢复后的下一步，不自动启动 profiling。

## 4. 已有离线分析入口与解释规则

历史目录中有可复用的只读分析器：

- `tools/active/contact_linear_profile.py --sqlite <export.sqlite> --cost-jsonl <cost.jsonl> --output <new.json>`：按实际 kernel 符号核算 MAS/SpMV/PCG GPU 活动；未知符号保留，泛型 CUB 不伪归属到 rho 或 pAp。
- `tools/active/analyze_ipc_light_cost.py --sqlite <export.sqlite> --output <new.json>`：按物理帧裁剪 GPU 区间并计算 union。
- `tools/summarize_cost_trace_active.py <cost.jsonl> --output <new.json>`：原生 cost JSONL 汇总。

这些路径均位于旧项目 `stiff_toi_cudagraph_20260929`，只能作为离线读取入口，不能借此加载旧 launcher/构建逻辑。应记录分析器哈希，先确认新 capture 的表/字段和所需帧确实存在。可用当前 nsys 的 `export --type=sqlite --output=<new.sqlite> <nsight.nsys-rep>` 导出；本次未执行导出或分析新 capture。

解释时必须保留以下边界：

1. 原“pcg/线性阶段”包含矩阵转换、预条件器准备、实际迭代和结果分发，不等于纯 PCG kernel。现有 45% 占比不能直接归因给 SpMV。
2. CostScope 为嵌套 inclusive；用 parent_scope_id 去重或选最外层包络，不能将父子时段相加。CUDA event 间隔也包含 host 提交间隙，并非纯 GPU busy。
3. `graph.final_readback` 的 CPU 时长主要可能在等待整段 Graph；不能把它另加到 GPU 时间，也不能当作可全部省去的 memcpy 开销。重叠 GPU 活动需用区间 union 解释 wall，kernel duration sum 不自动等于可节省关键路径。
4. 捕获图内部禁止插入 CostScope events；Graph 重放成本必须来自 Nsight 节点或图整体活动，不能按捕获时的单次 NVTX 标签重复归属。
5. `node` 模式本机帮助明确可能有显著开销。追踪结果是诊断；不能与无诊断 r1/r2 的 9.538/10.131 秒直接计算加速，也不能用单帧百分比代表完整 59 帧。WDDM 共享桌面负载仍不受控。

本审查没有新增 kernel 候选，也没有将已有约 1.5× 的短主窗观测解释为完成同质量 2× 验收。
