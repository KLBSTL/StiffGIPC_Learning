# AutoDL 完整测试：构建身份与计时审查

2026-10-06。本代理只读取本机源码和既有工具，并新增两个独立 CPU 构建身份工具及其测试；没有 SSH、安装、原生修改、构建或 GPU 运行。主任务提供的 AutoDL 4090/CUDA 12.8/24 GB、磁盘约 6.6 GB 是本轮运行背景，本审查未独立复测远端。

## 本轮最小构建与测试入口

从新上传的仓库根目录执行，`build/active`、`build/base` 和对应命令日志均须不存在。不要把 Windows 对象或旧 Linux 构建目录复制进去。以下命令供主任务串行执行，本文未执行它们：

```sh
python3 tools/full_eval/build_identity.py snapshot --root "$PWD" --out reports/autodl_full_20261006/source_before.json
python3 -c 'import json,time; f=open("build_start.json","x"); json.dump({"time_unix":time.time()},f); f.close()'
python3 tools/build_linux.py --kind active --jobs 2 --compiler /usr/local/cuda-12.8/bin/nvcc
python3 tools/build_linux.py --kind base --jobs 2 --compiler /usr/local/cuda-12.8/bin/nvcc
python3 tools/full_eval/build_identity.py verify --root "$PWD" --before reports/autodl_full_20261006/source_before.json --out reports/autodl_full_20261006/build_identity.json
```

[build_linux.py](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/tools/build_linux.py:24) 使用独立 Ninja Release、显式 sm_89、独立源码根和新建日志，拒绝重用构建目录。日志第一行保存实际 ARGV。编译期间不再修改源；失败的尝试保留证据，不把部分旧对象移到新身份。

CTest 前移除全部继承的 `GIPC_*`，否则另一个 fixture 环境变量可能先于当前 fixture 分支触发。可用 Python 标准库启动下列命令，环境由 `os.environ` 过滤 `GIPC_*` 后复制；不需要为构建工具安装 NumPy：

```sh
python3 - <<'PY'
import os, subprocess
env = {k:v for k,v in os.environ.items() if not k.startswith('GIPC_')}
subprocess.run(['ctest','--test-dir','build/active','-j','1','--timeout','120','--output-on-failure'], env=env, check=True)
PY
```

当前 active 定义 **7 项**：diag_fused_update_equivalence、pcg_guard_boundaries、ipc_budget_invariants、ipc_compensated_controller、contact_pool_equivalence、edge_query_order_equivalence、pcg_chunk_stop_equivalence。验收应要求七项实际执行，不能把“未发现测试”当通过。baseline 没有 CTest 定义，不能宣称其 CTest 通过。fixture 入口在 GLUT 初始化前；**完整 batch active/base 都仍先创建 GLUT 窗口**，所以完整运行需要有效 DISPLAY，例如由已有 `/usr/bin/xvfb-run -a` 包住新 controller。不要为此改冻结基线。

## 旧 seal 的实际能力和补充身份

[linux_runner.seal_build](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/tools/bench/linux_runner.py:71) 验证构建后源码、compile_commands 中 gipc 的源集合、可执行文件和少量构建证据。它没有检查对象文件实际存在/唯一/参与链接，没有编译前后源码快照，也不要求 Ninja 链接命令；Ninja 通常没有独立 link.txt。它自己的 scope 已明确只是构建后身份，不能放大为完整 clean-build 证明。

[fixed_quality.base_seal](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/tools/diagnostic/fixed_quality.py:116) 额外检查 baseline 源根、Assets 与 active 相等、原停止条件和 compiler binary，但同样未绑定实际对象/device-link/host-link。其全 Assets 相等条件在当前目录会拒绝下述 8 个旧缓存差异；新 full_eval controller 需要消费本附件的严格等价结果，不能直接把该旧入口的失败忽略。保留这两个旧工具；新增附件补齐，不覆写历史证据。

新增 [build_identity.py](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/tools/full_eval/build_identity.py) 提供：

- `snapshot`：冻结 active/base 的 StiffGIPC、Assets、MeshProcess 及两份 CMakeLists；记录两个构建目录此前不存在，拒绝已有构建目录。
- `verify`：前后完整源/输入清单相等；当前 active **37 TU=33 CUDA+4 C++**、base **35 TU=31 CUDA+4 C++**，数量不符报错。要求每 TU 有唯一、实际存在的 `.o`；核对 compilation database 与 `ninja -t commands gipc` 的真实编译命令（仅剔除 depfile 参数差异）。
- 验证所有对象都有 `.ninja_log` 执行条目、CUDA 对象参与唯一 device-link、device-link 与全部 TU 对象参与 host link；绑定对象、程序、compile_commands、CMakeCache、build.ninja、rules.ninja、.ninja_log、原 configure/build 日志、显式链接静态库和 compiler/tool binary 的 SHA。
- 保留命令原文、实际 ARGV、Ninja 相对执行时序、日志 mtime 和审查 UTC 时间。缺失响应文件不静默跳过；若生成器使用响应文件，应在新的 clean build 中保留 `-d keeprsp` 所需证据。
- `verify_identity(root, path, expected_sha=None)`：供新 controller 复核已封存附件，不再运行 Ninja；文件哈希、源/输入增删、对象、链接、编译器符号链接目标变化均拒绝。新 session seal 还应绑定附件路径及 SHA。
- 实际本机资产核对为 **61 个共同文件字节一致、0 个 active-only、8 个 baseline-only**。例外仅允许 `sorted_mesh/` 中 cipc_table/cloth_high 的 `_sorted.16.obj` 和 `.part`、cube/high_mat 的 `_sorted.16.msh` 和 `.part`，无目录级豁免。解析本轮 cloth_hang_l、cloth_fixed_bunny_l、bunny_cloth_bunny_l 的每个 `stiff_mesh`，验证冻结 mesh SHA 及 metis_sort 派生 `_sorted.16`/`.part` 缓存全部存在于共同资产；被选场景引用的例外会拒绝。8 个旧文件仍逐项包含在 baseline source 哈希中，`assets_identical=false` 如实保留，`assets_equivalence.passed=true` 只适用于这三个场景。

兼容主任务先行生成的通用快照：顶层 `sources`/`records`/`files` 数组，记录含相对 `path`/`relative_path`/`relative`、`sha256`、`bytes`/`size`。必须覆盖上述全部源根，缺 Assets 或 baseline 不接受。若通用快照没有构建目录存在性记录，附件会明确 `fresh_build_directory_prestate_recorded=false`，clean 来源须由主任务的新目录/命令执行证据补充，不能虚构此前已证明。

另要求本轮 supervisor 在快照写入后、启动任何构建前记录根目录 `build_start.json:time_unix`。附件强制 before 文件 mtime 严格早于该时间，所有实际对象及 configure/build 日志 mtime 不早于该时间，并逐对象保存 Ninja 相对 start_ms。supervisor 比 Ninja/configure 更早，二者相加只是每次编译绝对开始时间的保守下界，**不是实际精确编译开始 UTC**；不允许 post-build-only 快照冒充 prebuild。此时间链依赖受控 supervisor 的顺序和同机文件时钟，不是防篡改时间戳服务。主任务已经生成的 `build_pre_sources.json`、`build_start.json` 可直接使用，不应重写。

本附件不是编译器正确性证明，系统头文件和动态库没有穷尽冻结；`-l...` 仍明确保留为未解析系统库选项。编译环境中的 CC/CXX/CUDAHOSTCXX/FLAGS、工具链路径与依赖版本应由主任务记录；既有 builder 继承环境，不能只根据 Release 字样假定所有历史编译参数一致。

CPU 验证命令：`E:/Anaconda/envs/DL/python.exe -B tools/full_eval/build_identity_test.py -q`，**25/25 通过**（23.35 秒）。全部使用临时假源码/对象/编译器和 mock Ninja 输出，不执行 GPU 或编译。覆盖缺/重对象、错误 host/device link、编译选项分歧、无实际执行记录、前后源变化、缺响应文件、数量变化和封存后的对象变动；另覆盖精确 8 缓存豁免、未批准额外文件、被选场景引用例外、共同文件字节变化、构建后快照时间及缺绝对开始证据。实际本机三场景 Assets 检查也通过（61 common/8 exceptions）。

首个真实远端 verify 在输出附件写入前拒绝了 NVCC 的 `-x cu -dc source -o object` 命令，因为初版只识别 `-c`。失败日志保留，未重建原生程序。新适配只接受唯一的 `-c`，或 CUDA 源的 `-dc`/`--device-c`，要求紧随参数准确等于 metadata 源路径；不从其它位置猜测源文件。CPU fixture 已采用实际 `-dc` 形式，并检查 long alias、错误源、重复/混合/缺失标记以及 C++ 源误用 CUDA 标记。远端 actual verify 是否通过由主任务后续收据确认。

## 本轮必须统一的计时单位与边界

| 字段 | 实际含义 | 使用限制 |
|---|---|---|
| `trace/frames.csv:solver_ms` | CPU steady_clock 包住 `IPC_Solver` 和随后的 `cudaDeviceSynchronize`，毫秒 | 本轮整段/固定窗口加速比的可比主时间；**不是 CUDA-event 时间** |
| controller `solver_seconds` | 上述帧列求和除 1000 | 要求完整逐帧记录且两程序同帧窗；不能拿一个程序的 wall_seconds 作分母 |
| `wall_seconds` | 子进程墙钟，含启动/GLUT/载入/导出/退出及监视采样边界 | 用于超时/资源和端到端附加观察；baseline 少 velocity 导出，不能将其当等工作量求解速度 |
| `timeCost.txt:totalTime` | IPC 内 start/end0 CUDA-event 区间累计，**秒** | 比 CPU 帧计时范围窄；事件区间也包含主机提交空隙，并非 GPU kernel 时间总和 |
| `time0..time4` | 累计事件秒，分别对应 assembly/pcg/ccd/line_search/state_update | 每帧 `phase_ms` 对应毫秒；单位转换后可作累计一致性核查 |
| `phase_ms.pcg` | 从 calculateMovingDirection 前至返回的事件区间 | 包含 RHS/矩阵转换/预条件准备、PCG、分发，不能称纯 PCG 迭代成本 |
| `time_makePD` | 装配中的子项 | 不能再与 assembly 相加 |
| `totalCgTime` | 实际累加 `cg_count`，名称虽带 Time，单位是 solver 返回的迭代计数 | 不是秒；触顶返回值可能是 max_iter，而实际更新仅 max_iter−1 |
| 打印 `average time cost` | `totalTime / totalNT` | 分母不是物理帧数，不能当平均帧时间 |

CPU 帧计时来源：[active gl_main.cu:1898](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/app/gl_main.cu:1898)、[baseline gl_main.cu:1851](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/baseline/StiffGIPC/app/gl_main.cu:1851)。旧 linux_runner 的 timing_scope 把它称为 CUDA event，属于文案错误，新报告必须纠正。帧 trace/velocity、物理能量导出、defer-stats 写出发生在外层 stop 时间戳之后；IPC 内 `timeCost` 写入和部分日志仍在 CPU 帧区间内。

阶段事件来源：[core/GIPC.cu:11389](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:11389)；全帧事件和累计写出：[core/GIPC.cu:11808](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:11808)；线性准备与分发范围：[global_linear_system.cu:243](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/linear_system/linear_system/global_linear_system.cu:243)。此外 [calculateMovingDirection:10991](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:10991) 把返回 iter 写回统计，所以失败 run 的 pcg.iterations 不应冒充准确 GPU kernel 迭代数。

## 阶段与质量数据缺口

active 的 movement/residual 退出装配有 `ipc_exit_assembly_ms`，不在五阶段累计内；baseline 没有该字段。缺失必须为 unknown，不能填零后宣布成本全部归类。`solver_ms - sum(phase_ms) - 已观测退出装配` 只是未归类包络，混有主机工作、事件/日志管理和其它求解步骤，不能都叫可消除 CPU 开销；负小差也不应通过截零掩盖。若一个物理帧含多个 subIP，当前退出装配字段是赋值而非累加，亦不能先验当完整全帧累计。

`Timer{"pcg"}` 等内部计时器当前在 `build_gipc_system` 被 `Timer::disable_all()` 关闭；缺 timer 子项不代表零成本。Graph final_readback 是等待整个图的 CPU 区间，不能与图的 GPU 活动相加或当纯复制成本。当前 K1/K4 必须以 resolved 和每次 PCG execution/chunk 信息验真实执行路径，不能只看请求开关。

baseline 导出位置、拓扑、质量/边界属性、帧时间、phase_ms 和 iteration_limit，但**没有实际速度导出或 resolved_config**，数值 breakdown 遥测亦不完整。跨 baseline 的实际速度差应标 unavailable；不得由位移差分冒充速度，也不得为了补字段改冻结 baseline。active 自身 velocity 应按 0..steps 检查数量、`vertex_count*24` 字节和有限性。有限位置及 iteration_limit=false 不足以认证独立路径 CCD 或全部内部数值正常。

完整测试保持 IPC legacy `.01`、至少 6 次接受更新、PCG rho `1e-4`、材料/FullCCD 原设定。`toi_compensated` 是另一个 IPC 停止实验；`toi_al` 是不同方法，两者不能混进本轮“执行优化相对 Stiff”分组。停止条件或轨迹/迭代工作量变化必须单列。

磁盘方面，现有 runner 仅检查启动保留 4 GiB、运行保留 1 GiB；它不预估所有后续轨迹大小。单 run 的 state/velocity 裸数据约 `(steps+1)*vertex_count*24*(1+是否导出velocity)` 字节，另有 stats/日志。主任务的归档回执必须先绑定真实完整证据，再允许回收原始数据；部分帧或资源失败只能报告 prefix，不能进入完整速度中位数。

## 执行后核对（2026-10-07）

上述25项为开发时的独立检查，随后增加两项对象时间/执行记录拒绝测试，身份模块最终27项。
真实远端NVCC `-dc`映射已通过；全部37/35编译单元和38/36实际链接对象封存。
全工具四模块最终53项，本机复核命令如下，2026-10-07结果53/53通过（29.337秒）：

```text
E:/Anaconda/envs/DL/python.exe -m unittest discover -s tools/full_eval -p '*_test.py' -v
```

远端新工具53项、已有bench20项、quality6项合计79项通过。受原资源保护执行的原生命令
`ctest --test-dir build/active -j 1 --timeout 120 --output-on-failure`为7/7通过，监管墙钟2.287秒。
这些是程序/算子检查，不能替代材料、真实速度及独立接受路径CCD认证。

75次100帧和3项有条件跳过任务已经完成记账；最终冻结控制器再次核对返回
`plan_finished`、`gpu_launched=false`，没有追加GPU测试。
本机核对全部75份原始档案和1404份元数据，SHA与字节数均一致，
详见[下载核对](DOWNLOAD_VERIFICATION.json)和[完整报告](FULL_TEST_REPORT.md)。
