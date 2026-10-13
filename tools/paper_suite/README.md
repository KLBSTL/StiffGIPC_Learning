# 七类历史场景：100 帧、三轮

本套件按用户提供的三张图准备实际测量和原始关键帧，不复制图中的数值。
2026-10-10 用户选择七类场景各一种分辨率；当前选历史 low 输入。

| 图/分类 | Scene ID | dt (s) | 每方法帧数 | 重复 |
|---|---|---:|---:|---:|
| Sphere | cloth_sphere7_l | .01 | 100 | 3 |
| Bunny | cloth_fixed_bunny_l | .01 | 100 | 3 |
| Table | cloth_table_l | .01 | 100 | 3 |
| Stack | cloth_stack10_l | .02 | 100 | 3 |
| Hang | cloth_hang_l | .04 | 100 | 3 |
| Mixed | bunny_cloth_bunny_l | .01 | 100 | 3 |
| Boxes | boxes1920_cloth_l | .01 | 100 | 3 |

保持历史 dt、几何、变换、实例、材质、摩擦和 CCD。主实验显式使用当前
停止配置 Newton direction=.01、PCG rho=1e-4、MAS legacy、最少六次更新。
它们是可审计的统一实验设置，**没有宣称与截图产生时的所有参数相同**。
两个全 MAS 高精度开关和 IPC 自适应 PCG 均关闭。

## 五个完整方法

1. **stiff**：独立 frozen `baseline/` 二进制；这是主表的 Base。
2. **host**：当前二进制，原累计规则，所有额外执行优化关闭。
3. **graph**：和 host 同二进制、同参数，仅 PCG conditional Graph 开启。
4. **pure**：当前 combined 执行优化，现有纯累计规则，beta=.001，保留 movement exit。
5. **ours**：同 pure 的执行优化，现有 compensated budget AND residual gate；
   beta=.001、residual relative=.03，保留 movement exit。不是 AL/Robust TOI backend。

combined 包括 CCD BVH refit、batched energy、energy reuse；本轮也保持 FEM
direct pullback、MAS pivot-only 和 discrete BVH refit 开启。它们属于主方法
的已有执行优化组合。Graph-only 表中这些全部关闭，避免混入其他优化收益。

完整计划是 **7×5×3=105 次运行、10,500 物理帧**，三轮轮换方法及场景顺序。
每个方法每次都从零初始化，不从前一个方法的终点开始。超时上限每次 600 s；
失败无隐式重试，跳过该场景在当前阶段的剩余任务，记录部分帧，继续其他场景。
只将三次完整、有限、配置通过的观察纳入对应统计。没有预设或承诺 2×。

## 图和表

- `phase_table.csv/.tex`：Scene、Method、Hess、LSolver、CCD、LineS、Misc、TimeTot、
  P-bar、N-bar、P-tot、S；使用 frozen Stiff Base 和 Ours。
- `graph_table.csv/.tex`：平均 ms/step、Base/Graph 比、平均线性系统数和平均 PCG。
  此表 Base 是当前同一个二进制的 host，以隔离 Graph；与主表 Base 明确区分。
- `checkpoint_table.csv/.tex`：Table 第 28 步、Bunny 第 46 步；Original(Graph)、Pure TOI、
  Ours 的 N、P、t(s)、位置 RMS 与最大差(mm)。
- 每个场景原始帧 0/25/50/75/100；额外 Table 28、Bunny 46。全部 0–100 的
  状态和实际速度保留（frozen Base 只有位置，缺失速度明确记录）。
  输入门槛通过时另输出 Table 28/Bunny 46 的 Original/Pure/Ours 两行三列
  检查点图，三轮分别保留，不放大位移或用图像相似代替数值误差。
- `render.py` 输出与图一相近的五列、两行对照、七类场景各自的时间序列、
  PNG/PDF 和不缩放的原始世界坐标 OBJ。每次重复单独出图，不挑选最漂亮的一次。
  同一场景共享相机与空间范围，不重建、不平滑、不放大变形。

Hess 为已有 assembly 阶段，不声称是 Hessian 内核独占时间；LSolver 包含线性
准备及求解。Misc=TimeTot−四个主阶段，包含 state update、未分类/终止装配和
其他开销。TimeTot 是同步后主机测得的逐帧 solver_ms 之和，排除初始化及 trace
导出。三轮各列取运行级中位数，独立列的中位数不保证相加恰等于总时间；原始
三轮值始终保留。S 是三组配对比的中位数，不用不同轮的最快值相除。

## 同检查点消融

独立 seed 运行到 Table 28、Bunny 46，并在第 27/45 步结束后保存完整帧边界
状态，再完成一个连续原规则步骤，供恢复等价检查。每个状态分别执行三个
Original、Pure、Ours 和 strict 单步，共 24 次单步；另有两次 seed。

每行使用同一个 active 二进制、同一个原始检查点、相同 combined Graph
执行开关，所有派生矩阵、MAS 和 Graph 缓存冷启动重建。恢复前校验 binary ID、
物理上下文、静态拓扑/材质指纹及缓冲区；恢复后校验 device bytes，初始位置
和实际速度必须逐字节等于检查点。新的 schema 与旧冻结程序检查点分离。

每个候选和参考必须通过同二进制、同帧、逐字节初始位置/实际速度、全部
设备缓冲区与物理上下文恢复的输入门槛。没有通过就不输出 matched 消融表，
也不启动完整矩阵。三个 cold Original 与 seed 的 hot 连续终点位置/速度还
分别记录 absolute/relative 1e-8 检查，但不能把终点确定性当作相同输入的必要
条件：本地同规则 cold/cold 也出现微米级差异，hot/cold 的 1e-8 检查未通过。
该失败和原始误差保留，**没有宣称冷缓存与热缓存终点等价，也没有放宽物理
质量阈值**。同检查点表所有行统一冷缓存，只比较 cold/cold 规则。

strict 使用 movement-only 和 PCG rho=1e-6，保留同一个 movement threshold=.01；
禁用累计提前退出。这是更严的数值参考，不是精确解、严格非线性误差界或
物理质量认证。误差取候选三轮与参考三轮的全部 3×3 配对中最差值，单位 mm。
截图原文六次候选测量按本次用户要求改为三次，不混用次数。

## AutoDL 依赖与操作

本套件不创建实例、不安装依赖、不连接服务器。上传准备好的 tar.gz 后，在
**新目录**解包。需要现有 NVIDIA 驱动、CUDA Toolkit、CMake、Ninja、g++、
Eigen3、nlohmann_json、GLEW、GLUT、GLU、Xvfb/xauth、Python3、NumPy、Matplotlib。
为 WHILE conditional Graph，构建脚本检查 CUDA >=12.4；最终支持仍须远端编译
和逐 PCG execution=conditional_graph 的 smoke 校验。
[NVIDIA CUDA 12.4 官方说明](https://docs.nvidia.com/cuda/archive/12.4.1/pdf/CUDA_Runtime_API.pdf)。
本地已测试 CUDA13.0 Windows 构建，Linux 工具链仍待 AutoDL 验证。

为保留所有原始帧，建议预留 30 GiB 磁盘；运行中仍执行 4 GiB 启动/1 GiB 期间
空间门槛和显存预算 `min(.75*free_before,free_before−1536)`、768 MiB 空闲底线。
串行 GPU，发现其他 compute 进程中止本任务进程组，绝不清理外部进程。
2 帧 smoke 不能证明 100 帧时的显存峰值或数值稳定性。

```bash
tar -xzf paper_suite_20261011_v4.tar.gz
cd paper_suite
bash tools/paper_suite/build_autodl.sh
SESSION=runs/autodl_seven_100x3
MANIFEST=build/autodl_programs.json
xvfb-run -a python3 tools/paper_suite/run_suite.py run --manifest "$MANIFEST" --session "$SESSION" --stage smoke
xvfb-run -a python3 tools/paper_suite/run_suite.py run --manifest "$MANIFEST" --session "$SESSION" --stage seed
xvfb-run -a python3 tools/paper_suite/run_suite.py run --manifest "$MANIFEST" --session "$SESSION" --stage matched
python3 tools/paper_suite/analyze.py --session "$SESSION" --output "$SESSION/checkpoint_analysis"
# 仅当 checkpoint_input_gate.json 的 passed=true 时执行下一行；runner 也会检查。
xvfb-run -a python3 tools/paper_suite/run_suite.py run --manifest "$MANIFEST" --session "$SESSION" --stage performance
python3 tools/paper_suite/analyze.py --session "$SESSION" --output "$SESSION/final_analysis"
for r in 1 2 3; do
  python3 tools/paper_suite/render.py --session "$SESSION" --repeat "$r" --output "$SESSION/figures_r$r"
done
python3 tools/paper_suite/collect.py --session "$SESSION" --manifest "$MANIFEST" --output autodl_seven_100x3_results.tar.gz
```

build 脚本拒绝已有构建目录。运行和分析拒绝覆盖旧阶段/输出；失败后保留原目录
与 ledger，诊断原因后另开新 session，不能静默替换失败轮。`collect.py` 包含原始
运行、状态、速度、配置、阶段记录、GPU监测、误差、图片、两个实际二进制、
编译日志/命令与构建清单，旁存 SHA256。
下载时先核对 archive SHA，再据清单核对源代码/输入/二进制与原始数据。

## 本地复核入口

```powershell
& 'E:\Anaconda\envs\DL\python.exe' -X utf8 -B tools/paper_suite/contracts_test.py
& 'E:\Anaconda\envs\DL\python.exe' -X utf8 -B tools/paper_suite/prepare.py plan --output runs/new_preparation.json
& 'E:\Anaconda\envs\DL\python.exe' -X utf8 -B tools/paper_suite/local_smoke.py --session runs/new_local --build build/paper_suite_20261010 --stage smoke
& 'E:\Anaconda\envs\DL\python.exe' -X utf8 -B tools/paper_suite/local_smoke.py --session runs/new_local --build build/paper_suite_20261010 --stage checkpoint
& 'E:\Anaconda\envs\DL\python.exe' -X utf8 -B tools/paper_suite/prepare.py bundle --output transfers/paper_suite_20261011_v4.tar.gz
```

缺失数据只生成待测/缺失记录及表头，不填示例数字。有限性、拉伸和固定点漂移
是辅助检查；本套件尚未完成穿透、材料保真和同质量速度认证。应在完整结果返回
后检查真实关键帧、约束指标和重复波动，报告失败及方法适用范围。

v4 启动前最多等待10秒，必须连续两次 GPU utilization<=5 且没有其他 compute
进程；所有采样写入 `gpu_preflight.json`，等待时间不进入 solver 时间。发现其他
compute 进程立即拒绝启动，不终止外部进程。仍保持原有空闲、显存与磁盘门槛。
