# Report 5/6 第一轮执行与分支决定

本轮状态：完成参考计时、独立基线复核、短帧 Nsight、探针实现和完整诊断，结束第一轮成本归因。生产结构缓存与 stackless 查询均暂缓；没有证据支持它们通过原计划的净整场 5% 进入门槛。当前相对 Stiff 约 1.44–1.46×，未达到 2×。本机为共享 WDDM，所有速度是诊断结果，不认证 2× 或同质量加速。

## 1. 参考结果和收益归因

两场景均从零连续 120 帧，dt=.01、legacy MAS、原 FP64 路径、rho=1e-4、累计阈值 .01、最少 6 次更新及完整 CCD。每场景三轮交错四臂，共 24 次完成。

| 场景 | Stiff 秒 | 活动 host 秒 | Graph-only 秒 | Graph＋四组件秒 | Stiff/Graph 配对中位 | host/Graph 配对中位 | Graph/四组件 配对中位 | Stiff/四组件 配对中位 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 落球布料 L | 11.851232 | 12.137970 | 9.202858 | 8.107089 | 1.2878× | 1.3315× | 1.1052× | 1.4618× |
| 固定兔子布料 L | 17.293908 | 17.513746 | 13.088928 | 11.995537 | 1.3194× | 1.3469× | 1.0938× | 1.4417× |

秒数是三轮中位数；倍数是逐轮配对比率的中位数，两者不可互相代算。四组件为 FullCCD refit、批量能量、能量复用和普通 BVH refit（固定间隔 8），并保留 Graph K1。同一活动程序的 host/Graph 才隔离 Graph 执行收益；活动 host 相对 Stiff 仍慢约 2.45%／1.20%。组合结果不能拆成四项各自的独立收益。

Graph 的线性阶段分别减少约 3.111／4.544 秒；PCG 次数没有一致减少，落球 Graph 三轮甚至多 642–2050 次，因此主要证据是执行成本下降。四组件进一步减少 CCD 约 .357／.363 秒、线搜索约 .535／.732 秒，并少做 540–603 次能量评估。完整工作量、输入和身份核验见 [参考成本表](REFERENCE_COST_ANALYSIS.md) 与 [JSON](REFERENCE_COST_ANALYSIS.json)。

## 2. 剩余成本与 2× 缺口

| 四组件组合的阶段计时中位数，毫秒 | 落球 L | 固定兔子 L |
|---|---:|---:|
| 装配 | 1365.727 | 1714.584 |
| 线性阶段 | 3035.349 | 5428.663 |
| CCD | 666.267 | 1214.534 |
| 线搜索 | 2778.907 | 3175.492 |
| 更新 | 34.416 | 37.478 |

这些是 CUDA event 端点包络，会包含提交间隙，不能当作互斥的 kernel 活跃时间。CPU 等待、捕获、嵌套范围和退出轮装配不能再累加。相对本轮 Stiff，四组件仍需约 26.9%／27.9% 的整场降幅才达到 2×。Graph 捕获／实例化仅占 Graph 整场 .441%／.287%；单改缓存命中率无法承担缺口。

Nsight 已按旧冻结活动身份完成两次有界捕获：落球 f49 节点级、固定兔子 f40 Graph 整体。离散 VF/EE 查询、edge-triangle 安全查询值得继续定位；能量最终归约仅 .449／.179 毫秒，当前不优先重写归约。转换含库约 4.807／4.447 毫秒，MAS 准备约 10.110／9.827 毫秒，其中数值逆准备占很大部分，不能把它算为符号缓存可省成本。这两帧不是整场门槛证明，且捕获的是 Graph-only，不是四组件组合。

完整分类、CPU/GPU 包含关系和资源字段见 [Nsight 成本分析](PROFILE_COST_ANALYSIS.md)。查询 `localMemoryPerThread=0`、`localMemoryTotal` 非零不能证明动态 spill；寄存器 162／138 只是调查线索。不得以 `stack[65]` 的存在直接宣布栈是瓶颈。

## 3. 本轮实现

- Converter 的精确有序 raw key、排序映射、partition 和 canonical row/col 对结构观察；每个实际线性系统关联帧与线性编号，私有缓冲，保留完整排序与数值归约。
- BVH DCD 与 FullCCD 私有 replay 观察 AABB、叶访问、窄相、栈高水位和 warp 长尾；每帧每类至多一次，核对完整 typed tuple multiset 与 MatIndex，保留生产数据和安全判断。
- 公共 requested/resolved 配置、默认关闭入口与非法组合检查；新增两项 CPU 契约和两项 GPU  fixture，复用原 13 项守卫。
- Windows runner 使用被监管的 Job 管理本任务进程与后代；profile 使用独立冻结身份封存，保留原资源／GPU 串行保护与每身份两次捕获预算。

探针是诊断基础设施，不计为执行提速组件；canonical pair 相同也不证明 raw permutation、MAS 层级、数值因子或接触安全可复用。

## 4. 验证和质量边界

CPU 契约已通过：bench 23、local 16、profiler 17；结构分析器最终 8 类自测通过。两次首次 clean build 发现新增 GPU fixture 的 JSON 依赖、helper 命名空间和 C++17 目标属性遗漏，均做最小修复，失败日志保留。第三次独立 clean build `report56_stage1_verified_20261008` 成功，37 个原生编译单元，程序 SHA256 为 `d4da6bd5ec8544b87d0f7b77f67c6bb02a95c2afa19a24a8a6013f69d89f3902`。

17 项 CTest 全部通过，总计 20.50 秒。新结构 CPU 契约 98 项；真实 Converter GPU fixture 41 组／71,250 项检查；BVH fixture 覆盖空树、尾部、固定体和深树，复用原有停止／Graph／碰撞检查。结构、控制和旧程序历史重复的离线数值／字节核对亦已完成，见 [验证回执](STRUCTURE_QUERY_VALIDATION.json)。

原 Stiff 三次标定后，另做两次独立复核／场景（4 次完整 120 帧）。两场景都有整段材料指标超出事先冻结范围：[冻结协议](BASELINE_MATERIAL_PROTOCOL.json)、[复核统计](BASELINE_REVIEW_SUMMARY.json)。落球 p99 拉伸比超过上界约 7.64e-8／2.19e-7；固定兔子最大拉伸比超过约 1.15e-5／1.04e-3。后者约为 .104 个百分点的伸长率差异，不能解读为物理失稳或已证明算法退化。

结论是这套已冻结范围未被独立基线复现，质量协议目前无法认证；不扩大范围，不用候选重新定门槛。原 Stiff 实际速度的观测中性、独立接受路径 CCD、最终 300 帧和受控 4090 配对均待后续完成；本轮性能探索可以继续，但不推广默认组合。

## 5. 整场探针结果与下一轮

17 项 CTest 后，同程序关闭态两场景对照和两次独立探针运行均完成 120 帧，通过运行守卫。每次预算仍为 180 秒，原显存／磁盘保护保留。

| 观察 | 落球 L | 固定兔子 L |
|---|---:|---:|
| 实际线性系统数量 | 676 | 691 |
| ordered raw 命中／全部相邻机会 | 54/675 = 8.00% | 58/690 = 8.41% |
| mapping/partition 联合命中／全部相邻机会 | 54/675 = 8.00% | 58/690 = 8.41% |
| canonical pair 命中／全部相邻机会 | 176/675 = 26.07% | 185/690 = 26.81% |
| 线性探针 CPU 包含时间 ms | 300.801 | 353.575 |
| BVH 私有 replay tuple 守卫 | 240/240 | 240/240 |
| 观察到的最大待处理节点栈 | 10 | 10 |
| 关闭态整场 solver 秒（一次） | 7.635986 | 11.549531 |
| 诊断态整场 solver 秒（一次） | 11.111230 | 17.622110 |

只在可比样本中计算的 raw 条件命中率为 75%／82.86%，其分母仅为 72／70 个系统；不能把该高比例当成整场命中率。重成本窗口没有联合可比样本。canonical 相同不意味着 raw 映射或 MAS 层级／数值准备可复用。

诊断态和关闭态的方向／PCG／能量工作量并非完全相同，因此两个 solver 秒数不能直接相减作为纯探针开销。线性六阶段时间、父子范围、快照与读回分开报告，不重复加总。BVH 每类每帧只观察首次成功查询，不覆盖所有 Newton／线搜索调用；各场景初始化 DCD 有一个 linearID=0 样本，不进入 239 个求解内诊断范围的计时总和。

两场景初始位置／实际速度逐字节相同；frame1–120 有差异。布料全程位置 RMS 差约 .0173／.0136 场景长度单位，最大约 .210／.122；实际速度 RMS 差 .140／.130 长度单位/秒，最大 2.10／3.01。旧程序三次重复已有厘米量级位置分岔，但它是不同程序的历史参考，不能用来认证本次探针中性。尤其固定兔子的局部速度差大于该旧重复范围，原因待确认；未声称 Hessian／RHS／M 逐系统 checksum 中性通过。详见 [结构与查询分析](STRUCTURE_QUERY_ANALYSIS.md)。

## 6. 分支决定与下一轮方向

**结构缓存：暂缓。** 整场 raw／联合复用机会约 8%，canonical 约 26%；两个重帧中 Converter 数值工作占约 72–74%，MAS 数值准备占约 93–96%。即使算术顺序保持，也没有充分的时间加权可消除成本和未来 guard／恢复净成本支持整场 5%。不缓存旧数值因子，不因高条件命中率推广。详见 [线性分支门槛](LINEAR_BRANCH_DECISION.md)。

**stackless：暂缓。** DCD 和 FullCCD 的整场包含事件扣除已知私有 replay 子范围后，分别仍有约 1.598／.380 秒（落球）、1.708／.739 秒（固定兔子）。这些残余含遍历、距离分类、ground、计数读回和提交，不是纯遍历成本。

额外只做一次 f49 DCD EE 的 Nsight Compute 检查，从零完成 120 帧，捕获一个 kernel、一个 pass：时长 1.189248ms，动态 local load/store 为 1,213,799／615,043 sectors，occupancy 12.57%，long scoreboard 21.98%；36 blocks 分配于 40 SM，网格不足也影响占用率。原 collector 因 wide CSV 解析出现 `KeyError`，原失败回执和原脚本快照保留；仅离线修正解析，未重跑 GPU，未改驱动或时钟设置。详见 [单查询计数分析](QUERY_COUNTER_ANALYSIS.md)。

CPU-only 实际机器码检查可识别节点栈的 4 处 32 位 local 访问，以及 78 处 64 位临时量／调用邻域的 local 访问。静态指令数量不是动态耗时；不能把全部 local sectors 或 stall 归为遍历栈。完整 query ≥15%、整场净 ≥5% 和安全／质量的候选门槛仍未建立，故不贸然写 escape 视图。详见 [查询分支分析](QUERY_BRANCH_DECISION.md)。

下一轮已收窄到 **DCD EE 距离分类的 FP64 临时量和调用数据流**。先使用同状态算子检验和可消除成本预算决定是否值得实现，保留完整候选与分类公式；不直接缩栈、不改变精度／停止规则，也不恢复旧融合或 TOI 参数网格。当前证据支持优先调查该位置，尚未证明能形成 2× 所需的约 27% 整场降幅。

本轮保留默认配置，不推广未经质量和收益确认的组合。2×目标未完成；本轮完成的是有界参考／成本归因、可调试的诊断实现和分支取舍，不是全部最终验收。

## 7. 复现和证据

原始运行、轨迹、二进制、构建失败、profile report 和进程遥测保留于 `runs/report56_stage1_20261008` 与三个独立 `build/report56_stage1_*` 目录；Git 同步代码、复现脚本、统计和哈希，不同步原始轨迹／二进制／进程遥测。

关键验证命令（均已实际执行）：

```text
E:/Anaconda/envs/DL/python.exe -X utf8 tools/build_windows.py --kind active --label report56_stage1_verified_20261008 --jobs 2
E:/Anaconda/envs/DL/python.exe -X utf8 reports/report5_report6_audit_20261008/verify_fixes.py --build build/report56_stage1_verified_20261008 --out runs/report56_stage1_20261008/validation_full --stage full
E:/Anaconda/envs/DL/python.exe -X utf8 reports/report56_stage1_20261008/analyze_structure_queries.py --require-complete
E:/Anaconda/envs/DL/python.exe -X utf8 reports/report56_stage1_20261008/analyze_query_counters.py
```

GPU 原始任务目录拒绝覆盖，以上 build／run 命令不能直接复用已有输出作为一次新尝试。分析命令只读取封口证据，不启动 GPU。工具 CPU 检查与源／对象／链接／程序身份另见本目录回执和索引。
