# M0 第一阶段：本机基线、成本数据与精度口径

日期：2026-10-14。分支 `research/stiff-innovation-20261014`，native 起点 `a4b83c0`。

**本阶段已完成有界证据底座：两种同机构建身份、历史数据逐项核验、代表性短测、共同输入恢复、direct/compensated消融及真实线性残差核查。下一步可以进入P2-A的执行成本调查。数值路线的全场精度与物理认证尚未通过，不能据本报告宣称同质量2×。**

## 1. 最关键的确定结论

1. **res=.03并不保证最终残差≤.03。**当前movement退出优先，res只保护额外的累计退出。Bunny共同第14帧，gated第一轮以movement退出时相对残差为0.072074819，compensated为0.051585627，都超过.03。保留这项原有语义，本轮没有悄悄改成全局残差收敛。
2. **rho=1e-4并不是L2相对残差1e-4。**Bunny第14帧首方向的固定系统，Graph的显式 `||b-Ax||/||b||` 为0.052827437，固定MAS参考下的加权相对残差为0.009655143。共同精度对照必须同时使用显式残差和固定参考预条件作用。
3. **当前门控量不是物理速度误差。**它取ABD广义梯度标量和自由FEM梯度分量绝对最大值，再除冻结参考；没有质量逆/Jacobian/时间步的统一物理映射。Bunny的固定ABD梯度已清零，实测abd_inf=0，不能将本例归因为“固定ABD污染门控”。
4. **所测状态没有体现历史补偿的独立收益。**40次共同输入单步的同一实际前缀上，均未出现history_exit_veto。Mixed两轮direct/compensated均首次在第11次接受更新满足条件；Bunny/Boxes选定前缀主要走movement退出。独立运行的PCG次数浮动不能直接归为历史预算贡献。
5. **Mixed翻转属于现有基线问题。**本机冻结B0、B1-LEGACY和B1-MOV均从第13帧出现非正FEM几何J。SNK1能量对负J仍是有限多项式，IPC路径中的体积injective限步调用被注释；FullCCD不等于每个体积元素保持正J。本轮不改材料，不将翻转单独归因于TOI，也不认证其物理可接受性。

## 2. 实际修改与实现问题处理

唯一修改的既有源码文件为 `tools/local/windows_runner.py`：以前所有请求固定写 `from_zero=True`，连检查点恢复也如此；现在从实际 `GIPC_CHECKPOINT_LOAD` 环境生成 `from_zero` 和 `checkpoint_load`。

主矩阵30次恢复保留原始错误标记和全部证据，原runner逐字节归档为 `runs/innovation_m0_20261014/windows_runner_at_run.py`。分析以native `frame_context.json`、完整恢复报告和真实初始位置/速度为准。修正后新增的10次恢复与1次前缀通过端到端origin门槛，未重写旧测量。

新增工具仅服务M0：离线一致性分析、构建外围、有限本机控制、质量分析及独立核验。`StiffGIPC/`、`baseline/`、`Assets/`、根CMake和原native测试均未修改；没有新增加速候选或更换默认停止规则。

构建过程遇到并处理了三项实际问题：

- 沙箱中MSVC探测报C1902，保存失败目录，用正常本机构建环境在全新B1目录重建。
- B0有大小写不同的同名CUDA源，外围CMake为每个源分配独立对象名；35个原native TU与链接逐项核对，冻结目录不改。
- B0生成项目的源路径是相对项目目录，核验器先中止；修正外围路径解析后继续尚未编译的配置目录。B1原构建驱动的全目录保护曾因本轮临时Python测试触发；最终将实验测试放到tools/bench，全部预构建文件与当前文件一致，并用实际TU/对象/链接/命令和native输入重新独立封存。没有把失败的构建身份检查直接当作完成。

## 3. 确切版本、配置与资源

| 身份 | 源与功能 | exe SHA256 |
|---|---|---|
| B0 | 本提交冻结baseline；原movement+累计beta退出；原MAS；host PCG | 54f7364fd6e69d65787b244b11e61e642c2b1edc3ecf88133a1d728eb5481ea3 |
| B1 | 本提交当前执行组合，37个native TU；Graph K1/refit/batch/reuse/direct FEM/pivot sync | 62fa83baa3805eff40e3b1d79b434276fd510f6e8b9de1837a5da744423b8541 |

B0本来就有累计退出，不是“关闭累计”的Host消融。B0不导出实际速度、resolved-config或检查点，这些能力未伪造。

本机RTX3070 Laptop 8GB，驱动581.57，CUDA13.0，VS2022/MSVC14.44，SM86。核验两份Release CUDA设置和模型宏一致，仅各自源码/资产/输出路径不同；两方使用USE_SNK1、USE_FRICTION、USE_QUADRATIC_BENDING、ADAPTIVE_KAPPA。scene JSON里的 `friction_compiled=false` 是历史描述字段，不能覆盖实际编译宏；当前配置中的摩擦系数为原场景值。

dt=.04、rho=1e-4、movement=.01、累计=.01、min更新6、res=.03、legacy MAS、FullCCD，forcing off。参考和fixed诊断的精度变化单独命名。GPU任务串行；每个场景进程120s硬上限，资源不足/非有限/迭代上限均算失败。

实际电源方案为野兽模式，GUID `52521609-efc9-4268-b9ba-67dea73f18b2`。464个普通矩阵采样及各次运行前后核对均一致。GPU未锁频，WDDM桌面/OEM散热未独立控制，也没有用事件日志证明未采到的瞬时切换，因此时间为诊断值。

## 4. 运行范围和代表性

共61次native运行、257帧：主矩阵49次229帧，Boxes补充11次27帧，固定系统捕获1次1帧。普通矩阵合计solver时间131.598秒，逐次wall合计210.516秒；构建时间和独立fixed诊断另列。

- smoke：B0/B1各2帧。
- 三类前缀：Boxes20帧、Mixed15帧、Bunny14帧；各B0、B1-LEGACY、B1-MOV。
- 同输入单步：三场景五规则各2轮，全部用本轮同二进制完整检查点。
- 参考校准：Bunny6帧，内层/外层四种精度各2轮。
- Boxes第20帧只有3次方向，不能充分测试累计规则；据未改基线前20帧的最大N、并列取最早的预登记选择，补第17帧。补充前缀在候选实现之前冻结，非筛选新算法有利状态。
- Bunny第14帧首方向保存全A/b/MAS，现有fixed probe检查system_unchanged及primary_restored_bitwise，独立CPU重算真残差。

三场景B0/B1的初始位置、拓扑、边界、质量、body ID全部逐字节一致。40次普通共同输入恢复及1次fixed恢复通过物理context、所有设备buffer、帧号、位置和真实速度检查。冷重建缓存与热继续仍是不同实验单位。

## 5. 前缀成本表

| Scene/帧数 | 方法 | solver(s) | N总 | P总 |
|---|---|---:|---:|---:|
| Boxes/20 | B0 | 4.389205 | 120 | 4599 |
| Boxes/20 | B1-LEGACY | 3.769301 | 130 | 4676 |
| Boxes/20 | B1-MOV | 3.979265 | 141 | 5076 |
| Mixed/15 | B0 | 11.085844 | 181 | 14312 |
| Mixed/15 | B1-LEGACY | 8.286797 | 177 | 13592 |
| Mixed/15 | B1-MOV | 15.137553 | 327 | 22154 |
| Bunny/14 | B0 | 3.582462 | 87 | 12241 |
| Bunny/14 | B1-LEGACY | 2.212984 | 91 | 12914 |
| Bunny/14 | B1-MOV | 7.225714 | 292 | 45601 |

这是单次短前缀观察，两方法的后续轨迹已分叉；不能把总时间比全归为执行组件或认证同质量加速。N是方向求解次数，P是PCG迭代总量。分项在 `analysis/local_runs.csv`；原版未提供的逐Newton分项留空，未伪造零值。

## 6. 困难共同起点：两轮中位数

| Scene/物理帧 | 规则 | N | P | t(ms) | 对两条rho1e-6参考最坏cloth RMS(mm) |
|---|---|---:|---:|---:|---:|
| Boxes/17 | Movement | 24 | 533 | 680.030 | .677677 |
| Boxes/17 | Legacy TOI | 22 | 475 | 612.767 | .781924 |
| Boxes/17 | Direct gate | 24 | 533 | 666.900 | .677639 |
| Boxes/17 | Compensated | 24 | 533 | 671.305 | .677656 |
| Mixed/15 | Movement | 14 | 1429.5 | 983.038 | .069407 |
| Mixed/15 | Legacy TOI | 8 | 961 | 640.406 | 1.591338 |
| Mixed/15 | Direct gate | 11 | 1233.5 | 826.740 | .475650 |
| Mixed/15 | Compensated | 11 | 1230.5 | 812.680 | .474875 |
| Bunny/14 | Movement | 31 | 4528.5 | 932.376 | .754578 |
| Bunny/14 | Legacy TOI | 11 | 1858 | 424.861 | 11.367902 |
| Bunny/14 | Direct gate | 30 | 4419.5 | 920.231 | .756967 |
| Bunny/14 | Compensated | 32 | 4713.5 | 962.415 | .817026 |

rho1e-6仍保留外层.01；这些是位置差异，不是精确物理误差。对应cloth参考自身RMS分歧：Boxes .009053mm、Mixed .00004483mm、Bunny .814061mm。Bunny候选的约.75–.82mm差异处于参考自身分歧范围，不作独立质量定责。两轮中位数出现小数P不代表单次迭代为小数。速度和全部分区误差在summary.json。

## 7. 参考精度与共同线性口径

Bunny从相同初始状态运行6帧（物理时间.24s）：

| rho/outer | 对最严两轮参考最坏cloth RMS(mm) | 自身两轮cloth RMS(mm) |
|---|---:|---:|
| 1e-4/.01 | 5.428068 | .055309 |
| 1e-6/.01 | 5.938024 | .351311 |
| 1e-6/.001 | 1.293196 | .002187 |
| 1e-8/.0001 | .000881 | .000881 |

只收紧内层不形成更可信的外层解；本例甚至产生更大的重复分歧。收紧外层显著改善重复性，但.001与.0001的结果仍差约1.29mm。最严版本并未与再严一级/独立非线性求解器比对，所以不能把.0001结果认定为精确参考。本次把此限制记入后续数值验收条件，未为了“通过”放宽阈值。

同一Bunny第14帧固定A，保存第一重复的解后重算：

| 方法/rho | 两次迭代数 | 真L2相对残差 | 共同MAS加权相对残差 |
|---|---|---:|---:|
| Host/1e-4 | 225,224 | .050339478 | .009741097 |
| Graph/1e-4 | 224,224 | .052827437 | .009655143 |
| Host/1e-16 | 790,791 | 5.3958e-8 | 1.0488e-8 |
| Graph/1e-16 | 789,791 | 5.1850e-8 | 9.7112e-9 |

CPU与native显式残差核查一致；B_ref用保存的FP32 MAS系数，CPU归约顺序与GPU不同。rho_stop来自旧rho的兼容停止语义，也不能当作当前显式残差。这里的Graph是起点已有实现，不是本轮新增创新；固定算子时间也不替代完整非线性时间。

## 8. Mixed的J与指标实现边界

| 方法/15帧 | 首次非正J物理帧 | 最小FEM J | 单帧最多非正元素 |
|---|---:|---:|---:|
| B0 | 13 | -.645427 | 3 |
| B1-LEGACY | 13 | -.327865 | 2 |
| B1-MOV | 13 | -.508039 | 5 |

J使用 `det(Ds_current)/det(Ds_rest)`，分母取场景未变形初始状态。尤其对冷检查点，不能把已经变形/翻转的checkpoint当作rest：否则翻转状态相对自己会得到J=1。新增几何契约测试专门覆盖这个错误，后处理不复用会重置分母的旧通用指标。

最终复核还修正了固定ABD的指标分类：其point BoundaryType可能仍为Free，约束来自body层，不能放进自由顶点RMS稀释误差。当前三场景ABD各自全fixed或全free，按实际scene/body约束确认；若遇混合fixed/free ABD，工具明确要求导出映射后再计算。三项几何合同通过。修正后的最终分析为analysis_v2；初版analysis保留为被替代记录，表中的dynamic-cloth结果没有变化。

SNK1源码能量为 `volume/2 * [mu*(||F||²-3)+lambda*(detF-1-mu/lambda)²]`，对负detF仍有限；这是模型域的事实，不是“翻转物理正确”的证明。独立路径CCD、真实物理参考和长期稳定性仍待认证。

## 9. 历史成本分析与下一步有限选择

离线数据共126条完整轨迹、13230帧。逐Newton记录与逐帧N/P、相位和帧号全部核对；AutoDL36条与Windows90条分开统计。指标用前100帧，事件从各自完整105帧选取。

| 当前MOV历史数据 | Boxes首次无回退比例 | Bunny | Mixed |
|---|---:|---:|---:|
| AutoDL | 95.088% | 97.567% | 98.132% |
| 本机五版 | 95.098% | 97.498% | 98.331% |

这里“首次无回退”只指energy/intersection backtracks均为0，起始alpha已可能被CCD/CFL截短，不能解释为全步alpha=1。当前数据的intersection backtracks全为0；自然测试未覆盖相交回退，不能代替失败fixture。

起始geometric alpha、trial energy margin、容量增长和真正PCG内部kernel分解缺少直接观测，明确记为unavailable，未填零。已有alpha/P长尾足以选状态；细分CCD/能量限制及P2收益仍需小范围observer/同状态成本测量。

下一步仅进入 **M1.1/P2-A**：逐消费者核查BVH host scene、ABD两项能量读回和stream等待，先量完整线搜索成本。Bunny14、Mixed15、Boxes17作为困难回放，Boxes20作为便宜/未激活门控的退化对照。只有目标阶段净减少约15%、有整场5%空间才扩大P2-B；当前不开始完整GPU回溯、P1新控制器或P5生产补偿。

## 10. 验证、文件与尚未认证项

14个新增CPU证据/运行/几何契约测试和4个已有native CPU测试通过；完整独立核验命令：

```powershell
& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/bench/verify_m0.py --analysis-root reports/innovation_m0_20261014/analysis_v2
```

已退出0，核验61次运行/257帧、40次共同输入恢复、fixed恢复、编译宏/选项、所有源码/对象/链接/二进制身份、N/P与相位、power采样和元数据修正。收据：[data_validation.json](analysis_v2/data_validation.json)。

关键文件：

- [offline_summary.json](offline/offline_summary.json)：42个场景/规则单元的分布、字段覆盖、事件选择及原数据SHA。
- [summary.json](analysis_v2/summary.json)：正确固定ABD/自由顶点分区的位置/真实速度、重复差异、共同线性残差、参考限制。
- [local_runs.csv](analysis_v2/local_runs.csv)：60次普通运行的实际N/P/分项/总时间；fixed诊断独立保存。
- [geometric_quality.csv](analysis_v2/geometric_quality.csv)：每个普通帧的真实rest基准J/伸长/固定漂移。
- [same_prefix_gate_audit.csv](analysis_v2/same_prefix_gate_audit.csv)：40次前缀的两个门控谓词，未假定不同运行前缀相同。
- [origin_audit.csv](analysis_v2/origin_audit.csv)：旧记录误标与native真实帧上下文。
- [SOURCE_AUDIT.md](SOURCE_AUDIT.md)：逐行检查范围、约束、排除的问题与后续风险。

所有原始运行在本工作目录runs/，旧分支和旧结果未覆盖，全部拥有的native Job已关闭，未上传GitHub。此阶段完成的是可靠的局部实验底座；全场同质量2×、非线性参考收敛、体积翻转的物理接受标准和独立碰撞认证仍未证实。
