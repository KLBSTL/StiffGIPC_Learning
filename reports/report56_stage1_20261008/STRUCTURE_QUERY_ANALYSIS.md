# 结构与查询探针观察分析

数据完整：True。本报告只分析已关闭并通过证据哈希核对的探针输出；诊断计时不作为生产速度。

## cloth_sphere7_l

状态：validated_completed。

观察到676个生产线性系统、1个owner。

| 结构 | 相邻可比 | 命中 | 可比命中率 | 命中／全部观测 | 不可比原因 |
|---|---:|---:|---:|---:|---|
| raw | 72 | 54 | 75.000% | 7.988% | {"first": 1, "shape_changed": 603} |
| mapping | 70 | 54 | 77.143% | 7.988% | {"first": 1, "shape_changed": 605} |
| partition | 70 | 68 | 97.143% | 10.059% | {"first": 1, "shape_changed": 605} |
| converted_joint | 70 | 54 | 77.143% | 7.988% | {"first": 1, "shape_changed": 605} |
| canonical_pairs | 189 | 176 | 93.122% | 26.036% | {"first": 1, "shape_changed": 486} |

canonical pair命中但mapping／partition联合未命中：122个。该计数不能解释为MAS分区或数值因子可复用。

相邻可比分母只包括同owner、同shape、无gap的相邻系统；first／shape变化／gap被排除，因此较高的可比命中率不等于大部分系统可复用。

| 六阶段 | CPU包含ms | 有效GPU event ms | readback CPU子区间ms |
|---|---:|---:|---:|
| raw_compare | 30.152 | 0.673 | 28.829 |
| raw_snapshot | 189.440 | 4.424 | 0.000 |
| converted_compare | 7.849 | 0.802 | 6.537 |
| converted_snapshot | 45.516 | 6.898 | 0.000 |
| pair_compare | 13.367 | 4.937 | 9.145 |
| pair_snapshot | 13.330 | 7.846 | 0.000 |

线性探针CPU总包含时间300.801ms，event setup 0.001ms、workspace 0.211ms均属于其子项。它们不能与六阶段CPU、GPU、readback或父级CostScope重复相加。

BVH验证240/240次私有replay；初始化无linear ID记录1条，单列保存在JSON。生产typed tuple multiset相等、production outputs/tree storage untouched标志均检查为true；这不等于独立物理质量认证。

| 查询类 | replay次数 | 样本query数 | 样本AABB测试 | 样本leaf overlap | 栈高水位 |
|---|---:|---:|---:|---:|---:|
| DCD_VF | 120 | 369960 | 19462558 | 2590172 | 7 |
| DCD_EE | 120 | 1084800 | 92658104 | 15225372 | 9 |
| FullCCD_VF | 120 | 369960 | 21059662 | 2657770 | 10 |
| FullCCD_EE | 120 | 1084800 | 96314694 | 15691508 | 10 |

| 查询类 | 样本加权AABB均值 | 最大单query AABB | 最大逐replay p99／帧 | 最大warp均值/query均值／帧 |
|---|---:|---:|---:|---:|
| DCD_VF | 52.607 | 194 | 112 / 43 | 1.717 / 101 |
| DCD_EE | 85.415 | 356 | 226 / 47 | 1.947 / 50 |
| FullCCD_VF | 56.924 | 194 | 110 / 40 | 1.673 / 27 |
| FullCCD_EE | 88.786 | 284 | 214 / 46 | 1.636 / 25 |

BVH私有replay诊断host包含时间1827.940ms，私有device峰值1587916 bytes，样本readback总量93104640 bytes。这里只观测每帧每phase一次；最大逐replay p99不是全部query的汇总p99。

| reference选择窗口 | 帧 | 当前系统数 | raw命中/可比 | joint命中/可比 | canonical命中/可比 |
|---|---|---:|---:|---:|---:|
| all_frames | 1–120 | 676 | 54/72 | 54/70 | 176/189 |
| cold_start | 1–3 | 7 | 6/6 | 6/6 | 6/6 |
| reference_linear_heavy | 41–43 | 18 | 0/0 | 0/0 | 0/0 |
| reference_ccd_line_search_heavy | 48–50 | 33 | 0/0 | 0/0 | 0/0 |

0/0表示不可比较，不能解释为0%命中。窗口由已完成120帧Graph reference预选；当前探针的系统数、query尾部与计时仅为该窗口观察，不能外推全部查询或寄存器spill。

关闭态对照：recorded。线性数值Hessian／RHS／M没有逐系统前后checksum，因此未写成bitwise unchanged通过。

同exe/source、非诊断数值配置一致。关闭态solver 7.635986s，开启诊断 11.111230s；这两次单次WDDM桌面观测不构成性能比较。

| 实际导出 | body类型 | 121帧bitwise相等 | 121帧RMS差 | 最大3D差／帧 | frame120 RMS／最大差 |
|---|---|---|---:|---:|---:|
| state | ABD:fixed_obstacle | False | 4.20347869e-34 | 3.08148791e-33 / 1 | 4.22095685e-34 / 3.08148791e-33 |
| state | FEM_2D:cloth | False | 0.017325389 | 0.209718864 / 120 | 0.039211467 / 0.209718864 |
| velocity | ABD:fixed_obstacle | False | 3.8372335e-33 | 3.08148791e-31 / 1 | 0 / 0 |
| velocity | FEM_2D:cloth | False | 0.139600703 | 2.10214843 / 47 | 0.165668331 / 0.889207366 |

state 缺失帧数0，逐字节不同帧数120，首个不同帧1。


velocity 缺失帧数0，逐字节不同帧数120，首个不同帧1。

RMS按每body的3D顶点差计算；位置使用场景长度单位，实际速度使用长度单位/秒。frame0相等及后续差异是观测事实，单次开关对照不能直接判断探针改写了数值buffer，也不能认证中性或物理质量。

历史三轮all参考：recorded_historical_different_program。

旧exe `0241124167cc9f2f770f4f7b71d2844b71f6878b7be2525e3dde0cc24bb18f29`，不同于本次程序。仅作为既有重复差异量级的诊断参考，不是本次同程序控制、中性阈值或质量认证。

| 旧all重复对 | 实际导出 | body类型 | 121帧RMS差 | 最大3D差 |
|---|---|---|---:|---:|
| r1/r2 | state | ABD:fixed_obstacle | 5.76774661e-34 | 6.16297582e-33 |
| r1/r2 | state | FEM_2D:cloth | 0.00932206722 | 0.128552652 |
| r1/r2 | velocity | ABD:fixed_obstacle | 5.26520821e-33 | 6.16297582e-31 |
| r1/r2 | velocity | FEM_2D:cloth | 0.0987401674 | 1.942731 |
| r1/r3 | state | ABD:fixed_obstacle | 4.24674034e-34 | 3.08148791e-33 |
| r1/r3 | state | FEM_2D:cloth | 0.0163794556 | 0.215173954 |
| r1/r3 | velocity | ABD:fixed_obstacle | 3.8767258e-33 | 3.08148791e-31 |
| r1/r3 | velocity | FEM_2D:cloth | 0.134416819 | 2.40647242 |
| r2/r3 | state | ABD:fixed_obstacle | 3.96478557e-34 | 6.16297582e-33 |
| r2/r3 | state | FEM_2D:cloth | 0.0157764406 | 0.225239552 |
| r2/r3 | velocity | ABD:fixed_obstacle | 3.61933749e-33 | 6.16297582e-31 |
| r2/r3 | velocity | FEM_2D:cloth | 0.128428068 | 2.81162304 |

历史与本次共同非诊断配置不一致字段：{}。实际初始拓扑、body、边界、质量、位置和速度的逐字节对应结果保留在JSON。

## cloth_fixed_bunny_l

状态：validated_completed。

观察到691个生产线性系统、1个owner。

| 结构 | 相邻可比 | 命中 | 可比命中率 | 命中／全部观测 | 不可比原因 |
|---|---:|---:|---:|---:|---|
| raw | 70 | 58 | 82.857% | 8.394% | {"first": 1, "shape_changed": 620} |
| mapping | 70 | 58 | 82.857% | 8.394% | {"first": 1, "shape_changed": 620} |
| partition | 70 | 65 | 92.857% | 9.407% | {"first": 1, "shape_changed": 620} |
| converted_joint | 70 | 58 | 82.857% | 8.394% | {"first": 1, "shape_changed": 620} |
| canonical_pairs | 204 | 185 | 90.686% | 26.773% | {"first": 1, "shape_changed": 486} |

canonical pair命中但mapping／partition联合未命中：127个。该计数不能解释为MAS分区或数值因子可复用。

相邻可比分母只包括同owner、同shape、无gap的相邻系统；first／shape变化／gap被排除，因此较高的可比命中率不等于大部分系统可复用。

| 六阶段 | CPU包含ms | 有效GPU event ms | readback CPU子区间ms |
|---|---:|---:|---:|
| raw_compare | 29.349 | 1.245 | 27.986 |
| raw_snapshot | 199.183 | 7.902 | 0.000 |
| converted_compare | 12.184 | 1.234 | 10.890 |
| converted_snapshot | 82.465 | 10.358 | 0.000 |
| pair_compare | 14.848 | 4.889 | 9.903 |
| pair_snapshot | 14.382 | 5.259 | 0.000 |

线性探针CPU总包含时间353.575ms，event setup 0.002ms、workspace 0.303ms均属于其子项。它们不能与六阶段CPU、GPU、readback或父级CostScope重复相加。

BVH验证240/240次私有replay；初始化无linear ID记录1条，单列保存在JSON。生产typed tuple multiset相等、production outputs/tree storage untouched标志均检查为true；这不等于独立物理质量认证。

| 查询类 | replay次数 | 样本query数 | 样本AABB测试 | 样本leaf overlap | 栈高水位 |
|---|---:|---:|---:|---:|---:|
| DCD_VF | 120 | 1943280 | 113801706 | 12781326 | 7 |
| DCD_EE | 120 | 5792760 | 504569794 | 74816718 | 10 |
| FullCCD_VF | 120 | 1943280 | 130347050 | 13242041 | 9 |
| FullCCD_EE | 120 | 5792760 | 555843620 | 77645116 | 10 |

| 查询类 | 样本加权AABB均值 | 最大单query AABB | 最大逐replay p99／帧 | 最大warp均值/query均值／帧 |
|---|---:|---:|---:|---:|
| DCD_VF | 58.562 | 210 | 110 / 56 | 1.616 / 49 |
| DCD_EE | 87.104 | 280 | 162 / 52 | 1.463 / 57 |
| FullCCD_VF | 67.076 | 188 | 138 / 45 | 1.569 / 34 |
| FullCCD_EE | 95.955 | 326 | 210 / 45 | 1.416 / 24 |

BVH私有replay诊断host包含时间4783.325ms，私有device峰值6948796 bytes，样本readback总量495106560 bytes。这里只观测每帧每phase一次；最大逐replay p99不是全部query的汇总p99。

| reference选择窗口 | 帧 | 当前系统数 | raw命中/可比 | joint命中/可比 | canonical命中/可比 |
|---|---|---:|---:|---:|---:|
| all_frames | 1–120 | 691 | 58/70 | 58/70 | 185/204 |
| cold_start | 1–3 | 7 | 6/6 | 6/6 | 6/6 |
| reference_linear_heavy | 39–41 | 22 | 0/0 | 0/0 | 1/3 |
| reference_ccd_line_search_heavy | 57–59 | 18 | 0/0 | 0/0 | 0/0 |

0/0表示不可比较，不能解释为0%命中。窗口由已完成120帧Graph reference预选；当前探针的系统数、query尾部与计时仅为该窗口观察，不能外推全部查询或寄存器spill。

关闭态对照：recorded。线性数值Hessian／RHS／M没有逐系统前后checksum，因此未写成bitwise unchanged通过。

同exe/source、非诊断数值配置一致。关闭态solver 11.549531s，开启诊断 17.622110s；这两次单次WDDM桌面观测不构成性能比较。

| 实际导出 | body类型 | 121帧bitwise相等 | 121帧RMS差 | 最大3D差／帧 | frame120 RMS／最大差 |
|---|---|---|---:|---:|---:|
| state | ABD:fixed_obstacle | False | 1.78736313e-17 | 1.1443917e-16 / 1 | 1.79479502e-17 / 1.1443917e-16 |
| state | FEM_2D:cloth | False | 0.0136109915 | 0.122294792 / 91 | 0.0203369353 / 0.0851287418 |
| velocity | ABD:fixed_obstacle | False | 1.63163184e-16 | 1.1443917e-14 / 1 | 0 / 0 |
| velocity | FEM_2D:cloth | False | 0.129893437 | 3.00570248 / 54 | 0.134807533 / 0.805902312 |

state 缺失帧数0，逐字节不同帧数120，首个不同帧1。


velocity 缺失帧数0，逐字节不同帧数120，首个不同帧1。

RMS按每body的3D顶点差计算；位置使用场景长度单位，实际速度使用长度单位/秒。frame0相等及后续差异是观测事实，单次开关对照不能直接判断探针改写了数值buffer，也不能认证中性或物理质量。

历史三轮all参考：recorded_historical_different_program。

旧exe `0241124167cc9f2f770f4f7b71d2844b71f6878b7be2525e3dde0cc24bb18f29`，不同于本次程序。仅作为既有重复差异量级的诊断参考，不是本次同程序控制、中性阈值或质量认证。

| 旧all重复对 | 实际导出 | body类型 | 121帧RMS差 | 最大3D差 |
|---|---|---|---:|---:|
| r1/r2 | state | ABD:fixed_obstacle | 4.71507205e-18 | 1.38777878e-17 |
| r1/r2 | state | FEM_2D:cloth | 0.0139985484 | 0.0837336985 |
| r1/r2 | velocity | ABD:fixed_obstacle | 4.3042522e-17 | 1.38777878e-15 |
| r1/r2 | velocity | FEM_2D:cloth | 0.123527375 | 1.79462614 |
| r1/r3 | state | ABD:fixed_obstacle | 1.42531483e-17 | 5.55111512e-17 |
| r1/r3 | state | FEM_2D:cloth | 0.0133582701 | 0.0948522926 |
| r1/r3 | velocity | ABD:fixed_obstacle | 1.30112847e-16 | 5.55111512e-15 |
| r1/r3 | velocity | FEM_2D:cloth | 0.123461808 | 1.6675086 |
| r2/r3 | state | ABD:fixed_obstacle | 1.50127992e-17 | 5.7219585e-17 |
| r2/r3 | state | FEM_2D:cloth | 0.0123314768 | 0.074219031 |
| r2/r3 | velocity | ABD:fixed_obstacle | 1.3704748e-16 | 5.7219585e-15 |
| r2/r3 | velocity | FEM_2D:cloth | 0.125006123 | 1.34737452 |

历史与本次共同非诊断配置不一致字段：{}。实际初始拓扑、body、边界、质量、位置和速度的逐字节对应结果保留在JSON。

## 结论边界

Probe timings include diagnostic synchronization and replay; they are not production speed or a removable-cost estimate.
Readback CPU is contained in compare CPU. GPU intervals overlap completion waits. Parent CostScopes include diagnostic children and are never added to probe costs.
Canonical pair equality does not prove reusable MAS topology, mapping, partition, numerical factorization or contact safety.
BVH counters describe one sampled replay per frame/phase. Initialization without linear ID is separate. No extrapolation to all queries, register spill or occupancy.
Observer-off is optional and absent results stay pending. On/off RMS and maxima describe actual output differences, not physical errors or a diagnostic neutrality certificate.
Old-executable all-arm three-repeat differences are historical diagnostic context only, not a same-program control or predeclared neutrality threshold. No baseline velocity reconstruction, quality tolerance change or prior quality-gate promotion.

复现：`E:/Anaconda/envs/DL/python.exe reports/report56_stage1_20261008/analyze_structure_queries.py`；可用`--observer-off-dir`接入后续同程序关闭态目录，`--require-complete`要求两场景完整有效。
