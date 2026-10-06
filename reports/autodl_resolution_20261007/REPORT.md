# 布料场景分辨率诊断：L/M × 四配置

已记录 16/16 项，完整硬检查通过 16 项。每项从零运行 100 帧；每个配置仅一次，无置信区间，不认证同质量 2×。

本轮在运行前依据历史布料收益选取悬挂布料与布料落球（固定球）场景，各测 L/M；属于有选择的诊断样本，不能外推全部场景。材料只与本次单个 Stiff 样本列原始差异，不新增或放宽认证门槛。

## 四段速度比

每行均使用对应两臂的实际时间。节省比例为 1−1/ratio，节省秒数为 reference−candidate；负数表示回退。一个场景任一臂失败或不完整，该场景不计算速度比。

| 场景 | 比较 | 求解比 | 求解节省% | 求解节省秒 | core event比 | wall比 |
|---|---|---:|---:|---:|---:|---:|
| hang_l | original_stiff/ipc_host | 1.0127 | 1.25 | 0.0244 | 1.0171 | 1.0268 |
| hang_l | ipc_host/ipc_graph | 1.2124 | 17.52 | 0.3366 | 1.2163 | 0.9573 |
| hang_l | ipc_graph/combined_graph | 1.1004 | 9.13 | 0.1447 | 1.1036 | 1.2915 |
| hang_l | original_stiff/combined_graph | 1.3511 | 25.98 | 0.5057 | 1.3653 | 1.2695 |
| sphere_l | original_stiff/ipc_host | 0.9446 | -5.86 | -0.2463 | 0.9465 | 0.8883 |
| sphere_l | ipc_host/ipc_graph | 1.0926 | 8.47 | 0.3767 | 1.0937 | 1.1251 |
| sphere_l | ipc_graph/combined_graph | 1.1511 | 13.12 | 0.5340 | 1.1527 | 1.1333 |
| sphere_l | original_stiff/combined_graph | 1.1880 | 15.82 | 0.6645 | 1.1932 | 1.1327 |
| hang_m | original_stiff/ipc_host | 1.0764 | 7.10 | 0.3095 | 1.0792 | 1.0095 |
| hang_m | ipc_host/ipc_graph | 1.1878 | 15.81 | 0.6403 | 1.1898 | 1.1110 |
| hang_m | ipc_graph/combined_graph | 1.0809 | 7.49 | 0.2553 | 1.0816 | 0.9937 |
| hang_m | original_stiff/combined_graph | 1.3820 | 27.64 | 1.2051 | 1.3889 | 1.1144 |
| sphere_m | original_stiff/ipc_host | 1.0170 | 1.67 | 0.1114 | 1.0187 | 0.9686 |
| sphere_m | ipc_host/ipc_graph | 1.1337 | 11.79 | 0.7712 | 1.1358 | 1.0661 |
| sphere_m | ipc_graph/combined_graph | 1.0579 | 5.47 | 0.3157 | 1.0577 | 1.0137 |
| sphere_m | original_stiff/combined_graph | 1.2198 | 18.02 | 1.1983 | 1.2238 | 1.0468 |

## 每次运行与阶段成本

时间单位秒。求解为 CPU 同步包络；core event 在其中，wall 还含加载及不完全对称的导出成本，三者不能相加。linear 包含矩阵转换、MAS 准备/应用、PCG 与解分发；五阶段不是完整 wall 分解。

| 场景 | 配置 | 状态 | 求解 | core event | wall | 装配 | linear | CCD | 线搜索 | 状态更新 | 线性次数 | PCG次数 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hang_l | original_stiff | completed | 1.9461 | 1.9339 | 2.9737 | 0.1494 | 1.0072 | 0.2633 | 0.4813 | 0.0035 | 442 | 14024 |
| hang_l | ipc_host | completed | 1.9217 | 1.9014 | 2.8960 | 0.1486 | 0.9770 | 0.2587 | 0.4771 | 0.0035 | 442 | 14022 |
| hang_l | ipc_graph | completed | 1.5851 | 1.5633 | 3.0253 | 0.1485 | 0.6363 | 0.2582 | 0.4767 | 0.0035 | 442 | 14019 |
| hang_l | combined_graph | completed | 1.4404 | 1.4165 | 2.3425 | 0.1516 | 0.6511 | 0.1814 | 0.3873 | 0.0036 | 442 | 14024 |
| sphere_l | combined_graph | completed | 3.5350 | 3.5050 | 4.6866 | 0.5068 | 1.2746 | 0.3110 | 1.3473 | 0.0080 | 558 | 24343 |
| sphere_l | ipc_graph | completed | 4.0690 | 4.0401 | 5.3114 | 0.5308 | 1.3408 | 0.4547 | 1.6515 | 0.0085 | 589 | 25948 |
| sphere_l | ipc_host | completed | 4.4457 | 4.4186 | 5.9756 | 0.4971 | 1.9248 | 0.4197 | 1.5129 | 0.0082 | 558 | 23708 |
| sphere_l | original_stiff | completed | 4.1994 | 4.1824 | 5.3084 | 0.4716 | 1.7716 | 0.4027 | 1.4818 | 0.0070 | 546 | 23408 |
| hang_m | original_stiff | completed | 4.3596 | 4.3464 | 5.3171 | 0.3582 | 2.7219 | 0.3696 | 0.8386 | 0.0066 | 538 | 33448 |
| hang_m | ipc_host | completed | 4.0502 | 4.0273 | 5.2668 | 0.3358 | 2.4955 | 0.3478 | 0.7767 | 0.0061 | 510 | 31569 |
| hang_m | ipc_graph | completed | 3.4099 | 3.3848 | 4.7408 | 0.3568 | 1.7650 | 0.3657 | 0.8247 | 0.0065 | 533 | 33246 |
| hang_m | combined_graph | completed | 3.1545 | 3.1294 | 4.7710 | 0.3517 | 1.7564 | 0.2631 | 0.6884 | 0.0060 | 527 | 32936 |
| sphere_m | combined_graph | completed | 5.4528 | 5.4229 | 7.2641 | 0.6542 | 2.4420 | 0.4505 | 1.8048 | 0.0078 | 547 | 39247 |
| sphere_m | ipc_graph | completed | 5.7685 | 5.7358 | 7.3634 | 0.6632 | 2.4320 | 0.5520 | 2.0172 | 0.0082 | 553 | 38433 |
| sphere_m | ipc_host | completed | 6.5397 | 6.5149 | 7.8503 | 0.6220 | 3.3854 | 0.5175 | 1.9258 | 0.0072 | 546 | 38342 |
| sphere_m | original_stiff | completed | 6.6511 | 6.6368 | 7.6039 | 0.6330 | 3.4559 | 0.5236 | 1.9710 | 0.0070 | 555 | 39106 |

实际组合：FullCCD refit、批量能量、能量复用、普通 BVH refit（周期 8）。MAS 静态拓扑复用关闭；没有单项消融，不能把组合比归因某一组件。每次运行的实际 execution、回退原因和 resolved 开关保存在 JSON；线性次数/PCG 与轨迹变化会进入时间差。

## 原生碰撞活动观测

表中为帧末原生窄相对非零帧数与首次观测帧，用于描述 barrier 活动线索，不是独立接触/非零力认证，也不保证捕获帧内最早活动。geometric_contact 使用更紧距离分类，其 false 不能解释为无 barrier；不以 newton.active_pairs 的零值判断无接触。缺字段保留未知。

| 场景 | 配置 | self非零帧 | ground非零帧 | 任一非零帧 | 首次观测帧 | 数据完整 |
|---|---|---:|---:|---:|---:|---|
| hang_l | original_stiff | 10 | 0 | 10 | 41 | True |
| hang_l | ipc_host | 10 | 0 | 10 | 41 | True |
| hang_l | ipc_graph | 10 | 0 | 10 | 41 | True |
| hang_l | combined_graph | 10 | 0 | 10 | 41 | True |
| sphere_l | combined_graph | 79 | 0 | 79 | 22 | True |
| sphere_l | ipc_graph | 79 | 0 | 79 | 22 | True |
| sphere_l | ipc_host | 79 | 0 | 79 | 22 | True |
| sphere_l | original_stiff | 79 | 0 | 79 | 22 | True |
| hang_m | original_stiff | 63 | 0 | 63 | 38 | True |
| hang_m | ipc_host | 63 | 0 | 63 | 38 | True |
| hang_m | ipc_graph | 63 | 0 | 63 | 38 | True |
| hang_m | combined_graph | 63 | 0 | 63 | 38 | True |
| sphere_m | combined_graph | 79 | 0 | 79 | 22 | True |
| sphere_m | ipc_graph | 79 | 0 | 79 | 22 | True |
| sphere_m | ipc_host | 79 | 0 | 79 | 22 | True |
| sphere_m | original_stiff | 79 | 0 | 79 | 22 | True |

## 材料与分体状态差异

下列单次材料值与 JSON 中的带符号差异全部保留。不存在 FEM/ABD 的指标不填零最小 J；p99 是各帧空间 p99 的全段最大值。非正单元帧累计不是独立翻转事件数。

| 场景 | 配置 | 最大拉伸 | p99拉伸 | 固定点漂移m | FEM最小J | FEM非正峰值 | FEM负体积峰值 | ABD最小J |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| hang_l | original_stiff | 1.12887621 | 1.03240934 | 0.00000000 | 未测 | 0.00000000 | 0.00000000 | 未测 |
| hang_l | ipc_host | 1.12887620 | 1.03240933 | 0.00000000 | 未测 | 0.00000000 | 0.00000000 | 未测 |
| hang_l | ipc_graph | 1.12887622 | 1.03240934 | 0.00000000 | 未测 | 0.00000000 | 0.00000000 | 未测 |
| hang_l | combined_graph | 1.12887621 | 1.03240934 | 0.00000000 | 未测 | 0.00000000 | 0.00000000 | 未测 |
| sphere_l | combined_graph | 1.01897665 | 1.01414569 | 0.00000000 | 未测 | 0.00000000 | 0.00000000 | 1.00000000 |
| sphere_l | ipc_graph | 1.01975736 | 1.01415826 | 0.00000000 | 未测 | 0.00000000 | 0.00000000 | 1.00000000 |
| sphere_l | ipc_host | 1.01907810 | 1.01417694 | 0.00000000 | 未测 | 0.00000000 | 0.00000000 | 1.00000000 |
| sphere_l | original_stiff | 1.01895304 | 1.01417650 | 0.00000000 | 未测 | 0.00000000 | 0.00000000 | 1.00000000 |
| hang_m | original_stiff | 1.17914840 | 1.02833498 | 0.00000000 | 未测 | 0.00000000 | 0.00000000 | 未测 |
| hang_m | ipc_host | 1.17909142 | 1.02834607 | 0.00000000 | 未测 | 0.00000000 | 0.00000000 | 未测 |
| hang_m | ipc_graph | 1.17909897 | 1.02833906 | 0.00000000 | 未测 | 0.00000000 | 0.00000000 | 未测 |
| hang_m | combined_graph | 1.17910117 | 1.02834570 | 0.00000000 | 未测 | 0.00000000 | 0.00000000 | 未测 |
| sphere_m | combined_graph | 1.02213974 | 1.01469909 | 0.00000000 | 未测 | 0.00000000 | 0.00000000 | 1.00000000 |
| sphere_m | ipc_graph | 1.02196833 | 1.01454996 | 0.00000000 | 未测 | 0.00000000 | 0.00000000 | 1.00000000 |
| sphere_m | ipc_host | 1.02364955 | 1.01479917 | 0.00000000 | 未测 | 0.00000000 | 0.00000000 | 1.00000000 |
| sphere_m | original_stiff | 1.02371984 | 1.01457717 | 0.00000000 | 未测 | 0.00000000 | 0.00000000 | 1.00000000 |

分体差异为全段最大帧 RMS / 最大顶点差。原版未导出真实速度，不从位置差分伪造速度；缺缓存明确记为未测。

| 场景 | 比较 | 量 | 分体 | max-frame RMS | max vertex | 帧数 |
|---|---|---|---|---:|---:|---:|
| hang_l | original_stiff/ipc_host | position | cloth | 1.01759456e-06 | 7.43473132e-06 | 101 |
| hang_l | ipc_host/ipc_graph | position | cloth | 1.59985184e-06 | 2.48276001e-05 | 101 |
| hang_l | ipc_host/ipc_graph | velocity | cloth | 7.49147958e-05 | 0.0021815612 | 101 |
| hang_l | ipc_graph/combined_graph | position | cloth | 2.59340581e-06 | 2.97926883e-05 | 101 |
| hang_l | ipc_graph/combined_graph | velocity | cloth | 8.83324284e-05 | 0.00107245005 | 101 |
| hang_l | original_stiff/combined_graph | position | cloth | 3.06770299e-06 | 3.8616063e-05 | 101 |
| sphere_l | original_stiff/ipc_host | position | cloth | 0.0179741109 | 0.0987206504 | 101 |
| sphere_l | original_stiff/ipc_host | position | ABD | 9.97013806e-34 | 6.16297582e-33 | 101 |
| sphere_l | ipc_host/ipc_graph | position | cloth | 0.0184148012 | 0.0960351926 | 101 |
| sphere_l | ipc_host/ipc_graph | position | ABD | 1.02076281e-33 | 6.16297582e-33 | 101 |
| sphere_l | ipc_host/ipc_graph | velocity | cloth | 0.340359046 | 2.87660783 | 101 |
| sphere_l | ipc_host/ipc_graph | velocity | ABD | 1.02076281e-31 | 6.16297582e-31 | 101 |
| sphere_l | ipc_graph/combined_graph | position | cloth | 0.0236573498 | 0.118871362 | 101 |
| sphere_l | ipc_graph/combined_graph | position | ABD | 5.53129199e-34 | 6.16297582e-33 | 101 |
| sphere_l | ipc_graph/combined_graph | velocity | cloth | 0.274966921 | 2.72230494 | 101 |
| sphere_l | ipc_graph/combined_graph | velocity | ABD | 5.53129199e-32 | 6.16297582e-31 | 101 |
| sphere_l | original_stiff/combined_graph | position | cloth | 0.0266419657 | 0.126191439 | 101 |
| sphere_l | original_stiff/combined_graph | position | ABD | 7.01062774e-34 | 1.23259516e-32 | 101 |
| hang_m | original_stiff/ipc_host | position | cloth | 0.000745321246 | 0.0105683628 | 101 |
| hang_m | ipc_host/ipc_graph | position | cloth | 0.00106055245 | 0.0103286084 | 101 |
| hang_m | ipc_host/ipc_graph | velocity | cloth | 0.0154658118 | 0.466643704 | 101 |
| hang_m | ipc_graph/combined_graph | position | cloth | 0.000247373439 | 0.00494118306 | 101 |
| hang_m | ipc_graph/combined_graph | velocity | cloth | 0.010273742 | 0.426725987 | 101 |
| hang_m | original_stiff/combined_graph | position | cloth | 0.000579385544 | 0.00536693734 | 101 |
| sphere_m | original_stiff/ipc_host | position | cloth | 0.016252533 | 0.0850514894 | 101 |
| sphere_m | original_stiff/ipc_host | position | ABD | 0 | 0 | 101 |
| sphere_m | ipc_host/ipc_graph | position | cloth | 0.0206343126 | 0.0920042473 | 101 |
| sphere_m | ipc_host/ipc_graph | position | ABD | 2.99494149e-34 | 3.08148791e-33 | 101 |
| sphere_m | ipc_host/ipc_graph | velocity | cloth | 0.192130933 | 2.35798153 | 101 |
| sphere_m | ipc_host/ipc_graph | velocity | ABD | 2.99494149e-32 | 3.08148791e-31 | 101 |
| sphere_m | ipc_graph/combined_graph | position | cloth | 0.0152307293 | 0.064396712 | 101 |
| sphere_m | ipc_graph/combined_graph | position | ABD | 3.32543273e-34 | 3.08148791e-33 | 101 |
| sphere_m | ipc_graph/combined_graph | velocity | cloth | 0.154998897 | 1.95117324 | 101 |
| sphere_m | ipc_graph/combined_graph | velocity | ABD | 3.32543273e-32 | 3.08148791e-31 | 101 |
| sphere_m | original_stiff/combined_graph | position | cloth | 0.0216108 | 0.0817015356 | 101 |
| sphere_m | original_stiff/combined_graph | position | ABD | 1.44527793e-34 | 3.08148791e-33 | 101 |

## 前轮多次配对结果（仅作背景比较）

前轮主比较为七对，组件比较为前三对；本轮每臂一次。样本、部分场景与规模不同，不把组件统计中位数相乘重建主比，也不把前轮下界移用到本轮。

| 前轮场景 | Stiff/组合七对中位 | host/Graph三对中位 | Graph/组合三对中位 |
|---|---:|---:|---:|
| hang | 1.3613 | 1.2411 | 1.1411 |
| fixed | 1.2860 | 1.1965 | 1.0844 |
| mixed | 1.1058 | 1.0486 | 1.0455 |

完整数值、原始材料差异、缺失原因、失败/回退、程序/输入身份及来源 SHA 见 [RESULTS.json](RESULTS.json)；逐运行表见 [RUN_INDEX.csv](RUN_INDEX.csv)。本报告没有新模拟、独立接受路径 CCD 或质量认证。
