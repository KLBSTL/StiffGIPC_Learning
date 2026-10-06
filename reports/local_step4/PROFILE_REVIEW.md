# 固定兔子 f57：接触池节点追踪复核

2026-10-06。结论：**接触池确实替代了昂贵的离散遍历，但显式维护成本已经很小，没有证据支持继续重写维护路径。停止本轮池优化支线，默认保持关闭。** 三轮完整 51 帧悬挂布筛查的中位净省为 4.783%，低于预声明的 5% 门槛；这两份单帧追踪解释成本，不改变该决定。

本轮仅离线读取两份已有 SQLite、分析 JSON、cost JSONL、配置、stats 和原生源码；没有新增 GPU 运行或原生代码修改。两次捕获均已完成，后续不再追加 GPU 捕获。

## 1. 证据边界与可比性

输入：

- [PROFILE_OFF.json](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/local_step4/PROFILE_OFF.json)、[PROFILE_ON.json](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/local_step4/PROFILE_ON.json)。
- [池关运行目录](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/runs/local_step4_profile_fixed_off_node_20261006)、[池开运行目录](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/runs/local_step4_profile_fixed_on_node_20261006)中的 requested/resolved、cost、result 和 output/stats。
- [三轮前缀筛查](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/local_step4/POOL_SCREEN.md)及其 JSON。

两次 expanded_config **仅 `contact_pool` 不同**；均从零运行 `cloth_fixed_bunny_l` 59 帧，dt=.01，IPC conditional Graph，legacy MAS，旧停止规则 .01/min6，rho=1e-4。程序 SHA256 均为 `b2ad19884fc01b81827d560f9f1984346b5488851995c2fd495f7427eac0f812`，source digest 均为 `9e57e82e14d58703348e951be92c0652b958428ddbdc2295e6fecabbff62b738`。`mas_factor_action=triangular` 是 requested 字段；实际 legacy MAS 下该项为 inactive，不能把本轮归为稳定 Cholesky/B 因子实验。

两次都是单个完成的 f57 NVTX 物理帧，cost 分别 266/290 条，production、CPU/NVTX-only、GPU events 关闭；无越界或窗口外 GPU 活动记录。result 均为 completed、59 帧、capture_verified=true，同时 performance_certified=false、physical_quality_certified=false、desktop_load_uncontrolled=true。**这不是相同冻结状态，亦不是速度认证或新的质量通过结论。** 本轮 contact_pool_validate=false，不能从“validation_failed=0”冒充执行了同状态验证。

## 2. 查询收益与维护成本

下表 GPU 值为实际 kernel 活动时长之和；CPU 值为 NVTX 包络 union。两者不可相加。

| f57 实测项 | Pool off | Pool on | 解释 |
|---|---:|---:|---|
| CPU 物理帧包络 ms | 161.738546 | 148.197846 | 单次诊断比 1.09137；不可推广 |
| GPU activity sum ms | 112.626809 | 98.475947 | 减少 14.150862 ms |
| GPU interval union ms | 112.626585 | 98.475947 | 单次诊断比 1.14370；非整场速度 |
| 普通 VF/EE 查询次数 | 6+6 | 0+0 | pool 命中 6/6，old_queries=0，无 fallback |
| 普通 VF/EE kernel ms | 21.635034 | 0 | 剩余 6 次 ground 查询仍执行 |
| cp_classify ms | 0 | 0.811871 | 6 次，替代普通 VF/EE 遍历 |
| 其余显式 cp_* kernel ms | 0 | 0.743902 | 缓存、顶点/叶子/映射守卫、身份检查 |
| pool scope 归属 memcpy/memset GPU ms | 0 | 0.342720 | 72 次；下述独立归属方法 |
| 完整 swept VF/EE kernel ms | 8.669873 | 8.475375 | 两次均 6+6 次；ON 为捕获 identity 的 pool 变体 |
| 安全 edge-triangle intersection ms | 9.784111 | 9.508461 | 两次均 6 次；完整相交检查保留 |
| BVH kernel ms | 1.613785 | 1.612285 | 普通/扫掠树维护仍存在 |

全部显式 `cp_*` 共 54 次 kernel，1.555773 ms。另在原始 SQLite 中，把 memcpy/memset 的唯一 `correlationId` 对应 runtime API，要求该 API 同线程且完整包含于 `collision.contact_pool.prepare/guard` NVTX 内，得到 72 次 GPU 内存操作、0.342720 ms。只归属实际 GPU 活动，不计其 API 阻塞时间，不把其他 CUB/Graph 节点猜归 pool。新增总 memcpy 次数恰为 54，memset 为 18；与这 72 次归属相符。两类活动不重叠计项，显式维护合计 **1.898493 ms**。

可复算 GPU sum 账目为：

`98.475947 - 112.626809 = -21.635034 + 1.898493 + 5.585679 ms`。

最后 5.585679 ms 是**其余活动的净变化**，包含不同工作量、不同状态、捕获变体、运行波动；不能称为池维护开销。单独把 21.635034−1.898493 当作已证明的同状态净加速也不成立。捕获 identity 的额外工作内嵌在 swept kernel 中，本轮只能观测该 kernel 总时长，无法分离其纯附加成本。

源码解释与计数一致：[ipc_contact_pool.inl](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/collision/ipc_contact_pool.inl:286)在 swept capture 后保存状态、守卫范围和身份；[try_discrete](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/collision/ipc_contact_pool.inl:324)先验证地址/属性/方向/拓扑/范围，再用原 `_checkPTintersection` / `_checkEEintersection` 窄相位分类器生成普通接触。普通 BVH 与安全相交检测保持运行；[线搜索](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/StiffGIPC/core/GIPC.cu:11025)仍先 buildBVH、isIntersected，再 buildCP。**不能为继续提速直接删除这些安全检测。**

Pool-on f57 的 `pool_pairs=284751` 是六次 seal 的累计条目数；VF/EE 分别累计 56980/227771，不能把 284751 说成同时活跃接触数。六次 classify 都处理非空池，prepare_calls=12（capture 与 seal 各六次），池峰值 40,720,112 bytes。

## 3. 工作量已经分岔，线性阶段并未被池加速

| inner | PCG off/on | active_pairs off/on | 接受 alpha off/on |
|---|---:|---:|---:|
| 0 | 97 / 99 | 1089 / 1286 | .291318 / .265181 |
| 1 | 93 / 89 | 1252 / 1176 | .541700 / .864527 |
| 2 | 81 / 64 | 1224 / 1169 | .867775 / 1 |
| 3 | 67 / 73 | 1161 / 1135 | 1 / 1 |
| 4 | 70 / 76 | 1136 / 1093 | 1 / 1 |
| 5 | 59 / 83 | 1104 / 1065 | 1 / 1 |

两次均六个接受更新、cumulative_toi 退出、final_beta=0、没有 PCG 触顶。但 PCG 合计 **467→484（+3.64%）**；线性系统编号分别 314–319 与 308–313，开始 f57 前已有历史工作量差异，不能把逐 inner 当成同 A/b 对照。接触数与 alpha 的变化也显示输入轨迹不同；本证据不能把分岔单独归咎于 pool、原子归约或质量变化。

实际 Graph-node 活动由 8406→8712，差 306=17×18，与额外 17 PCG 迭代一致。Graph GPU sum **34.625436→39.575132 ms**（增加 4.949696 ms），CPU linear.total 包络 **80.610369→84.910731 ms**。因此“pool 查询收益被更高线性成本抵消一部分”有直接账目支持，但不是池使线性算子变慢的因果证明。

剩余 SpMV 类别为 12.294395→13.427814 ms；其中真正乘法为 11.568828→12.650439 ms，另含输出清零。MAS 各 kernel 的单次耗时变化并不一致，例如 local_action 次数增加而时长降低，prolong 次数增加同时总时长约翻倍；单个带节点追踪的帧不适合据此认证某个算子改进或退化。

## 4. CPU 等待、未知分类与可优化上界

ON 的 CPU graph.final_readback 包络为 **56.927947 ms**（OFF 54.039479），但 GPU memcpy 真正传输总时长仅 **0.768319 ms**（OFF .632928，且这是全帧所有 memcpy）。`cudaMemcpy*` API union 为 123.629702 ms（OFF135.850450），包含等待此前 GPU 队列完成。显式 synchronize API union 为1.940185 ms（OFF1.680271）。这些量互相可能重叠，**不可相加，也不能把 readback 的 57 ms 宣称成可删掉的 CPU/传输开销**。

内部 Graph NVTX scope 仍无法由 runtime 相关性唯一恢复；ON 全部39.575132 ms Graph-node union保留为内部 scope 未归因。kernel 符号分类独立有效。unknown_kernel 仍为 OFF34.220970/ON34.726847 ms，包含本报告明确按符号识别的 edgeTri、FEM装配、pool守卫、排序与内存辅助kernel等；没有更改冻结分析器及原有JSON分类。CUB未识别操作数亦保持未归因。未知不等于无用、也不等于可去除。

即使把显式池维护 1.898493 ms 全部消除，约占 ON CPU 帧包络 **1.28%**；这只是乐观的串行相减预算，实际还受重叠/依赖影响，远低于5%目标。SpMV约占该包络9.06%，即使算子提速15%，也只对应约1.18%帧耗时的理想节省（若指耗时直接减少15%，约1.36%）。没有足够证据为了宣称新实现而立刻再写一个 kernel。

完整 swept + edgeTri 在 ON 中仍有 **17.983836 ms GPU sum**（OFF18.453984），其中 edgeTri 单项9.508461 ms。这确认它们是剩余成本，但并未证明其中多少可在保持完整安全语义时移除。

## 5. 有界下一步与停止决定

[POOL_SCREEN](E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning/reports/local_step4/POOL_SCREEN.md)的三轮 51 帧悬挂布 off/on 为 1.03632、1.05602、1.05024，中位净省 **1−1/1.0502357396=4.783282%**。这与本次固定兔子单帧是不同场景/窗口，不能拼接为统一速度证据。依已声明门槛，**池继续 opt-in/default off，停止池维护优化，不扩参数网格，不进入长验收。**

唯一预声明的下一方向是**安全 edge-triangle intersection 的只读语义研究**，不立即实现新候选、不新增 GPU 捕获：

1. 沿 `_edgeTriIntersectionQuery` → `edgeTriIntersectionQuery` → `isIntersected` 核对完整覆盖、相邻单元排除、固定边界条件、任一命中语义与回退路径。当前源码在叶子处才排除所有顶点均固定的组合；本追踪没有节点访问量和固定/固定拒绝比例，不能据此宣称大量可省。
2. 只利用已有源码、拓扑及这两份捕获，判断是否存在可证明冗余的检测。现有 VF/EE 接触池不能直接视作覆盖 edge-triangle 安全相交的候选集；必须先证明覆盖，不能以相似名称替代证明。
3. 只有明确可省成本达到既定阶段门槛，且可给出不缩减完整安全语义的有界实现与验证方案，才提交一个候选。当前约9.51 ms edgeTri意味着若想单靠它省本帧5%，需要消除约7.41 ms、约78%的该kernel时间，尚无此证据。若只读研究不能支持这一规模，就结束研究并报告未达标，不追加 kernel 或GPU试验来延长支线。

目前没有数据支持承诺2×；同样不能把本帧CPU比例1.091×包装为整场、同质量或相对StiffGIPC的认证加速。

## 6. 文件身份与复算方法

固定分析输入：

| 文件 | SHA256 |
|---|---|
| PROFILE_OFF.json | `32133c583d1cd8103d1f2b956c6fd69bee4eedba051e587d7645bf776348b27f` |
| PROFILE_ON.json | `d00d05cf9d6ee7f1ef0f4b9681d82a5e6a075fb5319cc92b8acd69f6b71b549c` |
| fixed_off.sqlite | `56b7ab952ae30d0391327bf6602ed956ef5247ea9586fe8202d86c19cb4d6d74` |
| fixed_on.sqlite | `27809629d60a7f72ee0d3fb5ef774ae60be29cf8df20abc8755bc15161237611` |
| off cost.jsonl | `71c4e2f069461d88885fffbc4c54e0a16f9b059d25a774ffefe38212faad48eb` |
| on cost.jsonl | `b878a8853711f3dff6ee35955432feaeb65343d80f76831c117ed45bc51e4ebe` |

PCG/alpha/pairs直接读取 `output/stats.json.frames[56].newton`，并核实其中 `pcg.frame=57`；GPU分类和符号直接来自两份profile JSON。CPU不同stage为inclusive union，禁止父子相加。显式pool GPU kernel取完整符号包含 `::cp_` 的条目之和；pool memcpy/memset采用第2节的唯一runtime相关性及完整NVTX包含规则。SQLite用`mode=ro`和`PRAGMA query_only=ON`读取。

只读核对源码SHA256：`ipc_contact_pool.inl=0c2e6e14992e3aba7cbcafaf2071982d3a91d6a413bd987e0cfddbbe4f216bf3`，`core/GIPC.cu=5e4ffa38535ccff16095c9c5d59f005f5b0de9bf4616f8b277c333314e346539`。本报告只新增文件，没有改写任何既有报告、分析器、配置或原生代码。
