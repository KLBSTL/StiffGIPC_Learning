## 9. 已批准的下一轮：精确线性结构诊断模块

本节记录后续授权实现，不能当作第6节C1缓存优化已实现或已达性能门槛。主任务先逐字节冻结旧程序的实际编译输入与Assets，再允许恢复编辑。本分工只加入观测；所有排序、当前FP64数值聚合、MAS准备与PCG仍走原路径。材料、rho=1e-4、legacy .01/min6、完整CCD和baseline没有在本分工改动。

### 9.1 文件与公共接口

| 文件 | 本轮职责 |
|---|---|
| `StiffGIPC/linear_system/utils/converter.h` | 纯CPU Config/Result接口、惰性私有Probe owner；不公开GPU kernel定义 |
| `StiffGIPC/linear_system/utils/converter.cu` | 完整排序前观察raw row/col；完整转换后观察sorting mapping、partition及compact canonical pairs |
| `StiffGIPC/linear_system/utils/linear_structure_probe.h` | 不依赖CUDA或JSON的配置、形状、状态分类、bijective key编码、Result及成本契约 |
| `StiffGIPC/linear_system/utils/linear_structure_probe.cuh` | 私有DeviceBuffer snapshots、精确GPU比较、快照准备、可选events及gap失效 |
| `tests/linear_structure_probe_contract_test.cpp` | 98项CPU边界/状态契约，实际执行待主任务 |
| `tests/linear_structure_probe_gpu_test.cu` | 41组GPU夹具计划，直接使用生产Probe和实际Converter，实际执行待主任务 |

公共Converter方法：`configure_structure_probe(const LinearStructureProbeConfig&)`、`reset_structure_probe() noexcept`、`last_structure_probe() const noexcept`。最后一个返回借用的`const LinearStructureProbeResult*`，未观察/关闭时为nullptr；在下一次观察、配置或reset前复制结果再转换JSON。主任务负责配置、JSON和CMake接线。

Config默认`enabled=false`、`gpu_events=true`、`include_converted_structure=true`。默认关闭的Converter没有Probe owner分配；关闭观察时没有比较、复制、event创建、GPU诊断调用或生产kernel内新增分支。启用时首次配置仅分配host owner；GPU snapshots/events在实际观察中惰性准备。pause/reset只清元数据并保留容量，析构时正常释放已有资源。一个Probe绑定首次观察的CUDA device，切换device明确拒绝，不误用另一设备的buffers/events。

### 9.2 三种结构身份分别观测

| 身份 | 严格形状 | 精确内容与位置 |
|---|---|---|
| ordered raw | owner、device、矩阵block_rows/cols、input/output offsets、完整raw长度 | `Converter::convert`排序前完整有序row/col，私有uint64 snapshot |
| mapping/partition | 上述raw形状及unique_count | 转换后完整raw长度的uint32排序排列与partition；私有两数组snapshot |
| compact canonical pairs | 独立owner、device、block_rows/cols、unique_count；不含raw长度或staging offsets | 转换后从`start`开始的unique row/col prefix，私有uint64 snapshot |

`(uint32(row)<<32)|uint32(col)`是int32二元组的无碰撞编码，不是概率hash。每次可比较的非空结构在GPU逐项读取并比较完整数组，仅将mismatch flag读回；没有以hash命中代替相等。first、owner/shape变化和内容变化时准备新快照；精确命中不重复复制。空数组按已观察的严格形状判断逻辑相等，比较/复制kernel不启动。

mapping/partition不变不能回答canonical pairs是否不变。例如raw keys数值改变但排序次序和分组保持一致，map/partition可能同时相同。相反，raw输入重排或重复条数变化时canonical pairs可能仍相同，因此独立pair形状不包含raw multiplicity。`pair_equal`只证明这组compact有序pairs相等；它**不是**FEM/MAS hierarchy复用的充分条件。后续仍必须验证子系统分区、topology epoch、MAS层数/块身份以及Graph失效等guard，且每次重算当前数值。

`observe_converted(mapping, partition, unique_count, compact_rows=nullptr, compact_cols=nullptr, observe_pairs=true)`在include_converted_structure下观察后三种输入。真实Converter提供compact指针，unique=0可显式传nullptr并观察空pair；仅测试mapping/partition的夹具用最后一项false显式跳过pair。非空却缺少compact指针拒绝。一次raw系统最多一次converted观察。

### 9.3 所有权、流顺序与观测间隔

生产raw prefix会在转换中被compact输出覆写，shared mapping/partition scratch随后可能被MAS重用。本实现的四种快照buffers均由独立Probe持有，不借用生产scratch地址。raw pack在排序之前排入同流；转换后mapping、partition和canonical pair快照在下一次scratch重用之前排入同流。所有诊断kernel、copy、flag读回和events显式使用`cudaStreamPerThread`；生产Converter/CUB wrappers按现有CMake的`--default-stream=per-thread`使用同一流。

关闭、reset、未观察converted系统、显式跳过pair都会失效对应观察元数据。重开后为first，raw的`gap_before`及pair的`pair_gap_before`记录间隔；跨未观察系统不产生相邻命中。建议独立诊断臂观察每个Converter调用、全部Newton。若只选帧，必须把各段first/gap独立计数。统计应同时报告所有相邻非空系统的owner/shape misses与eligible compare条件命中率，不能只用后者作为整场可复用比例。空结构的逻辑hit也不能与实际GPU比较命中混算。

该诊断包含host flag同步，位于PCG Graph之外；没有宣称探针可在Graph capture中调用。关闭诊断的正常生产路径继续按原Graph行为执行。

### 9.4 成本与Result口径

Result包括raw/converted/pair状态、各自observed/equal/comparison_performed/snapshot_updated、owner/dims/offset/count、observation_index、unique_count、logical snapshot bytes与实际capacity bytes；以及raw_compare/raw_snapshot、converted_compare/converted_snapshot、pair_compare/pair_snapshot的StageCost。unique_count仅在converted_observed=true时有意义。

每个StageCost记录`cpu_ms`、`gpu_ms`、`gpu_valid`、`readback_cpu_ms`、`completion_waited`。events开启时GPU区间只覆盖相应同流比较/快照工作；CPU区间仍可能等前序排队工作。events关闭时GPU耗时未知，gpu_valid=false；compare仍进行同流flag读回+同步，snapshot的CPU耗时则仅是提交成本、completion_waited=false。不能把这个提交耗时称作完成快照的GPU成本。GPU区间不包含host flag读回。

`workspace_cpu_ms`记录resize/分配的CPU区间，`event_setup_cpu_ms`记录首次event创建，`probe_cpu_ms`累加诊断观察段，不含正常converter工作或首次configure的host owner分配。CPU scope可能包含等待，不能将它全部视作可消除计算。逻辑snapshot内存为`16*raw_count + 8*unique_count`字节，另有flag、events和metadata；容量按DeviceBuffer既有增长策略保留，实际内存以capacity字段为准。完整成本判断还需独立臂CPU/GPU包络，不能只求stage GPU时间之和。

### 9.5 夹具、当前证据与待执行命令

CPU夹具覆盖0/1/31/32/33/255/256/257、enabled默认、owner、offset/count/device/dims变化、int32边界拒绝、disabled bypass和int32 key编码。GPU夹具两种event模式下覆盖以上尾长及4097、单raw尾key变化、same-count重排、borrowed storage relocation、mapping/partition单值独立变化、unique metadata变化、1→257→33→257容量、owner/dims、pause与遗漏观察gap、read-only guards。独立canonical夹具覆盖同raw重排与重复条数增长而pair相同、单canonical key变化、owner/dims、0 unique及3→6容量增长。

真实Converter夹具每种尾长执行三轮：初始转换；相同raw keys而FP64数值变化；raw重排而canonical pairs相同。CPU stable-sort检查生产map/partition，CPU二进制分数FP64聚合检查当前numeric blocks，并检查外部guards。没有复制修复/比较kernel到测试替身。

本分工当前仅执行`git diff --check`，结果通过；没有编译、CPU测试执行或GPU执行，也没有新的性能证据。主任务的CMake targets为`linear_structure_probe_contract_test`、`linear_structure_probe_gpu_test`，CTest names为`linear_structure_probe_contract`、`linear_structure_probe_gpu`；GPU target链接实际converter.cu和timer.cpp。

以下是待主任务在完成fresh configure、选定实际build目录并取得串行GPU执行时段后执行的命令，不是已经执行的结果：

```powershell
& 'D:/computer/cmake/bin/cmake.exe' --build 'build/report56_recheck_final_20261008' --config Release --target linear_structure_probe_contract_test linear_structure_probe_gpu_test
& 'D:/computer/cmake/bin/ctest.exe' --test-dir 'build/report56_recheck_final_20261008' -C Release -R '^linear_structure_probe_(contract|gpu)$' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 'build/report56_recheck_final_20261008/Release/linear_structure_probe_gpu_test.exe'
compute-sanitizer --tool synccheck --error-exitcode 1 'build/report56_recheck_final_20261008/Release/linear_structure_probe_gpu_test.exe'
```

这轮只测C1的结构命中与成本门槛。仅有结构相等或component夹具通过，不能证明缓存算法正确、数值/物理轨迹等价、全场净收益≥5%或相对原版Stiff达到2×。
