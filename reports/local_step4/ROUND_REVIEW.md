# Step4：比较口径、显存恢复复测与真实GPU成本

2026-10-06，本机RTX3070 Laptop、WDDM；没有连接AutoDL。

## 为什么看起来速度比降了

**当前证据不支持源码精简造成大幅退速。** 最近固定兔子相对Stiff的总比是
三组59帧配对中位1.475×。1.071×和悬挂1.137×是已有Graph执行组合上加入
接触池的增量；不是整套实现相对原版的总比。4090的1.887×来自不同硬件、
21帧AL后端及因子作用配置；局部MAS约2.2×没有包含全部准备及整场成本。
详见 [比较身份核查](SPEED_COMPARISON_AUDIT.md)。

同一Step3程序、相同配置和近乎相同工作量，关闭不用的GPU应用后，悬挂首对
off由3.168775降至1.905721秒，on由2.786704降至1.838939秒；分别快39.86%和
34.01%。两次均297个方向，PCG相差最多1次。绝对更快却增量比更小，支持
桌面运行环境参与差异。没有单独控制历史时钟/功耗/负载，不宣称唯一因果证明。

## 三轮复测及逐轮决定

用户释放应用后创建新会话，GPU串行；r1 off→on、r2 on→off、r3 off→on。
每轮实际分析由主任务复核，再用其SHA进入下一轮；没有覆盖旧失败会话。

| 轮次 | off/on整段 | 接触41–50帧 | 当轮分析后的决定 |
|---|---:|---:|---|
| r1 | 1.03632× | 1.10672× | 与旧单次差异大，继续已声明的反序复核 |
| r2 | 1.05602× | 1.13603× | 线搜索省时重复，CCD抵消项重复；完成第三轮 |
| r3 | 1.05024× | 1.11483× | 六次齐全；只进入已声明两次成本追踪 |

配对中位整段1.05024×、净省4.783%；几何均值1.04749×。不能把1.050×写成
净省5.0%。六次51帧、旧硬检查及材料界限通过，固定点漂移0；最大off/on
顶点位置差约79.4微米、实际速度差约6.62mm/s。旧基线标定失败和固定场景
超界不因此消失，也不继承为本轮完整质量认证。

[POOL_SCREEN.md](POOL_SCREEN.md) 与 [POOL_SCREEN.json](POOL_SCREEN.json) 保存各轮
时间、工作量、阶段和窗口统计。每轮线搜索省约152–165ms，CCD增加58–62ms，
解释了复用查询的局部收益被抵消。没有据资源失败前缀补算加速比。

## 实现及执行的下一步

新增 `tools/local/profiler_windows.py`、`windows_owned_job.py` 和只读
`analyze_profile.py`，原求解器/基线/封存运行器/质量协议没有修改。
profiling继承已完成同身份固定兔子59帧配置，只改四个诊断字段；
Windows进程先挂入专属Job后运行，超时/异常只清理自己的完整子进程树。
共用 `.local_gpu.lock`；保留120秒、显存、磁盘、外部计算进程保护。
每原seal只允许两个不可变launch槽，失败也消耗槽；本轮两个已用完，无自动续测。

| 第57帧诊断 | 池关 | 池开 | 可用于什么判断 |
|---|---:|---:|---|
| CPU NVTX帧包络，ms | 161.739 | 148.198 | 单次带追踪的墙钟范围，不作正式加速比 |
| GPU活动union，ms | 112.627 | 98.476 | 实际设备活动覆盖时间，不与CPU等待相加 |
| 离散VF/EE查询，ms | 21.656 | 0.021 | 12次遍历被池分类替换，保留地面查询 |
| 完整swept查询，ms | 8.670 | 8.477 | 两版都保留安全CCD |
| edge-triangle检查，ms | 9.784 | 9.508 | 两版都保留接受状态安全检查 |
| SpMV含初始化，ms | 12.294 | 13.428 | 仍有工作，但不是整场唯一热点 |
| Graph节点GPU union，ms | 34.625 | 39.575 | 真实节点活动，不把读回等待当额外GPU成本 |
| Newton/PCG | 6/467 | 6/484 | 同配置而非同状态，不能归因全部差值给池 |

池开显式cp_*核合计约1.556ms（包括约0.812ms分类）。只优化这些核即使全
消除也无法达到5%的帧收益，更不能推断2×。CPU `graph.final_readback`约
54/57ms包含GPU等待；未知kernel包含FEM及安全查询，不能都算成浪费。
完整审查、scope局限和逐算子成本见 [PROFILE_REVIEW.md](PROFILE_REVIEW.md)，
原始离线归约为 [PROFILE_OFF.json](PROFILE_OFF.json)、[PROFILE_ON.json](PROFILE_ON.json)。

## 验证命令与边界

在仓库根目录：

```powershell
& 'E:/Anaconda/envs/DL/python.exe' -X utf8 tools/local/contracts_test.py
& 'D:/computer/cmake/bin/ctest.exe' --test-dir build/local_step3_20261006 -C Release --output-on-failure
```

在 `tools/local`：

```powershell
& 'E:/Anaconda/envs/DL/python.exe' -m unittest fixture_windows_contracts_test cpu_fixed_reference_test analyze_profile_contracts_test profiler_windows_contracts_test -q
```

分别8/8、5/5、37/37通过；总45项CPU工具检查。CTest包含原GPU夹具，由主任务
在capture结束后串行执行。一开始把所有同名`contracts_test`放入一次模块加载
触发历史模块名称冲突；未改封存工具，按上述独立入口重跑全部通过。这个
命令入口问题不解释仿真耗时差异，失败命令没有计入通过数。

两次实际capture命令分别为：

```powershell
& 'E:/Anaconda/envs/DL/python.exe' -X utf8 tools/local/profiler_windows.py run --reference-run runs/local_step3_20261006/fixed_r1/fixed_off_r1 --output runs/local_step4_profile_fixed_off_node_20261006 --mode node
& 'E:/Anaconda/envs/DL/python.exe' -X utf8 tools/local/profiler_windows.py run --reference-run runs/local_step3_20261006/fixed_r3/fixed_on_r1 --output runs/local_step4_profile_fixed_on_node_20261006 --mode node
```

都完整59帧、native requested/resolved一致、NVTX及CUDA节点齐全、专属Job退出。
SQLite导出写到独立目录，不修改capture收据。工具和输入SHA保存在各自
requested/evidence和离线分析JSON；程序仍是 `b2ad1988…f812`。
[VERIFICATION.json](VERIFICATION.json)记录重新核验的seal、八次完整运行收据、
实际capture及新工具SHA；没有把重型追踪轨迹计入性能组。

## 有限下一方向

**已验证：**三轮短窗齐全、短窗旧界限通过、两个真实GPU成本capture完成。
**支持但未证明：**环境影响先前比率；密集接触阶段池更有收益。
**已排除为本轮优先项：**池维护重写、把Graph等待当纯传输删除、仅凭SpMV
局部提速宣称达到2×。没有证明所有相关技术方向永久无效。
**待验证：**同质量100帧/300帧、独立接受路径CCD、正式总比和2×。

接触池整段净省4.783%，未到原5%组件门槛，也未到10%阶段门槛；停止性能
调优支线并继续默认关闭，不追加参数网格或长验收。下一轮先核查剩余安全
edge-triangle查询的重复工作、几何覆盖和结果语义。候选必须保留完整检测、
同状态核对结果并达到预声明整场收益；没有可省成本证据即停止，不凭热点
排名再写一个低潜力kernel。材料和legacy/rho保持不变。
