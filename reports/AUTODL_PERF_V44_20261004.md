# v44 阶段诊断与清理结果

本轮先完成历史清理，再运行默认关闭的阶段诊断。两组兔子40帧目标探针均完成34帧，在第35帧达到240秒限制。主要时间集中在线性求解调用阶段，接触集合的CPU处理次之；不能再仅凭CCD候选增长判断CCD核函数是主要瓶颈。完整轨迹和性能仍未通过。

## 清理与磁盘

- 本地21份v42/v43因子SHA前后通过，15份相同文件改为硬链接，逻辑回收400011264字节（381.48MiB），原路径与压缩备份保留。历史输出禁止原位写入。
- AutoDL恢复后执行v43 finalizer：四版冻结身份通过，两个重复包及五份相同因子处理完成，实际可用增加533483520字节（508.77MiB）。
- v44结果下载核验后再次删除两份精确SHA相同的远端传输包，115116898字节（109.78MiB）。原始数据、二进制及本地压缩备份保留。
- 本轮最终数据盘可用1635569664字节（1.52GiB），GPU利用率0%、空闲24081MiB。新版本及实验输出占用了部分回收空间。
- `df -h / /root/autodl-tmp`显示系统盘可用9.3GiB；不能将此前SSH超时归因于系统盘满。`apt-get clean`可清理约29MiB安装缓存，本轮未执行。

审计：`CLEANUP_LOCAL_FACTORS_20261004.json`、`FINAL_REMOTE_V43_20261004.json`、`CLEANUP_FINAL_V44_20261004.json`。

## 实现和冻结

独立`sources/stiff_perf_v44`只修改`StiffGIPC/solver/toi_solver.cu`。新增默认关闭的`GIPC_STAGE_LOG`、`GIPC_STAGE_FROM_FRAME`、`GIPC_CCD_GEOMETRY_DIR`：阶段记录逐条flush，保存每次已完成PCG和内层试探；最近CCD的start/target/direction滚动保存，元数据作为写入完成标记。默认路径无新增GPU同步或文件I/O。

564个源文件及两二进制身份通过。v43冻结源码未改变。

- source_digest：`9d767f83e12495ba42c346ba75a35819125f801f3b4593d53c4e6901237f0a1e`
- v44二进制SHA256：`182beea05af1359651619d51cae1946d9a8fe58f4ffd3867a6528c34b28bd98a`
- 结果包95504208字节，SHA256：`e1256d31ea966d55ea67e8c0dba6406b3d7aca1cb9d165f3d6cab4c30021678d`

## 验证与保留失败

AutoDL CUDA12.8/SM89 Release构建成功；CTest 2/2，TOI组件6+7+7+17项通过。五次悬挂两帧运行完成，PCG次数均为[2,21,3,24]，无已完成调用的breakdown或上限。

原两帧开关逐位对照断言**失败**，保留原`smoke.log`和`V44_smoke.json`，未放宽或重写原门槛。开关最大绝对差1.5699247457590104e-16；诊断关闭组自身重复最大差1.964574336543734e-16。五运行全部配对最大差2.0513105103425744e-16。这只支持将本版用于探索性定位，不证明诊断对长轨迹严格无影响，更不认证逐位轨迹一致。

开启诊断的两组短测试各40条事件，通过事件连续性、阶段配对和几何核验。完整下载后，396个文件逐项SHA通过，本地CPU重新分析与远端JSON完全一致；几何有限值、direction=start-target逐位关系、位移范数均通过。几何快照不是完整仿真重启点，不包含速度和所有帧历史，也不是超时瞬间PCG输入。

## 第35帧（未完成）的阶段结果

| 指标 | Graph | host |
|---|---:|---:|
| 已完成物理帧 | 34 | 34 |
| 墙钟秒，含监视器约1秒误差 | 240.996 | 240.601 |
| 第35帧已完成PCG调用 | 799 | 890 |
| 第35帧线性求解调用累计秒 | 177.977 | 197.626 |
| 导数组装累计秒 | 2.152 | 2.231 |
| 线搜索及其诊断累计秒 | 0.945 | 1.059 |
| CCD候选生成累计秒 | 0.584 | 0.623 |
| CCD窄相累计秒 | 0.749 | 0.650 |
| 接触集合CPU处理累计秒 | 32.586 | 17.330 |
| 最大CCD候选数 | 56951567 | 27278805 |
| 最大试探相对安全状态位移，m | 8.590 | 15.329 |
| 超时正在进行的位置，零基outer/inner | 8 / 190 | 10 / 178 |

阶段名`pcg_begin`实际包围`calculateMovingDirection → solve_linear_system`，包括全局线性系统构建、预条件器相关准备、PCG及解分发，**不是纯PCG迭代核函数时间**。下一轮需继续细分。CPU集合处理也包含部分候选数据读回和map/set操作，并非纯GPU CCD。

第30帧起的诊断窗口分别记录853/951次完成的PCG，最大迭代1361/1401；这些完成调用均无上限或breakdown。两次超时各有一个PCG调用未返回，不能把其未完成事件当成已通过。

本轮host与先前v43停在不同帧（35而非38），不能视为同状态比较。短程已观察到运行间末位差异，不能据此证明长轨迹分叉的具体原因。未盲目继续严格rho全轨迹；应先保存并验证相同状态/系统，避免把不同轨迹的结果误判为PCG精度的因果效应。

## 指标边界

- host启动时GPU空闲19908MiB，而Graph启动24081MiB；基线存在约4173MiB占用，符合上一轮退出后资源尚在回收的可能性，但未保存该时刻逐进程采样。host报告的显存增量0不可用，不代表无需显存。末尾确认无GPU计算进程。
- 开启同步、逐阶段写盘、PCG审计，所有耗时只作诊断；不作Graph/host性能排名。
- 原始统计文件仍只包含完成帧，本轮额外阶段日志保留了未完成帧的事件。不同覆盖范围不能直接相加或替代。
- 本轮未导出全部接受子路径，未进行新的完整路径CCD；未完成40/100帧或物理质量验收。

## 执行入口

远端根：`/root/autodl-tmp/stiff_toi_cudagraph_20260929/v44_20261004`。

```text
bash tools/autodl_v44_build.sh
ctest --test-dir builds/autodl-stiff_perf_v44 --output-on-failure
GIPC_VALIDATE_COMPONENTS=reports/components.json builds/autodl-stiff_perf_v44/gipc
python3 tools/benchmark_perf_v44.py --phase smoke
/root/miniconda3/bin/python tools/repeat_smoke_v44.py
python3 tools/benchmark_perf_v44.py --phase graph
python3 tools/benchmark_perf_v44.py --phase host
/root/miniconda3/bin/python tools/analyze_stages_v44.py . reports/STAGES_V44.json
```

其中smoke在两次仿真完成后因逐位门槛失败退出；其余构建/检查/分析返回0。graph/host运行结果分别为timeout，benchmark调度脚本返回0仅表示成功保存结果。

本地`tools/verify_v44.py`和`tools/analyze_stages_v44.py downloads/autodl_perf_v44_20261004 reports/STAGES_V44_CPU_20261004.json`均返回0。详细证据见`VERIFICATION_V44_20261004.json`、`STAGES_V44_CPU_20261004.json`和下载目录下的原始阶段日志。

## 下一步

1. 在线性求解调用内拆分全局系统构建、MAS/Cholesky准备、PCG迭代、审计及解分发，补充可靠的GPU空闲基线等待。
2. 在第35帧内层增长开始处保存可恢复的完整状态及固定A/b；恢复后先验证关键缓冲和组装/求解一致性，再对同状态的默认/严格rho、内层退出与试探累计运动做对照。
3. 将滚动CCD几何与对应outer关联，验证大位移与候选/CPU集合增长的关系；之后再决定算法修复。保留PCG预算、物理与安全条件，不以增加运行时限作为修复。
