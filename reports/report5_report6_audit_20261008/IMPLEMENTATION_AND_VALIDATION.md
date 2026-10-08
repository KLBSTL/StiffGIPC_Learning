# 本轮实施与验证

2026-10-08。对应[完整对照与计划](ANALYSIS_AND_PLAN.md)。三名子代理并行审查，主任务交叉检查、整合及GPU串行验证。附件内的示意补丁未直接应用。

## 实际修改

| 变更 | 生产位置 | 作用与边界 |
|---|---|---|
| 尾warp安全参与CUB归约 | `StiffGIPC/linear_system/utils/spmv.cu:25` | 无效lane不读写全局矩阵/向量，携零值独立head参加32-lane归约；共享临时存储复用前同步；原FP64和对称贡献保留 |
| 线搜索接受门 | `StiffGIPC/solver/line_search_acceptance.h:19`；`StiffGIPC/core/GIPC.cu:11100` | 原9次能量回退预算；NaN/Inf、阈值溢出、无进展和预算耗尽明确失败；安全回退后的真实终态重查能量，不能继承旧trial接受 |
| CUDA事件生命周期 | `StiffGIPC/cuda_tools/scoped_cuda_events.h:12`；`GIPC.cu:11453,11714` | 帧级2事件及Newton6事件由局部owner释放，包含异常/terminal路径；部分创建失败清理，record/sync/elapsed边界保留 |
| 独立测试 | `tests/sym_spmv_tests.cu`；`tests/line_search_acceptance_test.cpp`；根`CMakeLists.txt` | 直接链接生产SpMV；纯接受条件helper边界；加入现有CTest |

baseline、材料、全局FP64/legacy MAS、rho=1e-4、IPC legacy .01/min6、dt=.01、完整CCD和既有执行开关默认值保持。非法接受路径现在会失败，不宣称所有旧路径完全一致。没有实现或默认推广尚未达到热点门槛的symbolic cache、stackless BVH、row-gather、segmented energy或多流。

## 构建身份

命令（仓库根目录，PowerShell7调用DL解释器）：

```text
E:/Anaconda/envs/DL/python.exe tools/build_windows.py --kind active --label report56_audit_20261008 --jobs 2 --configured
```

此前在新空目录使用VS2022/vcpkg、CUDA architecture86完成configure，并仅编译独立测试。`--configured`复用的是同一新配置，生产gipc此前无对象；全量native构建37个编译单元，逐单元对象/真实编译/链接覆盖和pre/post源身份检查通过。日志中68条CUDA命令包含MSBuild重复打印，不是68个不同CUDA单元。完整记录：`build/report56_audit_20261008/windows_manifest.json`，smoke还用未修改的`active_identity`复查对象映射、header时间围栏和实际链接。

修复版`gipc.exe` SHA256：`14017344bd0d19b2ffa3bb28259dcbb509e09c1c0ee06571a29b803b7ac9d310`。环境是本机RTX3070 Laptop、CUDA13.0.88、MSVC19.44；这不是AutoDL CUDA12.8/sm89版本，不能移用旧4090速度或宣布跨平台性能通过。

## 检查结果

受保护验证脚本沿用项目GPU锁及Windows owned Job，每个检查120秒预算、显存/磁盘保护、失败不自动重试。全过程CUDA检查串行；编译不启动GPU。

```text
E:/Anaconda/envs/DL/python.exe reports/report5_report6_audit_20261008/verify_fixes.py --build build/report56_audit_20261008 --out runs/report56_audit_20261008/narrow --stage narrow
E:/Anaconda/envs/DL/python.exe reports/report5_report6_audit_20261008/verify_fixes.py --build build/report56_audit_20261008 --out runs/report56_audit_20261008/full --stage full
E:/Anaconda/envs/DL/python.exe tools/bench/contracts_test.py
```

- 窄CTest 2/2通过；完整CTest **9/9通过**，包含既有PCG、chunk停序、预算、compensated和碰撞fixture。既有contact-pool fixture只是正确性检查，没有重新启动其性能研究或生产默认。
- 生产SpMV **171 case、513 Graph replay**通过，独立CPU参考最大归一化作用误差0（fixture采用二进制可精确表示输入；不代表一般浮点路径逐位等价）。每种sanitizer再次执行同一固定算子suite，无额外模拟或参数网格。
- Compute Sanitizer memcheck：**0 errors**；racecheck：**0 hazards / 0 errors / 0 warnings**；synccheck：**0 errors**。
- 接受条件CPU helper：**2304 checks passed**，包括9次预算、NaN/Inf、阈值溢出、最终安全state变化、subnormal步长下溢与严格进展。
- 公共配置/runner合同：**20/20通过**；`git -c core.whitespace=cr-at-eol diff --check`通过。合同通过不等于所有新helper已获得完整物理质量认证。

检查收据和原始短日志保存在`runs/report56_audit_20261008/{narrow,full}`；复制的可交付结果、日志哈希和程序/源码身份见[VALIDATION_RECEIPT.json](VALIDATION_RECEIPT.json)。sanitizer只检查本轮SpMV fixture，不声称整个模拟器已通过所有内存/同步审计。

## 真实场景冒烟

```text
E:/Anaconda/envs/DL/python.exe reports/report5_report6_audit_20261008/smoke_fixes.py
```

复用未修改的Windows runner、active identity和配置validator；每臂从零、120秒上限、GPU串行，原始轨迹/速度/统计/日志保留在`runs/report56_audit_20261008/smoke`。配置显式IPC/legacy/rho1e-4/.01/min6/dt.01；all为Graph+四已有执行组件，普通BVH间隔8，退休组件关闭。

| 场景/配置 | 帧数 | 方向数 | PCG总量 | 最大能量回退 | 最大相交回退 | 结果 |
|---|---:|---:|---:|---:|---:|---|
| 落球L host | 30 | 109 | 2275 | 1 | 0 | 完整/有限/配置与数值guard通过 |
| 落球L Graph | 30 | 109 | 2276 | 1 | 0 | 同上 |
| 固定兔子L all | 60 | 337 | 20234 | 1 | 0 | 同上 |
| 混合兔子L all | 35 | 142 | 3657 | 0 | 0 | 同上 |

155帧冒烟均完成，无报告PCGbreakdown/上限、Newton上限或line_search_failure。落球覆盖接触22–24帧，固定兔子推进到60帧，混合覆盖24–26和33–35窗口。它们不是完整120帧配对，也没有新的基线材料标定、独立接受路径CCD或实际速度等价认证。相交回退在这些轨迹未触发；后置安全回退及预算极端路径只有CPU helper检查，仍需专门native注入/同状态fixture。事件长期资源行为未进行300帧压力认证。Windows共享桌面负载未控制，**不从这四组计算或认证新加速比**。

## 仍待完成的项目

七场景旧4090结果只能定义当前成本缺口，不能证明新修复版更快。两个主重构的收益仍需成本门槛与同质量配对；本轮没有实现>2×。当前AutoDL地址22209只读探测被连接拒绝，未上传/改动/清理远端、未新增远端GPU测试。本机验证继续完成，无需等待连接恢复才能交付审查和计划。

下一轮优先量化精确有序symbolic命中与比较成本、查询栈spill/访问长尾，再按主计划选择是否实现。失败保留证据；不因本次所有小测试通过就推广新性能默认，或删除历史质量差距。
