# 2026-10-06 大清理执行收据

本轮已完成审计清单中的历史运行大文件及无依赖旧源码、构建和缓存清理。执行仅限 `E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929`，未删除新仓库 `StiffGIPC_Learning` 的文件，没有运行 GPU、修改活动 native 或实验注册表。

## 实际空间与文件数

执行时间为 UTC 08:25:25–08:30:34。E 盘可用空间由 **104,331,886,592 B** 增至 **161,485,873,152 B**，实测增加 **57,153,986,560 B（53.2288 GiB，57.1540 GB）**。这是共享卷在元数据备份后、删除前至复核后的可用空间差；同期其他进程也可能改变卷空间，不把它与删除文件逻辑长度混同。

| 类别 | 删除文件数 | 删除逻辑字节 |
|---|---:|---:|
| 1,062 个历史运行的指定原始大文件 | 138,169 | 55,413,652,712 |
| 20 个旧源码副本目录 | 3,076 | 858,096,590 |
| 19 个旧构建目录 | 2,172 | 470,863,695 |
| 3 个缓存目录 | 166 | 2,457,607 |
| 合计 | **143,583** | **56,745,070,604** |

另删除 1,016 个空目录；删除前保留了 1,697 份旧构建元数据。`builds/local-fused-v34` 因受保护 manifest 仍引用而跳过，其余已批准条件项完成。没有扩大到整批 downloads、bundles 或工作区根目录。

## 保留与恢复

保留 55 个关键完整运行，包括最新两布料 Stiff/current 四次 100 帧基线、当前 contact_pool 四次守卫及失败 screen、冻结质量标定/复核、接受路径 CCD 审计、代表性失败系统与成本追踪。contact_pool 收益仍待验证，清理没有把资源失败或缺少测速证据改写成性能结论。

保留 `stiff_active`、`stiff_base`、`stiff_base_observed`、v34、v43、v50、v54、`stiff_robust_port` 八个源码目录及当前构建依赖。保留六套历史 A/b/M 系统、v55/v56 夹具、全部报告和工具代码。没有删除活动源中的 eligibility 或 bounded CCD 代码；其工程依赖整理属于新仓库单独工作。

历史源码可从已核验远端的 `archive/pre-cleanup-20261006` 恢复，提交为 `eaa1b2131bca06de41531c66ed64e74dd4316da0`。删除源码前按归档 Git blob 核验；未在归档中的旧重复 Assets 仅在与保留 base/v50 权威文件 SHA 一致时删除。源码归档不代表旧原始轨迹全部备份：本次明确裁剪的轨迹仅保留报告/结果元数据，不能再声称这些旧运行可直接原地重做完整物理验收。

## 核验结果

- 清理前逐文件验证目标绝对路径、大小、修改时间、硬链接数；拒绝工作区外路径及 reparse point，不跟随 junction/symlink。所有删除均由同一 PowerShell 7 进程按精确路径执行。
- 5,669 份保留运行元数据 SHA-256 前后全部一致。
- 16,368 个受保护文件的存在、大小、修改时间和硬链接数前后全部一致。这一项为文件身份元数据检查，不宣称所有大文件都做了内容重哈希。
- 清理后的独立检查重新核验六系统共 385 个文件 SHA-256 全部一致；八个保留源码目录均存在。
- 独立解析删除账本得到 143,583 条不重复且全在旧工作区内的路径，字节总数与执行摘要一致；1,062 条运行 tombstone 的 138,169 文件/55,413,652,712 B 与 raw 阶段一致，均未改变实验结果状态。
- `EXPERIMENT_DECISIONS.json` SHA-256 仍为 `b4dc17c54eb4cc7130d1111bd9316a99039d121c52c96f393235a15475a1e2ba`；冻结审计 JSON 的 SHA-256 仍为 `b4454372920c40f5d61d3c84b426555e7e0c62aa5c6045a3fde7f4b08764e2e5`。
- 当前 `builds/active/Release/gipc.exe` SHA-256 仍为 `faedd6f85f3c97356b2ad4bc607dfff7f0cd9c9ede79c6615336e0a679693b10`。本轮未重新构建，不因此新增算法或性能通过声明。

## 证据索引

- 冻结清单：[CLEANUP_AUDIT_20261006.json](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/CLEANUP_AUDIT_20261006.json)
- 执行摘要：[summary.json](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/cleanup_apply_20261006_082525_6344478/summary.json)
- 精确删除账本：[deleted_files.jsonl](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/cleanup_apply_20261006_082525_6344478/deleted_files.jsonl)
- 运行存储状态收据：[run_tombstones.jsonl](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/cleanup_apply_20261006_082525_6344478/run_tombstones.jsonl)
- 执行脚本：[CLEANUP_APPLY_20261006.ps1](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/CLEANUP_APPLY_20261006.ps1)，SHA-256 `6e817731aaba0b688a6a643397f997e1d83fef2a716b92e35738046125b719ec`。

同目录下 `source_recovery_*.json` 保存逐目录源码可恢复性依据，`build_metadata/` 保存删除前复制的构建元数据。三次较早预检分别因 Git 所有权、WMI 权限与 Git 输出格式问题停止，均为零删除；其收据目录 `cleanup_apply_20261006_080137_3847959`、`cleanup_apply_20261006_080201_7934897`、`cleanup_apply_20261006_080309_2792889` 完整保留。
