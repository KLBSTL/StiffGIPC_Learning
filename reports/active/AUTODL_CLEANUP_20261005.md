# AutoDL 历史数据清理记录（2026-10-05）

清理已完成。按清理前后磁盘可用字节数，净释放 **68,931,387,392 字节，即 68.93 GB（十进制）／64.20 GiB（二进制）**。本轮只清理 AutoDL 上经审核的历史路径，本机工作区没有删除操作。

| 磁盘 | 清理前可用（GiB） | 清理后即时可用（GiB） | 净增加（GiB） |
|---|---:|---:|---:|
| 系统盘 `/` | 8.74 | 27.09 | 18.35 |
| 数据盘 `/root/autodl-tmp` | 1.52 | 47.37 | 45.85 |
| 合计 | 10.27 | 74.47 | 64.20 |

依据是[清理前磁盘记录](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/downloads/autodl_cleanup_20261005/evidence/disk_before.txt)与[最终审计](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/downloads/autodl_cleanup_20261005/evidence/final_cleanup_audit.json)。这是清理结束时的空间快照；后续测试会继续产生文件。

清理范围按实验版本日期和明确路径确定：2026-10-01 之前的旧实验保留报告；之后的版本保留关键实验、参考实现和报告。没有按文件修改时间批量筛选。24 个目录的具体保留前缀见[清理策略](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/configs/active/autodl_cleanup_20261005.json)，执行计划 SHA256 为 `071acd235e83888107f2bee3f2070eba85fa36cb719fcfe994ac1a1af92d126c`。

- 删除目录内旧文件 **149,964 个**，另删除已完成报告提取与备份验证的归档 **59 个（12＋47）**，以及父目录下的 **3 个旧清理脚本**。
- 清理后重新核验 **11,240 份在原位置保留的报告 SHA256**；**3,392 个重要文件**的身份与审核计划一致。
- 保留 v37／v39／v41 的六份历史失败系统，v32 的核心移植源码，以及 v43 的稳定 MAS 源码、修正系统与代表性 host／Graph 实验。
- 当前目录 `/root/autodl_factor_retest_20261004_659cef89`、冻结输入、现用构建和运行环境均保留。清理期间 GPU 测试暂停。

删除前先复制报告，生成 SHA256 收据，在本机验证备份成员，再核验远端源文件身份与进程引用。删除按明确文件逐项记录。149,964 个目录内文件的逻辑长度合计 40.22 GB；存在硬链接，且保留了报告备份，因此不能将逻辑长度相加当作实际释放空间。

| 本机报告备份 | 已验证成员数（未去重） | 压缩包字节数 | 校验收据 |
|---|---:|---:|---|
| [首批 12 个归档的报告](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/downloads/cleanup_archive_reports_20261005.tar.gz) | 2,807 | 11,078,667 | [首批收据](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/AUTODL_ARCHIVE_BACKUP_RECEIPT_20261005.json) |
| [目录内保留报告](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/downloads/cleanup_tree_reports_20261005.tar.gz) | 11,240 | 142,890,156 | [目录报告收据](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/AUTODL_TREE_BACKUP_RECEIPT_20261005.json) |
| [后续 47 个归档的报告](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/downloads/cleanup_deferred_reports_20261005.tar.gz) | 1,929 | 95,689,588 | [后续收据](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/reports/active/AUTODL_DEFERRED_BACKUP_RECEIPT_20261005.json) |

三份本机压缩包均重新计算 SHA256，与对应收据一致。成员数可能在归档之间、或与目录备份重复，不能相加作为独立报告总数。报告选择范围以各提取清单为准；第二批还保留了归档内 `.log.gz` 的原始压缩字节。

完整逐项记录见[证据包](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/downloads/autodl_cleanup_20261005/cleanup_evidence_20261005.tar.gz)、[目录删除记录](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/downloads/autodl_cleanup_20261005/evidence/tree_apply.json)、[首批归档删除记录](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/downloads/autodl_cleanup_20261005/evidence/archives_pruned.json)与[后续归档删除记录](E:/university_class/ComputerGraphics/GIPC/stiff_toi_cudagraph_20260929/downloads/autodl_cleanup_20261005/evidence/deferred_pruned.json)。清理不构成求解器质量或加速比通过的证据；测试仍按原有质量门槛继续。
