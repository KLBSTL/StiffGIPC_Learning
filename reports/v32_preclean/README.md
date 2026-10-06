# 完整重编译前的诊断记录

这轮 2/30/60/100 帧记录全部保留在 `runs/local/robust_v32_*`；对应程序和 manifest 冻结在 `builds/local-fused-v32-preclean`。

最终报告检查发现新增 `frame_friction_frame_id=-1` 头字段的默认初始化没有进入旧 `GIPC.obj`（对象编译时间 17:59:50，头文件最后修改 18:10:18）。第一帧保存为 `captured_this_solve=false`，其余帧正常刷新。源码哈希相同仍不能据此证明增量构建已完整更新。

此目录保留当时矩阵、质量诊断与 CCD 结果。它们不进入最终性能验收；新程序执行 `--clean-first`，最终运行名使用 `robust_v32_final_*`，每帧快照捕获另设运行门槛，使用独立新 CCD 输出名。原有记录和供体未删除或覆盖。
