# 新程序 production DCD EE：CPU-only SASS来源检查

实际exe SHA256 `d4da6bd5ec8544b87d0f7b77f67c6bb02a95c2afa19a24a8a6013f69d89f3902` 已前后核验。CUDA13 cuobjdump只提取一个 production `_selfQuery_ee`，没有模拟器、GPU或新counter capture。完整静态SASS有4968条指令、82条local load/store站点。

| PC | 指令 | 静态控制流映射 |
|---|---|---|
| 0x0120 | `STL [R1+0x60], RZ` | root node init |
| 0x0360 | `LDL R4, [R19+-0x4]` | pending-node pop |
| 0x9c30 | `STL [R7+-0x4], R10` | left internal child push |
| 0x13500 | `STL [R19], R76` | right internal child push |

R1是该函数的local-frame基址。初始化写R1+0x60；R18从R1+0x64开始，R19保存pending指针；循环弹出R19-4，两个nonleaf路径push uint32子节点。末尾比较R1+0x60与pending指针，并在0x135a0回到0x360。这与源码stack[65]、root init、左push／右push／pop及while非空条件对应。它是控制流、地址与数据宽度的静态推断，不是调试行表的直接映射。

另外36条STL.64、42条LDL.64访问R1+0x00..0x58的较低frame区域，保存／读取64-bit值，出现于FP64运算与CALL邻域；不是上述从0x60开始的uint32节点栈槽。能区分这两类地址／控制流，却不能将所有低frame值严格命名为某个窄相临时对象、caller保存或register spill。

实际mlbvh编译命令含-lineinfo，但此次cuobjdump SASS没有源码PC注解；这不等于二进制没有line information。本有界检查未扩展行表解码。NCU仅收集aggregate local sectors与stall，没有逐PC执行次数或事务／延迟；因此不能把静态4/82的指令站点比例当成动态流量、时间或加速比例。node pop/push循环重复，距离分支条件也不同，静态指令站点数无法归一化它们。

结论：识别了真实遍历栈local指令，同时确认完整DCD EE有其他local值保存／读取。动态1,213,799 load sectors／615,043 store sectors的栈份额与窄相份额仍未确定，whole净省≥5%尚未建立。这次有界来源检查到此结束，不增加GPU计数或继续扩张追踪。

复现：`E:/Anaconda/envs/DL/python.exe reports/report56_stage1_20261008/inspect_query_sass.py`。所有静态指令、PC、opcode/address清单、source/工具/exe身份和实际命令在JSON；完整反汇编保留在NEW_EE_SASS.log。
