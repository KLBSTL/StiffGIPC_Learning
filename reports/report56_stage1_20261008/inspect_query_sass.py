"""Bounded CPU-only cuobjdump of the actual new production DCD EE kernel.

Never launches the simulator or a GPU profiler. Retains complete SASS plus a
static instruction inventory; counts must not be interpreted as dynamic cost.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
EXE=ROOT/'build/report56_stage1_verified_20261008/Release/gipc.exe'
CUOBJDUMP=Path('C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/cuobjdump.exe')
EXPECTED='d4da6bd5ec8544b87d0f7b77f67c6bb02a95c2afa19a24a8a6013f69d89f3902'
FUNCTION='_Z13_selfQuery_eePKiS0_PK7double3S3_PK5uint2PK4AABBPK4NodeP4int4SE_PjPidji'

def record(path):
    data=path.read_bytes();return {'path':str(path.resolve()),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

def require(condition,message):
    if not condition:raise ValueError(message)

def main():
    before=record(EXE);require(before['sha256']==EXPECTED,'Actual new program identity mismatch')
    command=[str(CUOBJDUMP),'--dump-sass','--function',FUNCTION,str(EXE)]
    process=subprocess.run(command,capture_output=True,timeout=30,check=False)
    text=process.stdout.decode('utf-8',errors='replace')
    (OUT/'NEW_EE_SASS.log').write_text(text,encoding='utf-8')
    if process.stderr:(OUT/'NEW_EE_SASS.stderr.log').write_bytes(process.stderr)
    require(process.returncode==0,'Offline cuobjdump failed')
    require(record(EXE)==before,'Actual exe changed during offline inspection')
    functions=re.findall(r'^\s*Function\s*:\s*(\S+)',text,re.M)
    require(functions==[FUNCTION],'Exactly the production DCD EE function required')
    instructions=[]
    for line,raw in enumerate(text.splitlines(),1):
        match=re.search(r'/\*([0-9a-fA-F]+)\*/\s+(.*?)\s*;',raw)
        if not match:continue
        assembly=match[2].strip();plain=re.sub(r'^@!?P\d+\s+','',assembly)
        opcode=plain.split()[0];base=opcode.split('.')[0]
        instructions.append({'line':line,'pc':'0x'+match[1],'assembly':assembly,'opcode':opcode,'base_opcode':base})
    local=[r for r in instructions if r['base_opcode'] in ('LDL','STL')]
    for row in local:
        address=re.search(r'\[([^]]+)\]',row['assembly'])
        row['address']=address[1] if address else None
        row['direct_R1_constant_address']=bool(address and re.fullmatch(r'R1(?:\+0x[0-9a-fA-F]+)?',address[1]))
    manifest=json.loads((ROOT/'runs/report56_stage1_20261008/probe/cloth_sphere7_l/build_manifest.json').read_text(encoding='utf-8'))
    compile_entries=[r for r in manifest['cuda_commands'] if str(r.get('source','')).replace('\\','/').endswith('/mlbvh.cu')]
    require(len(compile_entries)==1,'Exact mlbvh compile command record')
    lineinfo_flags=[re.findall(r'(?:-lineinfo\b|--generate-line-info\b|--device-debug\b|-G\b)',c)
                   for c in compile_entries[0]['commands']]
    stack_sites={'0x0120':'root node init','0x0360':'pending-node pop','0x9c30':'left internal child push','0x13500':'right internal child push'}
    require({r['pc'] for r in local if r['opcode'] in ('LDL','STL')}==set(stack_sites),'Four exact 32-bit stack candidates')
    for r in local:
        r['static_address_region']='node-ID stack candidate' if r['pc'] in stack_sites else 'lower frame fixed-offset 64-bit values'
        if r['pc'] in stack_sites:r['static_control_flow_role']=stack_sites[r['pc']]
    by_pc={r['pc']:r for r in instructions}
    evidence_pcs=['0x0120','0x0170','0x0350','0x0360','0x0460','0x9c30','0x134f0','0x13500',
                  '0x13510','0x13550','0x13560','0x13580','0x135a0']
    require(by_pc['0x135a0']['assembly']=='BRA 0x360','Traversal loop back edge')
    source=ROOT/'StiffGIPC/collision/mlbvh.cu';source_lines=source.read_text(encoding='utf-8').splitlines()
    source_evidence=[{'line':line,'text':text.strip()} for line,text in enumerate(source_lines,1)
        if 1632<=line<1777 and any(token in text for token in ('stack[65]','stack_ptr = stack','*stack_ptr++','*--stack_ptr','while(stack < stack_ptr)'))]
    report={'schema':'report56.new_ee_sass.v1','cpu_only':True,'new_gpu_launches':0,
        'performance_certified':False,'exe':before,'cuobjdump':record(CUOBJDUMP),'script':record(Path(__file__)),
        'command':command,'returncode':process.returncode,'sass':record(OUT/'NEW_EE_SASS.log'),
        'functions':functions,'instruction_count':len(instructions),'opcode_counts':dict(Counter(r['base_opcode'] for r in instructions)),
        'local_instruction_count':len(local),'local_opcode_counts':dict(Counter(r['opcode'] for r in local)),
        'local_address_forms':dict(Counter(r['address'] for r in local)),
        'local_instructions':local,'instructions':instructions,
        'sass_source_line_annotations_present':bool(re.search(r'\.loc\b|File\s+\"|line\s+\d+',text)),
        'actual_mlbvh_compile_lineinfo_flags':lineinfo_flags,
        'static_stack_control_flow_evidence':[by_pc[pc] for pc in evidence_pcs],
        'current_source':record(source),'current_source_stack_evidence':source_evidence,
        'stack_mapping_is_control_flow_inference_not_debug_line_mapping':True,
        'dynamic_pc_local_transaction_metrics_present':False,
        'scope':'Static instruction inventory only. No per-PC execution count, local-sector attribution, stall attribution, query/narrow timing split or predicted saving.'}
    (OUT/'NEW_EE_SASS_ANALYSIS.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    md=['# 新程序 production DCD EE：CPU-only SASS来源检查','',
        f'实际exe SHA256 `{EXPECTED}` 已前后核验。CUDA13 cuobjdump只提取一个 production `_selfQuery_ee`，没有模拟器、GPU或新counter capture。完整静态SASS有{len(instructions)}条指令、{len(local)}条local load/store站点。','',
        '| PC | 指令 | 静态控制流映射 |','|---|---|---|']
    for row in local:
        if row['pc'] in stack_sites:md.append(f'| {row["pc"]} | `{row["assembly"]}` | {stack_sites[row["pc"]]} |')
    md+=['','R1是该函数的local-frame基址。初始化写R1+0x60；R18从R1+0x64开始，R19保存pending指针；循环弹出R19-4，两个nonleaf路径push uint32子节点。末尾比较R1+0x60与pending指针，并在0x135a0回到0x360。这与源码stack[65]、root init、左push／右push／pop及while非空条件对应。它是控制流、地址与数据宽度的静态推断，不是调试行表的直接映射。','',
        '另外36条STL.64、42条LDL.64访问R1+0x00..0x58的较低frame区域，保存／读取64-bit值，出现于FP64运算与CALL邻域；不是上述从0x60开始的uint32节点栈槽。能区分这两类地址／控制流，却不能将所有低frame值严格命名为某个窄相临时对象、caller保存或register spill。','',
        '实际mlbvh编译命令含-lineinfo，但此次cuobjdump SASS没有源码PC注解；这不等于二进制没有line information。本有界检查未扩展行表解码。NCU仅收集aggregate local sectors与stall，没有逐PC执行次数或事务／延迟；因此不能把静态4/82的指令站点比例当成动态流量、时间或加速比例。node pop/push循环重复，距离分支条件也不同，静态指令站点数无法归一化它们。','',
        '结论：识别了真实遍历栈local指令，同时确认完整DCD EE有其他local值保存／读取。动态1,213,799 load sectors／615,043 store sectors的栈份额与窄相份额仍未确定，whole净省≥5%尚未建立。这次有界来源检查到此结束，不增加GPU计数或继续扩张追踪。','',
        '复现：`E:/Anaconda/envs/DL/python.exe reports/report56_stage1_20261008/inspect_query_sass.py`。所有静态指令、PC、opcode/address清单、source/工具/exe身份和实际命令在JSON；完整反汇编保留在NEW_EE_SASS.log。']
    (OUT/'NEW_EE_SASS_ANALYSIS.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps({key:report[key] for key in ['cpu_only','functions','instruction_count','local_instruction_count',
        'local_opcode_counts','local_address_forms','sass_source_line_annotations_present','actual_mlbvh_compile_lineinfo_flags']},ensure_ascii=False))

if __name__=='__main__':main()
