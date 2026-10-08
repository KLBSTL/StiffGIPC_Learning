"""CPU-only, read-only SQLite analysis of the two Stage-1 single-frame captures.

Run from any directory: E:/Anaconda/envs/DL/python.exe <this file>
Only the requested PROFILE_COST_ANALYSIS.json/.md artifacts are written.
"""
from __future__ import annotations
from collections import defaultdict
import argparse
import json
from pathlib import Path
import re
import sqlite3
import sys
sys.dont_write_bytecode=True

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
import analyze_profile as base
from report_profile_attribution import analyze as attribution
from local_identity import record

# Additions are separate from the unchanged original allowlist. Exact function
# tokens only; CUB algorithms retain unspecified operands unless a unique
# outside-Graph runtime launch can establish a native CPU-stage owner.
EXTRA_RULES={
    'collision.safety_edge_triangle':('_edgeTriIntersectionQuery','_edgeTriIntersectionQueryOrdered'),
    'collision.safety_ground':('_checkGroundIntersection',),
    'collision.accd_self_narrow':('_cub_reduct_self_step','_reduct_min_selfTimeStep_to_double'),
    'collision.ground_feasible_step':('_cub_reduct_ground_step','_reduct_min_groundTimeStep_to_double'),
    'collision.cfl_bound':('_cub_reduct_cfl','_reduct_max_cfl_to_double'),
    'collision.close_constraint':('_calSelfCloseVal','_checkSelfCloseVal','_cub_reduct_MSelfDist','_cub_reduct_MGroundDist','_reduct_MSelfDist','_reduct_MGroundDist'),
    'energy.producer':('_get_triangleFEMEnergy_Reduction_3D','_getQuadBendingEnergy_Reduction','_getBarrierEnergy_Reduction_3D',
        '_getFrictionEnergy_Reduction_3D','_getFrictionEnergy_gd_Reduction','_getKineticEnergy_Reduction_3D',
        '_computeSoftConstraintEnergy_Reduction','_computeGroundEnergy_Reduction','_getRestStableNHKEnergy_Reduction_3D',
        '_getFEMEnergy_Reduction_3D','_getBendingEnergy_Reduction','cal_abd_kinetic_energy_kernel','cal_abd_shape_energy_kernel'),
    'contact.friction_assembly':('_calFrictionHessian','_calFrictionGradient','_calFrictionLastH_DistAndTan','_calFrictionLastH_gd'),
    'fem.assembly':('_calculate_triangle_fem_gradient_hessian','_calculate_quad_bending_gradient_hessian',
        '_calculate_fem_gradient_hessian','_calculate_fem_gradient','_calculate_stableNHK_gradient_hessian','_getKineticGradient','_calKineticGradient'),
    'abd.preconditioner_prepare':('cal_abd_system_preconditioner_kernel1','cal_abd_system_preconditioner_kernel2'),
    'abd.contact_assembly':('write_barrier_hessian','setup_abd_system_hessian_abd_fem_kernel',
        'cal_abd_system_barrier_gradient_double3_kernel'),
    'abd.body_assembly':('cal_abd_body_gradient_and_hessian_kernel','write_abd_body_hessian'),
    'matrix.assembly_adjustment':('_reorder_triplets','_partition_collision_triplets','adjust_fem_fem_contact_indices_kernel',
        'setup_fem_mass_triplets_kernel','zero_fem_boundary_hessian_kernel'),
    'mas.topology_prepare':('_buildCollisionConnection_new','_buildConnectMaskLx_new','_nextLevelCluster','_prefixSumLx',
        '_computeNextLevel','_buildCML0_new','_preparePrefixSumL0_new','_aggregationKernel','_buildLevel1_new',
        'calculate_subsystem_bcoo_indices_kernel'),
    'linear.subsystem_assembly':('fem_assemble_kernel','abd_assemble_gradient_kernel'),
    'linear.retrieve_direction':('abd_retrieve_solution_kernel','fem_retrieve_solution_kernel','cal_dx_from_dq_double3_kernel'),
    'state.step':('_stepForward','step_forward_q_kernel','step_forward_vertices_kernel'),
    'state.frame_update':('_updateVelocities','_computeXTilta','update_velocity_kernel','cal_q_tilde_kernel'),
    'reduce.geometry_partial':('_cub_reduct_max_double3_to_double','_cub_reduct_dot','_cub_reduct_squared_norm'),
    'memory.kernel_fill':('buffer_fill_kernel',),
}
PATTERNS={group:[re.compile(r'(?<![A-Za-z0-9_])'+re.escape(fn)+r'(?=[(<])') for fn in names]
          for group,names in EXTRA_RULES.items()}

def enriched_classify(symbol):
    original=base.classify(symbol)
    if original!='unknown_kernel':return original
    hits=[group for group,patterns in PATTERNS.items() if any(p.search(symbol) for p in patterns)]
    if len(hits)>1:raise ValueError('Ambiguous extended classification: '+symbol)
    if hits:return hits[0]
    if symbol in {'memset32','memset32_post','memcpy32','memcpy32_post'}:return 'memory.graph_internal_kernel'
    if 'cub::' in symbol:
        if 'DeviceRadixSort' in symbol:return 'library.cub_radix_sort_unspecified_operand'
        if 'DeviceScan' in symbol or 'DeviceCompactInit' in symbol:return 'library.cub_scan_unspecified_operand'
        if 'DeviceSelect' in symbol:return 'library.cub_select_unspecified_operand'
    if 'fast_segmental_reduce_ptr_kernel' in symbol:return 'library.segmented_matrix_reduce_unspecified_owner'
    return original

def load_capture(folder):
    path=folder/'nsight.sqlite';before=record(path)
    connection=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
    connection.row_factory=sqlite3.Row;connection.execute('PRAGMA query_only=ON')
    try:
        tables={r[0] for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        def load(table):
            return [dict(r) for r in connection.execute(f'SELECT * FROM "{table}"')] if table in tables else []
        strings={r['id']:r['value'] for r in load('StringIds')}
        def name(row,fields):
            for field in fields:
                value=row.get(field)
                if value is not None:return strings.get(value,str(value)) if isinstance(value,int) else str(value)
            return '<unnamed>'
        ranges=[r|{'name':name(r,('text','textId'))} for r in load('NVTX_EVENTS')
                if r.get('end') and r['end']>r['start']]
        frames=base.canonical_frames([r for r in ranges if r['name']=='ipc.physical_frame'])
        if len(frames)!=1:raise ValueError('Exactly one canonical completed captured frame required')
        windows=[(frames[0]['start'],frames[0]['end'])]
        correlations=defaultdict(list)
        for r in load('CUPTI_ACTIVITY_KIND_RUNTIME'):correlations[r.get('correlationId')].append(r)
        def owner_for(row):
            matches=correlations.get(row.get('correlationId'),[])
            owner=None
            # Deliberately exclude all Graph nodes from CPU-range attribution.
            if not row.get('graphNodeId') and len(matches)==1:
                api=matches[0]
                owners=[n for n in ranges if n['name'].startswith(base.APP_PREFIXES)
                    and n.get('globalTid')==api.get('globalTid') and n['start']<=api['start']<=api['end']<=n['end']]
                if owners:owner=min(owners,key=lambda n:n['end']-n['start'])['name']
            return owner
        kernels=[]
        for row in load('CUPTI_ACTIVITY_KIND_KERNEL'):
            if not base.clip(row['start'],row['end'],windows):continue
            symbol=name(row,('demangledName','shortName','name','nameId'))
            kernels.append(row|{'symbol':symbol,'original_category':base.classify(symbol),
                'extended_category':enriched_classify(symbol),'outside_graph_runtime_owner':owner_for(row)})
        copies=[r|{'outside_graph_runtime_owner':owner_for(r)} for r in load('CUPTI_ACTIVITY_KIND_MEMCPY')
                if base.clip(r['start'],r['end'],windows)]
        graph=load('CUPTI_ACTIVITY_KIND_GRAPH_TRACE')
        columns=[dict(r) for r in connection.execute('PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)')]
    finally:connection.close()
    if record(path)!=before:raise ValueError('SQLite changed during read-only analysis')
    return kernels,copies,graph,ranges,windows,columns

def summarize_groups(rows,windows,key):
    groups=defaultdict(list)
    for row in rows:groups[row.get(key) or 'unattributed'].append(row)
    return {name:base.row_timing(members,windows) for name,members in sorted(groups.items())}

def source_evidence(symbol):
    names=[fn for group,fnames in EXTRA_RULES.items() for fn in fnames
           if any(p.search(symbol) for p in [re.compile(r'(?<![A-Za-z0-9_])'+re.escape(fn)+r'(?=[(<])')])]
    if not names:return []
    found=[]
    for path,contents in SOURCE_LINES.items():
        for line,text in enumerate(contents,1):
            if any(re.search(r'\b'+re.escape(fn)+r'\s*\(',text) for fn in names):
                found.append({'path':str(path.relative_to(ROOT)).replace('\\','/'),'line':line,'text':text.strip()})
                if len(found)>=4:return found
    return found

SOURCE_LINES={}

def source_anchors():
    requests=[('DCD VF / narrow distance classification','StiffGIPC/collision/mlbvh.cu','void _selfQuery_vf_body('),
        ('FullCCD VF / swept candidate output','StiffGIPC/collision/mlbvh.cu','void _selfQuery_vf_ccd_body('),
        ('DCD EE / narrow distance classification','StiffGIPC/collision/mlbvh.cu','void _selfQuery_ee_body('),
        ('FullCCD EE / swept candidate output','StiffGIPC/collision/mlbvh.cu','void _selfQuery_ee_ccd_body('),
        ('self ACCD point-triangle branch','StiffGIPC/core/GIPC.cu','value = 1.0 / point_triangle_ccd('),
        ('self ACCD edge-edge branch','StiffGIPC/core/GIPC.cu','value = 1.0 / edge_edge_ccd('),
        ('edge-triangle safety traversal','StiffGIPC/core/GIPC.cu','__global__ void _edgeTriIntersectionQuery('),
        ('energy production owner','StiffGIPC/core/GIPC.cu','CostScope cost_production("ipc.energy_production"'),
        ('energy final reduce/readback owner','StiffGIPC/core/GIPC.cu','CostScope cost_reduction("ipc.energy_scalar_reduce_and_readback"'),
        ('tetrahedral FEM gradient','StiffGIPC/fem/femEnergy.cu','__global__ void _calculate_fem_gradient('),
        ('ABD generalized-to-Cartesian move direction','StiffGIPC/abd_system/abd_system_function/cal_x_from_q.cu','__global__ void cal_dx_from_dq_double3_kernel(')]
    anchors=[]
    for description,path,needle in requests:
        hits=[(line,text) for line,text in enumerate(SOURCE_LINES[ROOT/path],1) if needle in text]
        if len(hits)!=1:raise ValueError('Source anchor missing or ambiguous: '+description)
        line,text=hits[0];anchors.append({'description':description,'path':path,'line':line,'text':text.strip()})
    return anchors

def analyze_one(folder,frame):
    original=base.analyze(folder/'nsight.sqlite',folder/'cost.jsonl',frame)
    attributed=attribution(folder)
    kernels,copies,graph,ranges,windows,columns=load_capture(folder)
    extended=summarize_groups(kernels,windows,'extended_category')
    graph_nodes=[r for r in kernels if r.get('graphNodeId')]
    outside=[r for r in kernels if not r.get('graphNodeId')]
    owned=summarize_groups(outside,windows,'outside_graph_runtime_owner')
    bvh_owners={k:v for k,v in owned.items() if 'bvh' in k}
    focus_owner_sets={
        'bvh.discrete':{'collision.discrete_bvh_build'},
        'bvh.swept':{'collision.swept_bvh_build_or_refit'},
        'matrix.convert_including_library_kernels':{'linear.matrix_convert'},
        'mas.prepare':{'mas.hierarchy','mas.numeric_fill','mas.numeric_aggregate','mas.legacy_factor'},
        'energy.all_owned_kernels':{r['outside_graph_runtime_owner'] for r in outside
            if (r['outside_graph_runtime_owner'] or '').startswith('ipc.energy_')},
    }
    focused={name:base.row_timing([r for r in outside if r['outside_graph_runtime_owner'] in owners],windows)
             for name,owners in focus_owner_sets.items()}
    focused['energy.final_reduce_owned']=base.row_timing([r for r in outside
        if r['extended_category']=='reduce.cub_unspecified_operand'
        and (r['outside_graph_runtime_owner'] or '').startswith('ipc.energy_')],windows)
    focused['energy.owned_gpu_copies']=base.row_timing([r for r in copies
        if (r['outside_graph_runtime_owner'] or '').startswith('ipc.energy_')],windows)
    if sum(v['count'] for v in extended.values())!=len(kernels):raise ValueError('Category partition count mismatch')
    if abs(sum(v['sum_ms'] for v in extended.values())-base.row_timing(kernels,windows)['sum_ms'])>1e-8:
        raise ValueError('Category partition sum mismatch')
    if any(r['outside_graph_runtime_owner'] for r in graph_nodes):raise ValueError('Graph node got CPU owner')
    symbols=defaultdict(list)
    for r in kernels:symbols[r['symbol']].append(r)
    changed=[]
    for symbol,rows in symbols.items():
        if rows[0]['original_category']=='unknown_kernel':
            changed.append({'symbol':symbol,'original_category':'unknown_kernel',
                'extended_category':rows[0]['extended_category'],**base.row_timing(rows,windows),
                'graph_node_count':sum(bool(r.get('graphNodeId')) for r in rows),
                'unique_outside_graph_cpu_owners':sorted({r['outside_graph_runtime_owner'] for r in rows if r['outside_graph_runtime_owner']}),
                'current_source_evidence_not_binary_identity':source_evidence(symbol)})
    changed.sort(key=lambda r:-r['union_ms'])
    resource_fields=[r['name'] for r in columns if any(t in r['name'].lower() for t in ['local','register','shared','stack'])]
    resources=[]
    for symbol,rows in symbols.items():
        if rows[0]['extended_category'] not in {'collision.discrete_query','collision.swept_query','collision.safety_edge_triangle','collision.accd_self_narrow'}:continue
        variants=defaultdict(list)
        for r in rows:variants[tuple(r.get(k) for k in resource_fields)].append(r)
        resources.append({'symbol':symbol,'category':rows[0]['extended_category'],
            'variants':[{'resources':dict(zip(resource_fields,values)),**base.row_timing(members,windows)} for values,members in variants.items()]})
    for group,stats in extended.items():
        stats['share_of_captured_frame_wall_percent']=stats['union_ms']/original['cpu']['physical_frame_nvtx_wall_ms']*100
        stats['share_of_gpu_work_union_including_whole_graph_percent']=stats['union_ms']/attributed['gpu_work_union_including_whole_graph']['union_ms']*100
    resolved=json.loads((folder/'resolved_config.json').read_text(encoding='utf-8'))
    config={name:resolved.get(name) for name in ['contact_backend','dt','ipc_newton_tol','pcg_rho_tol','configured_pcg_execution',
        'configured_pcg_graph_chunk','requested_preconditioner','mas','ipc_stopping','acceleration_features','report_components','cost_observation']}
    return {'folder':str(folder.relative_to(ROOT)).replace('\\','/'),'frame':frame,'capture_configuration':config,
        'identity':{name:record(folder/name) for name in ['nsight.sqlite','cost.jsonl','requested.json','resolved_config.json','build_manifest.json','capture_validation.json']},
        'original_rules_analysis':original,'original_attribution':attributed,
        'extended_symbol_categories':extended,'unknown_symbol_review':changed,
        'outside_graph_gpu_launch_owner_categories':owned,'bvh_gpu_launch_owner_categories':bvh_owners,
        'focused_gpu_components':focused,
        'outside_graph_gpu_copy_owner_categories':summarize_groups(copies,windows,'outside_graph_runtime_owner'),
        'graph':{'whole_graph_envelope':base.row_timing(graph,windows),
            'node_activity':base.row_timing(graph_nodes,windows),'node_symbol_categories':summarize_groups(graph_nodes,windows,'extended_category'),
            'outside_graph_kernels':base.row_timing(outside,windows),
            'cpu_scope_attribution_of_internal_nodes':False},
        'kernel_resource_schema':columns,'query_kernel_resources':resources,
        'resource_conclusion':'Every captured query/ACCD resource variant has localMemoryPerThread=0; localMemoryTotal is nonzero. These fields do not identify a stack-array allocation, dynamic allocation, spill instructions, local-memory transactions, occupancy, or a GPU bottleneck. No register-spill claim is made.'}

def write_markdown(result,path):
    profiles=result['profiles'];rows=[]
    for label,p in profiles.items():
        a=p['original_attribution'];o=p['original_rules_analysis']
        rows.append(f"| {label} / f{p['frame']} | {o['cpu']['physical_frame_nvtx_wall_ms']:.3f} | {a['gpu_work_union_including_whole_graph']['union_ms']:.3f} | {p['graph']['whole_graph_envelope']['union_ms']:.3f} | {p['graph']['node_activity']['union_ms']:.3f} | {o['unknown_kernel']['union_ms']:.3f} |")
    lines=['# Stage-1 Nsight 单帧成本分析','',
        '仅覆盖 sphere 第49帧与 fixed 第40帧的 capture。GPU 时间为裁剪到唯一 completed physical-frame NVTX 窗口后的区间并集；CPU NVTX 是包含子范围的包络。两者、等待 API 和 Graph 包络均不能相加。结果不构成整场收益或 5% 晋级门槛证据。','',
        '| Capture | CPU frame ms | GPU union 含 whole Graph ms | whole Graph ms | Graph nodes ms | 原规则 unknown ms |',
        '|---|---:|---:|---:|---:|---:|',*rows,'']
    lines+=['两份 resolved 配置均为 `conditional_graph` / chunk=1、legacy MAS，FullCCD refit、batched energy、energy reuse、discrete BVH refit 均关闭。因此 `swept_bvh_build_or_refit` 是范围名字，本次实际配置走 build。sphere 采用 node trace，fixed 采用 whole-Graph trace：0 表示该表没有记录该粒度，不能表示 Graph 或其算子没有执行。','']
    focus=['collision.discrete_query','collision.swept_query','collision.safety_edge_triangle','collision.accd_self_narrow',
        'collision.bvh','energy.producer','matrix.convert','mas.numeric_scatter','mas.numeric_reduce','mas.inverse_prepare','spmv']
    lines+=['| 符号类别：GPU union ms | sphere f49 | fixed f40 |','|---|---:|---:|']
    for name in focus:
        values=[p['extended_symbol_categories'].get(name,{}).get('union_ms',0) for p in profiles.values()]
        fixed=f'{values[1]:.6f}' if name!='spmv' else 'Graph 内部不可见'
        lines.append(f'| {name} | {values[0]:.6f} | {fixed} |')
    lines+=['','| 实际 kernel：每格为 launch次数 / GPU union ms | sphere f49 | fixed f40 |','|---|---:|---:|']
    for name in ['_selfQuery_vf','_selfQuery_ee','_GroundCollisionDetect','_selfQuery_vf_ccd','_selfQuery_ee_ccd',
                 '_edgeTriIntersectionQuery','_cub_reduct_self_step']:
        values=[]
        for p in profiles.values():
            hits=[r for r in p['original_rules_analysis']['kernel_symbols'] if r['symbol'].startswith(name+'(')]
            if len(hits)!=1:raise ValueError('Missing or ambiguous focused symbol: '+name)
            values.append(f'{hits[0]["count"]} / {hits[0]["union_ms"]:.6f}')
        lines.append(f'| `{name}` | {values[0]} | {values[1]} |')
    lines+=['','`collision.bvh` 仅含原符号规则识别的建树核函数，未包含被规则留作 unknown 的 radix/scan；下表由唯一 runtime correlation 的 BVH owner 给出包括库核函数的完整 launch 集合。两种口径有包含关系。','',
        '| 包含库核函数的 GPU launch 集合 / 归约 | sphere f49 ms | fixed f40 ms |','|---|---:|---:|']
    for name in profiles['sphere']['focused_gpu_components']:
        values=[p['focused_gpu_components'][name]['union_ms'] for p in profiles.values()]
        lines.append(f'| {name} | {values[0]:.6f} | {values[1]:.6f} |')
    lines+=['','| CPU NVTX 包络（包含子范围，不可相加） | sphere f49 ms | fixed f40 ms |','|---|---:|---:|']
    stages=['collision.discrete_bvh_build','collision.discrete_query','collision.swept_bvh_build_or_refit','collision.swept_query',
        'collision.self_ccd','ipc.line_search','ipc.energy_evaluation','ipc.energy_scalar_reduce_and_readback',
        'linear.matrix_convert','mas.prepare','mas.hierarchy','graph.entry','graph.capture_and_instantiate',
        'graph.replay','graph.initial_readback','graph.final_readback']
    cpu_maps=[{r['stage']:r['cpu_union_ms'] for r in p['original_rules_analysis']['cpu']['nvtx_envelopes']} for p in profiles.values()]
    for name in stages:lines.append(f'| {name} | {cpu_maps[0].get(name,0):.6f} | {cpu_maps[1].get(name,0):.6f} |')
    lines+=['','能量 scalar/readback 的 CPU 包络远大于其 GPU final-reduce kernel 并集。这些同步读取会等待前面的 GPU 工作；差值不能全归为 final sum、PCIe 传输或可消除的 CPU 成本。`energy.final_reduce_owned` 包含 FEM scalar owner 和 ABD evaluation owner 下的 CUB reduce，排除了 Graph/CCD/PCG 的其他 CUB reduce；GPU copy 单列，仍不与上述 CPU 包络相加。','']
    for label,p in profiles.items():
        lines+=['',f'## {label} f{p["frame"]}：归因细节','',
            'FullCCD query 生成 swept AABB 候选；真正 self ACCD 在 `_cub_reduct_self_step` 内。DCD query 已包含距离分类。edge-triangle 是线搜索安全查询，保留为独立类别。','',
            '| outside-Graph BVH launch owner | GPU union ms |','|---|---:|']
        for name,value in p['bvh_gpu_launch_owner_categories'].items():lines.append(f'| {name} | {value["union_ms"]:.6f} |')
        lines+=['','| 唯一 runtime + native NVTX 的 energy owner | GPU union ms |','|---|---:|']
        for name,value in p['original_attribution']['energy_kernel_owners'].items():lines.append(f'| {name} | {value["union_ms"]:.6f} |')
        if p['graph']['node_symbol_categories']:
            lines+=['','| Graph 内部：仅依实际符号分类 | GPU union ms |','|---|---:|']
            for name,value in p['graph']['node_symbol_categories'].items():lines.append(f'| {name} | {value["union_ms"]:.6f} |')
            count=p['graph']['node_activity']['count'];apply_count=p['graph']['node_symbol_categories'].get('mas.restrict',{}).get('count',0)
            lines+=['',f'Graph nodes 为{count}次活动、{apply_count}次 restrict 算子活动；节点首尾跨度跨越该帧多次 replay 与外部求解/碰撞，不能当成单一 Graph replay 用时。上述 CUB 只按算法识别，未从 CPU `pcg.reduce_device` 或 `mas` 范围归因内部节点。']
        else:
            count=p['graph']['whole_graph_envelope']['count']
            lines+=['',f'此 capture 仅有{count}条 whole Graph activity，内部 SpMV、MAS apply、PCG reduce/control 耗时不可分解。']
        lines+=['','| 原 unknown 符号（最高15项） | 补充类别 | GPU union ms |','|---|---|---:|']
        for row in p['unknown_symbol_review'][:15]:
            symbol=row['symbol'].split('(')[0].replace('|','\\|')
            library=re.search(r'(Device[A-Za-z0-9_]+Kernel|fast_segmental_reduce_ptr_kernel)',symbol)
            if library:symbol=library[0]+'<…>'
            lines.append(f'| `{symbol}` | {row["extended_category"]} | {row["union_ms"]:.6f} |')
        lines+=['','| 查询/ACCD kernel | 资源字段的实际取值 |','|---|---|']
        for row in p['query_kernel_resources']:
            symbol=row['symbol'].split('(')[0];values=[v['resources'] for v in row['variants']]
            lines.append(f'| `{symbol}` | `{json.dumps(values,ensure_ascii=False)}` |')
    lines+=['','## 源码语义锚点','',
        '以下当前源码锚点用于检查符号语义；capture 的二进制身份由各文件夹 build manifest / capture validation 保留，不能用当前源码行号替代二进制身份。完整原符号、所有原规则 unknown 条目、补充分类与源文件 SHA256 均在 JSON。补充后两份 kernel 均归类，通用库 operand 与 Graph 内部 CPU owner 仍明确保留为未确定。','']
    for anchor in result['semantic_source_anchors_not_binary_identity']:
        target=str(ROOT/anchor['path']).replace('\\','/')+':'+str(anchor['line'])
        lines.append(f'- {anchor["description"]}：[源代码]({target})')
    lines+=['','## 范围与复现','',
        '原 `analyze_profile` 规则与 `report_profile_attribution.analyze(folder)` 结果完整保存在 JSON；补充分类另存，不修改 tools。Graph 内部只按实际 kernel 符号分类，绝不使用 replay 的 CPU 范围归因。通用 CUB reduce 的 operand 仍未由符号确定；能量 final reduce 使用唯一 runtime correlation 的 owner 数据，不能把全部 CUB reduce 算为能量归约。','',
        '查询源码有编译期固定大小的局部 `uint32_t stack[65]`（260字节），安全 edge-triangle 为 `stack[64]`（256字节）。资源列确实存在：各查询/ACCD 的 `localMemoryPerThread` 均为0，`localMemoryTotal` 非零；这组 launch 描述无法确认数组放置位置，也未提供动态分配、spill load/store、local-memory transactions、occupancy 证据。DCD VF/EE 的 registersPerThread=162/138 与 FullCCD=40 可作为后续资源检查线索，不能据此证明 register spill 或 GPU 瓶颈。节点访问与成本因果仍需独立测量。','',
        'CPU/GPU 包络、Graph 内部节点与 whole Graph 均存在包含关系；跨度中的 gaps 不等同于可消除开销。单个 capture 未证明整场成本份额、整场5%收益或相对 Stiff >2×。','',
        '复现：`E:/Anaconda/envs/DL/python.exe reports/report56_stage1_20261008/analyze_profile_costs.py`。脚本以 SQLite `mode=ro` / `query_only` 打开，核对前后身份，输出本 JSON 与 Markdown。']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,default=Path(__file__).parent)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    global SOURCE_LINES
    SOURCE_LINES={path:path.read_text(encoding='utf-8',errors='replace').splitlines()
                  for path in sorted((ROOT/'StiffGIPC').rglob('*.cu'))}
    result={'schema':'report56.profile_cost_analysis.v1','read_only_inputs':True,'performance_certified':False,
        'analyzers':{name:record(ROOT/'tools/local'/name) for name in ['analyze_profile.py','report_profile_attribution.py']},
        'reproduction_script':record(Path(__file__)),'extra_exact_symbol_rules':EXTRA_RULES,
        'semantic_source_anchors_not_binary_identity':source_anchors(),
        'current_source_identity_not_binary_identity':{str(p.relative_to(ROOT)).replace('\\','/'):record(p) for p in SOURCE_LINES},
        'profiles':{label:analyze_one(ROOT/'runs/report56_stage1_20261008'/folder,frame)
            for label,folder,frame in [('sphere','profile_sphere_f49_node',49),('fixed','profile_fixed_f40_graph',40)]}}
    path=args.output_dir/'PROFILE_COST_ANALYSIS.json';path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write_markdown(result,args.output_dir/'PROFILE_COST_ANALYSIS.md')
    print(json.dumps({'passed':True,'profiles':{label:{'frame':p['frame'],
        'frame_cpu_ms':p['original_rules_analysis']['cpu']['physical_frame_nvtx_wall_ms'],
        'gpu_union_including_graph_ms':p['original_attribution']['gpu_work_union_including_whole_graph']['union_ms'],
        'whole_graph_ms':p['graph']['whole_graph_envelope']['union_ms'],'graph_nodes_ms':p['graph']['node_activity']['union_ms']}
        for label,p in result['profiles'].items()},'output':str(path)},ensure_ascii=False))

if __name__=='__main__':main()
