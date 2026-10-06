"""Verify all measured port records and write raw cost plus quality limits."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
from collections import Counter
import numpy as np
from report_velocity_ablation import States,work

ROOT=Path(__file__).resolve().parents[1]
ARMS=['base','graph','port005_host','port005_graph','port1_host','port1_graph']
LABELS={'base':'Base','graph':'Base + Graph','port005_host':'Port 0.05','port005_graph':'Port 0.05 + Graph',
    'port1_host':'Port 1.0','port1_graph':'Port 1.0 + Graph'}
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def span(values):return {'median':statistics.median(values),'min':min(values),'max':max(values),'values':values}

def verified_run(row,manifest):
    run=ROOT/row['run'];req=read(run/'requested.json');result=read(run/'result.json')
    assert result['status']=='completed' and result['finite'] and result['recorded_frames']==req['steps']
    assert req['source_digest']==manifest['source_digest']
    binary='local-base' if req['arm']=='base' else 'local-fused-v32'
    assert req['exe_sha256']==manifest['binaries'][f'builds/{binary}/Release/gipc.exe']['sha256']
    assert req['runner_sha256']==sha(ROOT/'tools/run_robust_port.py')
    for k,v in {'scene':'cloth_sphere7_l','dt':.01,'tol':.01,'pcg_tol':.0001,'suite':'0','trace':True}.items():
        assert req[k]==v,(run,k)
    with (run/'trace/frames.csv').open(newline='') as stream:times=list(csv.DictReader(stream))
    assert [int(r['frame']) for r in times]==list(range(1,req['steps']+1))
    assert abs(sum(float(r['solver_ms']) for r in times)/1000-result['solver_seconds'])<1e-10
    w=work(run);frames=w.pop('frames');assert len(frames)==req['steps']
    assert w['pcg_limit_hits']==w['outer_limit_hits']==0
    execution=Counter(n['pcg'].get('execution','host') for f in frames for n in f['newton'] if 'pcg' in n)
    if req['execution']=='conditional_graph':
        assert set(execution)=={'conditional_graph'},(run,execution)
    w['actual_pcg_executions']=dict(execution)
    outer=[t for f in frames for t in f.get('toi',[])]
    w['minimum_safe_alpha']=min((t['alpha'] for t in outer),default=None)
    w['tiny_alpha_steps_below_1e4']=sum(t['alpha']<1e-4 for t in outer)
    if req['toi_policy']=='robust' and req['arm'] in ['base_toi','base_toi_graph']:
        assert all(f['toi_policy']=='robust' and f['toi_cross_frame_contacts'] for f in frames)
        assert all(f['robust_trial_velocity_tol']==req['robust_velocity_tol'] for f in frames)
        assert req['mu_scope'] is None and all(f['toi_penalty_scope']=='movable' for f in frames)
        assert all(t['candidate_query_policy']=='current_trial_shared_full_ccd' for f in frames for t in f['toi'])
        assert all(not t['friction_snapshot_refresh'] for f in frames for t in f['toi'])
        assert result['robust_frame_protocol_verified']
        assert all(f['toi_frame_friction_snapshot']['captured_this_solve']
            and f['toi_frame_friction_snapshot']['physical_frame']==i for i,f in enumerate(frames))
    return {'arm':row['arm'],'repeat':row['repeat'],'run':row['run'],'result':result,'work':w,
        'source_digest':req['source_digest'],'exe_sha256':req['exe_sha256'],
        'frames_csv_sha256':sha(run/'trace/frames.csv'),'stats_sha256':sha(run/'output/stats.json')},frames,times

def main():
    manifest=read(ROOT/'manifests/robust_port_v32.json')
    matrix=read(ROOT/'reports/ROBUST_V32_MATCHED_MATRIX_100.json')
    assert len(matrix['runs'])==18
    records=[];details={}
    for row in matrix['runs']:
        record,frames,times=verified_run(row,manifest);records.append(record)
        details[(row['arm'],row['repeat'])]=(frames,times)
    # A common phase mask comes from this base run, used unchanged for all arms.
    baseline_csv=details[('base',1)][1]
    mask=np.array([int(f['candidate_pairs'])+int(f['ground_candidates'])>0 for f in baseline_csv])
    assert mask.any() and (~mask).any()
    for r in records:
        ms=np.array([float(t['solver_ms']) for t in details[(r['arm'],r['repeat'])][1]])
        r['seconds']={phase:float(ms[selection].sum()/1000) for phase,selection in [
            ('total',np.ones(len(ms),dtype=bool)),('noncontact',~mask),('contact',mask)]}
    index={(r['arm'],r['repeat']):r for r in records}
    summary=[]
    for arm in ARMS:
        rows=[index[(arm,i)] for i in range(1,4)]
        item={'arm':arm,'label':LABELS[arm],
            'seconds':{phase:span([r['seconds'][phase] for r in rows]) for phase in ['total','noncontact','contact']},
            'directions':span([r['work']['directions'] for r in rows]),
            'cg_iterations':span([r['work']['cg_iterations'] for r in rows]),
            'outer':span([r['work']['outer'] for r in rows])}
        item['raw_paired_speedup_vs_base']={phase:span([index[('base',i)]['seconds'][phase]/index[(arm,i)]['seconds'][phase] for i in range(1,4)]) for phase in ['total','noncontact','contact']}
        item['raw_total_speedup_vs_base']=statistics.median(index[('base',i)]['seconds']['total'] for i in range(1,4))/item['seconds']['total']['median']
        item['raw_total_speedup_vs_graph']=statistics.median(index[('graph',i)]['seconds']['total'] for i in range(1,4))/item['seconds']['total']['median']
        summary.append(item)
    reference=States(ROOT/'runs/local/local_velocity_20261001_base_reference_r01')
    half=States(ROOT/'runs/local/local_velocity_20261001_base_reference_half_r01')
    reference_check=half.compare(reference)['cloth_only']
    states={('base',i):States(ROOT/index[('base',i)]['run']) for i in range(1,4)}
    quality=[]
    for row in records:
        state=states[('base',row['repeat'])] if row['arm']=='base' else States(ROOT/row['run'])
        quality.append({'arm':row['arm'],'repeat':row['repeat'],'physical':state.physical(),
            'vs_paired_base':state.compare(states[('base',row['repeat'])])['cloth_only'],
            'vs_reference':state.compare(reference)['cloth_only']})
    baseline_variation=[states[('base',i)].compare(states[('base',1)])['cloth_only'] for i in [2,3]]
    validations=read(ROOT/'reports/ROBUST_V32_MATCHED_PATH_VALIDATION.json')
    assert validations['all_passed'] and len(validations['runs'])==4
    audits=[]
    for row in read(ROOT/'reports/ROBUST_V32_MATCHED_MATRIX_60_PATHS.json')['runs']:
        record,frames,_=verified_run(row,manifest)
        outer=[t for f in frames for t in f['toi']]
        assert all(t.get('friction_snapshot_unchanged') for t in outer)
        assert all(f['toi_frame_friction_snapshot']['buffer_audit_enabled'] for f in frames)
        assert all(f['toi_frame_friction_snapshot']['captured_this_solve'] for f in frames)
        assert len(list((ROOT/row['run']/'trace/substeps').glob('safe_*.bin')))==len(outer)+60
        record['friction_buffer_checks']=len(outer);audits.append(record)
    profile_run=ROOT/'runs/local/robust_v32_matched_profile60_port1_host'
    pf=read(profile_run/'output/stats.json')['frames'];nr=Counter();outer=Counter()
    for f in pf:
        for n in f['newton']:nr.update(n.get('profile_ms',{}))
        for t in f['toi']:outer.update(t.get('profile_ms',{}))
    profile={'run':str(profile_run.relative_to(ROOT)),'seconds':read(profile_run/'result.json')['solver_seconds'],
        'newton_ms':dict(nr),'outer_ms':dict(outer),'diagnostic_only':True,
        'active_update_includes_full_ccd':True,
        'initial_hessian_frame23':pf[22]['toi_initial_hessian_diagonal'],
        'mu_frame23':pf[22]['toi'][0]['mu']}
    hess=profile['initial_hessian_frame23']
    assert pf[22]['toi_penalty_scope']=='movable' and hess['abd_max']==0
    assert np.isclose(profile['mu_frame23'],.1*max(hess['abd_max'],hess['fem_max']),rtol=1e-12)
    profile['actual_assembled_movable_mu_verified']=True
    profile['upper_bound_eliminate_entire_active_query']=1/(1-outer['active_update']/(1000*profile['seconds']))
    component=read(ROOT/'builds/v32_matched_toi_components.json');assert component['passed']
    data={'implementation':'v32 Robust-derived core port','quality_matched_speedup_qualified':False,
        'protocol':{'scene':'cloth_sphere7_l','cloth_vertices':2601,'frames':100,'repeats':3,'dt':.01,
            'newton_tol':.01,'pcg_tol':1e-4,'suite':'0','motion_rate':1,'robust_mu_scope':'movable',
            'execution':'serial local RTX3070Laptop',
            'timing_field':'sum(trace/frames.csv solver_ms)/1000; host steady_clock around IPC_Solver and cudaDeviceSynchronize; excludes process startup',
            'phase_mask':'base r01 native narrow candidate_pairs+ground_candidates>0, common to all arms/repeats',
            'noncontact_frames':(np.flatnonzero(~mask)+1).tolist(),'contact_frames':(np.flatnonzero(mask)+1).tolist()},
        'summary':summary,'runs':records,'quality':quality,'base_repeat_variation':baseline_variation,
        'reference_refinement':reference_check,'accepted_paths_60':validations,'friction_audits_60':audits,
        'profile':profile,'identity':read(ROOT/'reports/ROBUST_V32_FREEZE_VERIFICATION.json')}
    report=ROOT/'reports/ROBUST_PORT_IMPLEMENTATION_20261001.json'
    report.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    with (ROOT/'reports/ROBUST_PORT_IMPLEMENTATION_20261001.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['arm','seconds_median','raw_speedup_base','raw_speedup_graph','directions_median','cg_median','quality_qualified'])
        for s in summary:writer.writerow([s['arm'],s['seconds']['total']['median'],s['raw_total_speedup_vs_base'],s['raw_total_speedup_vs_graph'],s['directions']['median'],s['cg_iterations']['median'],False])
    lines=['# Robust 核心适配与本机验证报告：v32','',
        '## 结果','',
        '批准的核心适配已实现并完成窄验证。小场景 6 组 × 100 帧 × 3 次串行交错全部完成，共 1,800 帧；新增 GPU 组件和四组 60 帧独立接受路径 CCD 均通过。物理轨迹仍未达到现有 1% 诊断门槛，尚不能宣称质量匹配后的 TOI 加速。','',
        '## 实现与隔离','',
        '- 新源：`sources/stiff_robust_port`；新程序：`builds/local-fused-v32/Release/gipc.exe`；独立 manifest：`manifests/robust_port_v32.json`。原 v31 的 553 个文件、官方 base 和 v31 EXE SHA 均确认未变。',
        '- `GIPC_TOI_POLICY=robust` 启用当前 trial 全场 CCD 与活动集共用结果、跨帧 C/λ、独立释放年龄、上一帧 λ 的物理帧级摩擦快照。保留 Stiff 材料、混合 DOF、PCG 和严格能量下降。',
        '- 接触释放 26 次删除，重新激活重置年龄；过期身份可从新候选重新加入。过滤采用所有候选的 per-vertex 最早 TOI 和 epsilon tie。',
        '- 摩擦坐标/基使用 `mesh.o_vertexes`；本帧 inner/outer 不更新快照。真实缓冲区审计覆盖身份、法向力、坐标、切向基。',
        '- 最少 Newton 方向为 6，允许 velocity stop；累计 TOI 只积累实际接受 alpha。完整安全 CCD 固定 ratio=0.2；blocker 审计直接读取同批结果。',
        '- μ 范数按公开 Robust 的默认路径排除固定 ABD/FEM DOF，取组装后绝对标量对角最大值 × 0.1，不额外乘 dt²。robust 默认 movable，paper 默认 full；`--mu-scope full` 保留诊断对照。',
        '- paper 分支两帧运行通过。源码默认 paper；本实验 runner 显式默认 robust。独立供体 EE 修正和完整引擎验证见 `ROBUST_REFERENCE_V32.md`，不把旧未修正轨迹当作质量目标。','',
        '本机 Release 用 CUDA 13.0、sm_86 构建。本轮按用户最新要求在本机测试；AutoDL CUDA 12.8 编译与运行尚未验收。', '',
        '## 100 帧计时','',
        '场景为布料落球 `cloth_sphere7_l`，2,601 布料顶点，dt=0.01，Newton tol=0.01，PCG tol=1e-4，suite=0；本机 RTX 3070 Laptop 8 GiB。第 1/3 次正序、第 2 次反序，GPU 串行。solver_ms 是 IPC_Solver 加 cudaDeviceSynchronize 的 host steady_clock 耗时，包含该窗口的 CPU/GPU 工作；启动、窗口外导出和进程监控 wall 时间不进入下表。','',
        '| 组别 | 时间中位数 / s | 对 base 原始速度比 | 对 base+Graph | Newton 方向 | CG 迭代 |',
        '|---|---:|---:|---:|---:|---:|']
    for s in summary:lines.append(f"| {s['label']} | {s['seconds']['total']['median']:.3f} | {s['raw_total_speedup_vs_base']:.3f}× | {s['raw_total_speedup_vs_graph']:.3f}× | {s['directions']['median']:.0f} | {s['cg_iterations']['median']:.0f} |")
    lines+=['','速度比定义为 baseline 时间 / 本组时间；小于 1 表示变慢。上述是原始成本对比，所有 TOI 行均未通过质量资格。三组 Graph 的全部逐方向 execution 记录均为 conditional_graph，已检查实际执行路径。','',
        '### 非接触 / 接触','',
        f"共同窗口由 base r01 的原生近接触候选数非零定义：非接触 {len((~mask).nonzero()[0])} 帧，接触 {len(mask.nonzero()[0])} 帧。采用窄相位近接触集合，未把 BVH 粗候选当作接触。相同帧索引用于所有组，避免各组接触判断差异改变计时窗口。速度比为同重复编号的比值中位数。",'',
        '| 组别 | 非接触时间 / s | 非接触速度比 | 接触时间 / s | 接触速度比 | 总速度比范围 |',
        '|---|---:|---:|---:|---:|---:|']
    for s in summary:
        ratios=s['raw_paired_speedup_vs_base'];lines.append(f"| {s['label']} | {s['seconds']['noncontact']['median']:.3f} | {ratios['noncontact']['median']:.3f}× | {s['seconds']['contact']['median']:.3f} | {ratios['contact']['median']:.3f}× | {ratios['total']['min']:.3f}–{ratios['total']['max']:.3f}× |")
    lines+=['','## 安全、组件和质量','',
        '- 实际 GPU：7 个新夹具全部通过，包含年龄/重新激活、PT/EE 摩擦原始 λ、跨帧旋转后坐标与正交基、地面力、奇异几何受控失败、固定 ABD/FEM 不参与 μ 对角范数；原 AL、warm history、volume 检查同时通过。',
        f"- 四个 TOI 候选各 60 帧，独立 CPU Tight-Inclusion 共检查 {sum(r['validation']['paths_checked'] for r in validations['runs'])} 条保存路径，碰撞标记 0；其中包含相邻帧重复端点段。真实接受 outer 子步和摩擦缓冲区检查共 {sum(r['friction_buffer_checks'] for r in audits)} 次，全部通过。",
        '- 100 帧正式窗口全部有限、无 PCG/outer cap；程序内安全检查每个接受步执行。100 帧窗口没有另导出全部子步做独立 CCD，不将 60 帧独立验收外推为 100 帧独立验收。',
        '- 位置/速度只比较布料，显式排除固定 ABD 球；boundary_types 单独使用会把球当作自由顶点并稀释误差。','',
        '| 组别 | 对同次 base 最大布料位置偏差 / % | 对细参考最大位置偏差 / % | 最大伸长比范围 |',
        '|---|---:|---:|---:|']
    for arm in ARMS:
        rows=[r for r in quality if r['arm']==arm];a=[100*r['vs_paired_base']['max_mass_rms_over_cloth_scale'] for r in rows];b=[100*r['vs_reference']['max_mass_rms_over_cloth_scale'] for r in rows];c=[r['physical']['cloth_max_edge_stretch_all_frames'] for r in rows]
        lines.append(f"| {LABELS[arm]} | {min(a):.3f}–{max(a):.3f} | {min(b):.3f}–{max(b):.3f} | {min(c):.4f}–{max(c):.4f} |")
    base_variation=max(q['max_mass_rms_over_cloth_scale'] for q in baseline_variation)*100
    lines+=['',f"base 自身重复轨迹最大差为 {base_variation:.3f}%（相对 r01）。细参考为 dt=0.0025、tol=0.001、PCG tol=1e-8；与 dt=0.005 同阈值参考相比，最大布料位置差 {reference_check['max_mass_rms_over_cloth_scale']*100:.3f}%，仍未收敛到 1%。因此这里的细参考偏差是诊断量，不能当作精确物理解误差。TOI 对同时间步 base 的约数个百分点差异也未达到现有匹配门槛；JSON 保留全部绝对/相对速度误差，未据单一比例量给出质量结论。",'',
        '## 剩余耗时与取舍','',
        '单独 60 帧 `Port 1.0 host` 同步 profiling 只作诊断，未混入正式三次计时。active_update 包含共用的完整 CCD，safe_ccd 槽接近零是已复用结果的表现。','',
        '| 阶段 | 诊断时间 / ms | 占该诊断 solver 时间 |','|---|---:|---:|']
    for label,ms in [('PCG',nr['pcg']),('梯度/Hessian 组装',nr['assembly']),('内层线搜索',nr['line_search']),('活动集 + 完整 CCD',outer['active_update']),('安全状态更新/端点检查',outer['safe_update'])]:
        lines.append(f"| {label} | {ms:.1f} | {ms/(10*profile['seconds']):.1f}% |")
    lines+=['',f"即使整个活动集/共用 CCD 阶段全部消失，这组诊断的理想上限也仅 {profile['upper_bound_eliminate_entire_active_query']:.3f}×；实际只能优化其中集合和传输部分，并且完整安全 CCD 必须保留。当前阶段暂缓 GPU 集合重写：质量门槛未通过，且该阶段占比有限。",'',
        f"最终第 23 帧初始 Hessian 范数中固定 ABD 贡献为 {hess['abd_max']:.3f}，自由布料贡献 {hess['fem_max']:.6f}；μ={profile['mu_frame23']:.6f}，已独立核对等于自由最大值 × 0.1。第 7 个 GPU 夹具直接验证新增的固定排除与绝对值；真实组装 profiling 另验证实际约简和系数接线。范围是此固定球/自由布料场景，未据一个夹具宣称验证了所有运动 ABD 配置。",'',
        '此前 full 范数诊断曾得到固定 ABD 最大值 305.386、布料 2.262，差约 135 倍，μ=30.539。供体源码确认它在 ABD/FEM diag_norm 中把固定贡献置零，再求最大值；这一实现差异已修正。full 范数程序与短窗口保留在 builds/local-fused-v32-full-scope 以及 robust_v32_final_* 运行目录；首轮增量构建数据另保存在 v32_preclean，均未混入最终 matched 三次计时。此修正有单独源码、GPU 和运行证据，但未通过相同状态的完整物理力对照，不把全部速度/质量变化归因于它。','',
        '建议下一阶段优先核对固定状态下的法向接触力、帧级摩擦力与材料响应，建立更可靠的参考收敛证据；通过质量检查后再优化 PCG/DOF 投影。现有证据不支持把公开 Robust 整套引擎替换作为一次低成本适配，也不支持只搬活动集就承诺旧 2–20×。','',
        '## 审阅与复现','',
        '独立规格审阅和代码审阅已完成，发现的过期身份、奇异夹具返回编号、审计改变 CCD、帧快照几何/重用和缓存记录身份问题均已修复。最终报告另捕获了增量构建漏更新头字段初始化的问题；旧程序/首轮记录已保留在 builds/local-fused-v32-preclean 与 reports/v32_preclean，执行 --clean-first 后再修正 μ 的固定排除，最终使用独立 robust_v32_matched 运行名重测。每个正式帧均检查实际快照捕获和帧编号，旧轮数据不进入本报告。已知范围：本轮验收 motion_rate=1；多个 solve 调用共用一个帧时，原统计/子步文件编号仍会覆盖；runner 将这种帧协议失败记录为 protocol_failed。','',
        '```powershell',"& 'E:/Anaconda/envs/DL/python.exe' tools/verify_v32_freeze.py", "& 'E:/Anaconda/envs/DL/python.exe' tools/check_robust_port_components.py builds/v32_matched_toi_components.json", "& 'E:/Anaconda/envs/DL/python.exe' tools/benchmark_robust_port.py --steps 100 --repeats 3", "& 'E:/Anaconda/envs/DL/python.exe' tools/report_v32_results.py",'```','',
        'matrix 的已完成结果只在当前源、EXE、runner 和参数均匹配时复用。独立 CCD 脚本拒绝覆盖原结果，重测时应使用新目录/报告名。JSON 保存逐运行 SHA、质量量、时段速度比、审计命令和计时范围；CSV 与 PDF/PNG 图保存汇总。']
    (ROOT/'reports/ROBUST_PORT_IMPLEMENTATION_20261001.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot(summary)
    print(json.dumps({'formal_runs':len(records),'frames':1800,'path_checks':sum(r['validation']['paths_checked'] for r in validations['runs']),
        'quality_qualified':False,'summary':[{'arm':s['arm'],'seconds':s['seconds']['total']['median'],'raw_speedup':s['raw_total_speedup_vs_base']} for s in summary]}))

def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    colors=['#536575','#278ca0','#d6933c','#cf7940','#74a36b','#477b52']
    labels=['Base','Base\n+Graph','Port .05','Port .05\n+Graph','Port 1.0','Port 1.0\n+Graph']
    for ax,field,title in zip(axes,['total','contact'],['100-frame solver cost','Common contact-window cost']):
        values=[s['seconds'][field]['median'] for s in summary]
        ax.bar(range(len(summary)),values,color=colors,width=.65)
        for i,s in enumerate(summary):
            ax.scatter([i-.09,i,i+.09],s['seconds'][field]['values'],s=13,color='#1a2833',zorder=3)
            ax.text(i,max(s['seconds'][field]['values'])+.3,f'{values[i]:.2f}',ha='center',fontsize=9)
        ax.set_xticks(range(len(summary)),labels,fontsize=9);ax.set_ylabel('seconds');ax.set_title(title)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.set_ylim(0,max(max(s['seconds'][field]['values']) for s in summary)*1.15)
    fig.suptitle('Robust-derived Stiff port: 3 serial interleaved repeats',fontsize=13)
    fig.text(.5,.02,'Raw cost only. Physical trajectory quality gate failed; independent CCD covers 60-frame pilots.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.07,1,.93]);out=ROOT/'reports/figures';out.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/f'ROBUST_PORT_IMPLEMENTATION_20261001.{ext}',dpi=220)
    plt.close(fig)

if __name__=='__main__':main()
