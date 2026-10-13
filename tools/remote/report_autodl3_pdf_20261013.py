"""Paper-style cloud reproduction report, with source values and real native views."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

from pypdf import PdfReader
from reportlab.pdfgen import canvas

REPO=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('paper_pdf',REPO/'tools/local/build_comparison_pdf_20261011.py')
p=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
SCENES=['Bunny','Mixed','Boxes']
METHODS=['original','pure','ours']


def heading(c,title,subtitle,number,total,section='AutoDL复测'):
    c.bookmarkPage(f'p{number}')
    c.addOutlineEntry(title,f'p{number}',0)
    c.setFont('Hei',17)
    c.setFillColor(p.colors.HexColor('#23384c'))
    c.drawString(p.MARGIN,p.PAGE_H-40,title)
    c.setFillColor(p.colors.black)
    p.text(c,subtitle,p.PAGE_H-52,9.2)
    c.setStrokeColor(p.colors.HexColor('#23384c'))
    c.setLineWidth(.7)
    c.line(p.MARGIN,p.PAGE_H-80,p.PAGE_W-p.MARGIN,p.PAGE_H-80)
    c.setFillColor(p.colors.HexColor('#606060'))
    c.setFont('Song',8)
    c.drawString(p.MARGIN,18,f'StiffGIPC | RTX 4090 | 2026-10-13 | {section}')
    c.setFont('Times',9)
    c.drawRightString(p.PAGE_W-p.MARGIN,18,f'{number} / {total}')
    c.setFillColor(p.colors.black)


def image_page(c,path,title,subtitle,number,total,section='附录'):
    heading(c,title,subtitle,number,total,section)
    source=p.source(path)
    with p.Image.open(source) as im:
        w,h=im.size
    available=p.PAGE_H-170
    scale=min(p.WIDTH/w,available/h)
    sw,sh=w*scale,h*scale
    c.drawImage(str(source),(p.PAGE_W-sw)/2,75+(available-sh)/2,width=sw,height=sh,mask='auto')
    p.text(c,'实际独立轨迹网格与测得数据；同场景各规则共用视角和范围，变形比例1。参考差异结合参考自身重复性解释。',60,8.7)
    c.showPage()


def table_rows(values):
    result=[]
    for s in SCENES:
        for m in METHODS:
            r=next(x for x in values if x['scene']==s and x['method']==m)
            result.append([s if m=='original' else '',p.LABELS[m],p.integer(r['N']),p.fmt(r['P'],1),
                p.fmt(r['t'],6),p.fmt(r['e_RMS_mm'],6),p.fmt(r['e_max_mm'],6)])
    return result


def main(root,out):
    data=root/'reports/autodl_recording105_20261013_v2'
    summary=json.loads(p.source(data/'summary.json').read_text())
    verified=json.loads(p.source(data/'delivery_verification_local.json').read_text())
    topology=json.loads(p.source(data/'cross_platform_topology.json').read_text())
    assert summary['complete'] and verified['passed']
    out.mkdir(parents=True,exist_ok=False)
    (out/'checkpoints').mkdir()
    p.register_fonts()
    phases=p.rows(data/'phase_table_first100.csv')
    local=p.rows(REPO/'reports/local_recording105_dt0p04_20261011/phase_table_first100.csv')
    cold=p.rows(data/'checkpoint_frame100_table.csv')
    trajectory=p.rows(data/'trajectory_frame100_table.csv')
    noise=p.rows(data/'reference_repeat_dispersion.csv')
    idx={(r['scene'],r['method']):r for r in phases}
    localidx={(r['scene'],r['method']):r for r in local}
    totals={m:sum(float(idx[s,m]['TimeTot']) for s in SCENES) for m in METHODS}
    collective={m:totals['original']/totals[m] for m in METHODS}
    packet=json.loads(p.source(root/'runs/autodl_recording105_20261013_v2/identities.json').read_text())
    c=canvas.Canvas(str(out/'AutoDL_3Scene_Main.pdf'),pagesize=(p.PAGE_W,p.PAGE_H),pageCompression=1)
    c.setTitle('StiffGIPC AutoDL三场景复测')

    heading(c,'AutoDL三场景复测：配置与本机对照','Bunny / Mixed / Boxes；当前记录版三种停止规则，另测较严数值参考。',1,6)
    y=p.PAGE_H-95
    y=p.text(c,'RTX 4090，Ubuntu 22.04，CUDA 12.8；355个仿真源码和输入文件与本机记录版逐字节一致。dt=0.04，候选rho=1e-4，FullCCD、MAS legacy、完整当前执行组合及conditional Graph K1。',y,10.5)-12
    y=p.text(c,'每规则105帧×3轮；性能汇总取前100帧，状态表取物理第100帧。MOV=Movement only；TOI=Cumulative TOI .01；GATE=Compensated + residual gate .03。较严参考为MOV/rho=1e-6，外层阈值仍为0.01。',y,10.5)-12
    vals=[]
    for s in SCENES:
        for m in METHODS:
            a,b=localidx[s,m],idx[s,m]
            vals.append([s if m=='original' else '',p.LABELS[m],p.fmt(a['TimeTot'],3),p.fmt(b['TimeTot'],3),
                p.fmt(float(a['TimeTot'])/float(b['TimeTot']),3),p.fmt(a['S_movement_over_rule'],3),p.fmt(b['S_movement_over_rule'],3)])
    y=p.table(c,['Scene','Rule','Local t (s)','Cloud t (s)','Local / Cloud','Local S','Cloud S'],vals,y,
        [75,65,115,115,125,95,95],size=10.5,row_height=22,key='platform_comparison')-12
    y=p.text(c,'t为前100帧时间的三轮中位数；S为同一平台MOV时间中位数/当前规则时间中位数。跨平台时间比同时包含GPU、CPU、驱动与编译器差异，以及可能不同的数值轨迹，不能全部归因单项优化。',y,9.6)-10
    y=p.text(c,f'所选三场景合计：TOI相对MOV {collective["pure"]:.6f}×，GATE {collective["ours"]:.6f}×。这是所选三场景时间中位数之和的比值；长期精度结合第4、5页差异与参考重复性解释。',y,10)-10
    p.text(c,'跨平台初始位置/速度/质量/边界/物体编号完全相同。三角形排列顺序不同，连接关系和方向、四面体数据一致；后续浮点运算顺序及轨迹仍可能不同。',y,9.4)
    c.showPage()

    heading(c,'AutoDL：前100帧分项性能表','dt=0.04；rho=1e-4；105帧×3轮，汇总前100帧；时间单位s。',2,6)
    vals=[]
    for s in SCENES:
        for m in METHODS:
            r=idx[s,m]
            vals.append([s if m=='original' else '',p.LABELS[m]]+[p.fmt(r[k]) for k in ['assembly','pcg','ccd','line_search','Misc','TimeTot']]+
                [p.fmt(r['P_bar'],2),p.fmt(r['N_bar'],2),p.integer(r['P']),p.fmt(r['S_movement_over_rule'])])
    y=p.table(c,['Scene','Method','Hess','LSolver','CCD','LineS','Misc','TimeTot',p.BarLetter('P'),p.BarLetter('N'),'Ptot','S'],
        vals,p.PAGE_H-96,[65,80,58,65,58,58,58,73,60,60,72,55],size=11,row_height=24,key='cloud_phases')-18
    y=p.text(c,'Hess包含终止组装；LSolver包含PCG准备、同步、捕获和执行；Misc包含状态更新及窗口内未分类时间。各列独立取三轮中位数，不要求列和精确等于TimeTot。',y,10.3)-13
    y=p.text(c,'N为实际求解方向数，包含最终Movement判断方向；P为PCG迭代数，平均PCG迭代数=Ptot/Ntot，每帧平均方向数=Ntot/100。S以AutoDL同一程序的MOV规则为基线，区别于冻结StiffGIPC Base。',y,10.3)-13
    p.text(c,'正式计时为同步host求解窗口；初始化、导出、检查点hook及写盘单独记录。成本探针只在额外单步诊断运行启用，不混入这张正式性能表。',y,10.3)
    c.showPage()

    image_page(c,data/'visuals/all_scenes_frame100_r1.png','AutoDL：三规则独立轨迹第100帧',
        '第1轮；行：Movement only / Cumulative TOI .01 / Compensated + residual gate。',3,6,'主报告')

    heading(c,'第100步：共同检查点与独立轨迹','上表相同初始状态、冷恢复单步；下表从第0步独立演化。两种差异分别解释。',4,6)
    y=p.text(c,'共同检查点：MOV第1轮第99步后同一状态；N/P/t取3次中位数，位置差异取候选3次与参考3次的最差9配对。',p.PAGE_H-94,9.6)-5
    headers=['Scene','Rule','N','P','t (s)','RMS diff. (mm)','Max diff. (mm)']
    y=p.table(c,headers,table_rows(cold),y,[85,75,40,65,100,160,160],size=10.1,row_height=16.5,key='cold')-12
    y=p.text(c,'独立轨迹实际第100帧：前99步各自演化；t为该帧实际求解时间，差异取最差9配对。',y,9.6)-5
    y=p.table(c,headers,table_rows(trajectory),y,[85,75,40,65,100,160,160],size=10.1,row_height=16.5,key='trajectory')-9
    p.text(c,'冷恢复包含PCG/Graph准备及首次执行，不是纯稳态重放。较严参考非精确解，长期差异须结合参考自身重复性；完整轨迹图对应下表。',y,9.3)
    c.showPage()

    heading(c,'PCG工作量与参考重复性','同一平台内解释N/P的变化与PCG阶段均摊成本；时间取前100帧三轮中位数。',5,6)
    vals=[]
    for s in SCENES:
        for m in METHODS:
            r=idx[s,m]
            vals.append([s if m=='original' else '',p.LABELS[m],p.integer(r['N']),p.integer(r['P']),p.fmt(r['pcg'],3),
                p.fmt(r['pcg_stage_us_per_iteration'],2),p.fmt(r['TimeTot'],3),p.fmt(r['S_movement_over_rule'])])
    y=p.table(c,['Scene','Rule','Ntot','Ptot','LSolver (s)','LS/P (us)','TimeTot (s)','S'],vals,
        p.PAGE_H-94,[70,60,85,95,120,120,120,65],size=11,row_height=22,key='pcg_work')-14
    y=p.text(c,'较严参考自身三轮：第100帧三个两两配对的最差位置差异。',y,10)-6
    vals=[]
    for s in SCENES:
        selected=[r for r in noise if r['scene']==s and r['frame']=='100' and r['partition']=='all']
        assert len(selected)==3
        vals.append([s,p.fmt(max(float(r['position_rms_mm']) for r in selected),6),
                     p.fmt(max(float(r['position_max_mm']) for r in selected),6)])
    y=p.table(c,['Scene','Reference RMS diff. (mm)','Reference max diff. (mm)'],vals,y,
        [90,210,210],size=11,row_height=23,groups=0,key='reference_noise')-12
    p.text(c,'LS/P是整个线性求解阶段均摊成本，包含准备、归约、同步与捕获，不能解释为单个迭代核时间。方向数、迭代数及每帧峰值见附录；参考自身差异较大时，候选长期分歧不能全部归因停止规则。',y,10)
    c.showPage()

    heading(c,'记录开销、原始数据与核验','正式候选27次＋较严参考9次；共同检查点/开关63次；细分成本诊断9次。',6,6)
    overhead=p.rows(data/'recording_overhead_summary.csv')
    vals=[]
    for r in overhead:
        vals.append([r['scene'] if r['method']=='original' else '',p.LABELS[r['method']],r['pairs'],r['same_selected_counter_pairs'],
            p.fmt(r['median_solver_ratio_all_pairs'],4),p.fmt(r['median_solver_ratio_selected'],4) if r['median_solver_ratio_selected'] else '--'])
    y=p.table(c,['Scene','Rule','Pairs','Matched','ON/OFF: all','ON/OFF: matched'],vals,p.PAGE_H-94,
        [80,75,65,90,150,150],size=11,row_height=22,key='overhead')-17
    notes=[
        'Matched要求方向数、PCG迭代、Newton记录、能量/回退及Graph执行等选定计数相同。没有匹配样本记为--。开关时间比仍受调度和频率影响，不能当作已分离的精确记录开销。',
        '原始记录包括逐帧、逐Newton、步长alpha/beta、rho比、接触候选、Graph捕获/命中、残差监测、位置与实际速度、GPU采样、初始化/导出/写盘，以及第100步嵌套CostScope。嵌套父子区间不能相加。',
        '核验通过：3780条轨迹帧、36条第100帧、63个检查点输入、36个轨迹初态、162份OBJ坐标/拓扑无损。355个源码与输入文件与本机相同；初始状态与质量等一致，三角形仅排列顺序不同。',
        '首次启动适配因缺少Linux环境元数据在原生程序启动前中止，0帧；该证据保留在原批次。正式数据来自v2批次，使用相同干净构建二进制，未改动数值算法。',
        '附录提供六列关键帧0/25/50/75/100/105及三轮实际视觉图；data/提供展示源CSV，delivery_verification.json和PDF_manifest.json提供校验清单。'
    ]
    for note in notes:
        y=p.text(c,note,y,9.9)-11
    c.showPage();c.save()

    appendix=canvas.Canvas(str(out/'AutoDL_3Scene_Appendix.pdf'),pagesize=(p.PAGE_W,p.PAGE_H),pageCompression=1)
    appendix.setTitle('AutoDL三场景逐帧曲线与实际关键帧')
    page=1
    for s in SCENES:
        image_page(appendix,data/f'plots/{s.lower()}_performance.png',f'{s}：逐帧性能与参考差异',
            '105帧×3轮；曲线为中位数，阴影为三轮范围；虚线标记第100帧。',page,15);page+=1
    for repeat in (1,2,3):
        for s in SCENES:
            image_page(appendix,data/f'visuals/{s.lower()}_timeline_r{repeat}.png',f'{s}：关键帧，第{repeat}轮',
                'dt=0.04；三种停止规则；物理帧0 / 25 / 50 / 75 / 100 / 105。',page,15);page+=1
    for repeat in (1,2,3):
        image_page(appendix,data/f'visuals/all_scenes_frame100_r{repeat}.png',f'第100帧三场景总览，第{repeat}轮',
            '行：Movement only / Cumulative TOI .01 / Compensated + residual gate。',page,15);page+=1
    assert page==16
    appendix.save()
    for s in SCENES:
        c=canvas.Canvas(str(out/f'checkpoints/Checkpoint100_{s}.pdf'),pagesize=(p.PAGE_W,p.PAGE_H),pageCompression=1)
        heading(c,f'{s}：AutoDL第100步共同检查点','dt=0.04；相同MOV第1轮第99步后状态；冷恢复单步。',1,1)
        values=[r for r in table_rows(cold)[SCENES.index(s)*3:SCENES.index(s)*3+3]]
        for r,m in zip(values,METHODS):
            r[1]=p.FULL_LABELS[m]
        y=p.table(c,headers,values,p.PAGE_H-140,[85,255,40,65,100,160,160],size=12,row_height=38,groups=0,key=f'checkpoint_{s}')-25
        p.text(c,'N/P/t取候选3次中位数；RMS和最大点差异取候选3次与较严参考3次的最差9配对。较严参考为Movement only、rho=1e-6，外层阈值仍为0.01，非精确解。冷恢复时间包含PCG/Graph准备和首次执行。',y,11)
        c.showPage();c.save()
    (out/'data').mkdir()
    for name in ['phase_table_first100.csv','checkpoint_frame100_table.csv','trajectory_frame100_table.csv',
                 'recording_overhead_summary.csv','reference_repeat_dispersion.csv','run_costs.csv']:
        path=p.source(data/name)
        (out/'data'/name).write_bytes(path.read_bytes())
    (out/'delivery_verification.json').write_bytes((data/'delivery_verification_local.json').read_bytes())
    (out/'cross_platform_topology.json').write_bytes((data/'cross_platform_topology.json').read_bytes())
    evidence={}
    for file in sorted(out.rglob('*.pdf')):
        reader=PdfReader(file)
        assert len(reader.outline)==len(reader.pages)
        assert all(len(page.extract_text())>70 and '\ufffd' not in page.extract_text() for page in reader.pages)
        evidence[file.relative_to(out).as_posix()]={'pages':len(reader.pages),'sha256':p.sha(file)}
    manifest=dict(complete=True,inputs=p.INPUTS,display_rows=p.DISPLAY,pdfs=evidence,
        weighted_speedups=collective,solver_seconds_sum_of_scene_medians=totals,
        native_program_sha256=packet['programs']['active']['exe']['sha256'],
        source_proof=verified,topology_comparison=topology,rendered_pdf_review='pending')
    (out/'PDF_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(out),pdf_files=len(evidence),pages=sum(x['pages'] for x in evidence.values()),speedups=collective)))


if __name__=='__main__':
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument('--root',type=Path,required=True)
    args.add_argument('--out',type=Path,default=REPO/'output/pdf/stiffgipc_autodl3_20261013')
    a=args.parse_args()
    main(a.root.resolve(),a.out.resolve())
