"""Create additional spreadsheet-friendly tables from sealed P3 analysis."""
from collections import defaultdict
import csv
import json
from pathlib import Path
from p3_balanced_analyze import ROOT,PHASES
from local_identity import record
from linux_runner import require,write_new
from m0_analyze import csv_write


def main():
    folder=ROOT/'reports/innovation_p3_balanced_20261015/analysis'
    summary=json.loads((folder/'summary.json').read_text())
    with (folder/'frames.csv').open(encoding='utf-8-sig',newline='') as stream:frames=list(csv.DictReader(stream))
    with (folder/'geometry.csv').open(encoding='utf-8-sig',newline='') as stream:geometry=list(csv.DictReader(stream))
    differences=json.loads((folder/'state_differences.json').read_text())
    require(len(frames)==len(geometry)==600 and len(differences)==300,'Missing full-frame data')
    presentation=[];deltas=[];quality=[];paired_frames=[];flat_differences=[];diagnostics=[]
    for pair in summary['pairs']:
        scene=pair['scene'];f,w=pair['F'],pair['W']
        for arm,run in (('F',f),('W',w)):
            presentation.append(dict(scene=scene,method=arm,frames=100,
                **{phase+'_seconds':run[phase+'_ms']/1000 for phase in PHASES},
                solver_seconds=run['solver_ms']/1000,mean_solver_ms=run['mean_solver_ms'],
                P_bar=run['mean_pcg_iterations_per_solve'],N_bar=run['mean_solves_per_frame'],
                N_total=run['N'],P_total=run['P'],speedup_vs_F=1 if arm=='F' else pair['solver_speedup']))
            rows=[r for r in geometry if r['run']==run['run']]
            def values(key):return [float(r[key]) for r in rows if r.get(key) not in ('',None)]
            quality.append(dict(scene=scene,method=arm,max_cloth_stretch=max(values('max_cloth_stretch')),
                min_fem_J=min(values('fem_min_J'),default=None),
                max_fem_nonpositive=max(values('fem_nonpositive')) if values('fem_min_J') else None,
                max_fixed_drift_m=max(values('fixed_drift_m'),default=None)))
        deltas.append(dict(scene=scene,solver_delta_seconds=(w['solver_ms']-f['solver_ms'])/1000,
            **{phase+'_delta_seconds':(w[phase+'_ms']-f[phase+'_ms'])/1000 for phase in PHASES},
            N_delta_percent=100*(w['N']/f['N']-1),P_delta_percent=100*(w['P']/f['P']-1),
            pcg_stage_ms_per_iteration_F=f['pcg_ms']/f['P'],pcg_stage_ms_per_iteration_W=w['pcg_ms']/w['P']))
        for i in range(1,101):
            a,b=[next(r for r in frames if r['scene']==scene and int(r['frame'])==i and r['mode']==mode)
                for mode in ('fused','two_stage')]
            d=next(r for r in differences if r['scene']==scene and r['frame']==i)
            free=d['position']['partitions']['free'];cloth=d['position']['partitions']['cloth_dynamic']
            paired_frames.append(dict(scene=scene,frame=i,F_solver_ms=float(a['solver_ms']),W_solver_ms=float(b['solver_ms']),
                frame_time_ratio=float(a['solver_ms'])/float(b['solver_ms']),F_N=int(a['N']),W_N=int(b['N']),
                F_P=int(a['P']),W_P=int(b['P']),free_position_rms_difference_mm=free['rms'],
                free_position_max_difference_mm=free['max'],cloth_position_rms_difference_mm=cloth['rms'],
                cloth_position_max_difference_mm=cloth['max']))
        q=next(r for r in summary['queries'] if r['scene']==scene)
        p={mode:next(r for r in summary['profiles'] if r['scene']==scene and r['mode']==mode) for mode in ('fused','two_stage')}
        diagnostics.append(dict(scene=scene,query_CPU_speedup=q['timing']['cpu']['paired_geometric_speedup'],
            query_CPU_F_first=q['timing']['cpu']['first_mode_speedup']['F'],query_CPU_W_first=q['timing']['cpu']['first_mode_speedup']['W'],
            separate_GPU_speedup=q['timing']['gpu']['paired_geometric_speedup'],
            profile_F_EE_frame_fraction=p['fused']['ee_frame_gpu_envelope_fraction'],
            profile_W_EE_frame_fraction=p['two_stage']['ee_frame_gpu_envelope_fraction']))
    for d in differences:
        for kind in ('position','actual_velocity'):
            for partition,v in d[kind]['partitions'].items():
                flat_differences.append(dict(scene=d['scene'],frame=d['frame'],kind=kind,partition=partition,
                    unit=d[kind]['unit'],**v))
    for name,rows in (('presentation_table.csv',presentation),('phase_deltas.csv',deltas),('quality_summary.csv',quality),
        ('comparison_frames.csv',paired_frames),('state_differences.csv',flat_differences),('diagnostic_summary.csv',diagnostics)):
        csv_write(folder/name,rows)
    write_new(folder/'exports.json',dict(exporter=record(Path(__file__)),presentation=presentation,deltas=deltas,
        quality=quality,diagnostics=diagnostics,source_records=[record(folder/name) for name in
            ('summary.json','frames.csv','geometry.csv','state_differences.json')]))
    print(json.dumps(dict(exported=True,diagnostics=diagnostics,deltas=deltas,quality=quality),ensure_ascii=False))


if __name__=='__main__':main()
