"""Attribute actual Nsight GPU activities only after schema/symbol/correlation checks."""
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from prepare_cost_split import RAW,REPORT,read
from local_identity import record,verify,tree
from linux_runner import require,write_new
from analyze_cost_split import audit_outputs,csv_new
import numpy as np

PATTERNS={
    'MAS':['abd_preconditioner_apply_kernel','__buildMultiLevelR_optimized_new','_schwarzLocalXSym6','__collectFinalZ_new'],
    'reference':['dot_kernel','reduce_1Block_kernel','balanced_tiny_solve','gemv2N_kernel',
        'abd_preconditioner_apply_kernel','__buildMultiLevelR_optimized_new','_schwarzLocalXSym6','__collectFinalZ_new',
        'dot_kernel','reduce_1Block_kernel','balanced_tiny_solve','balanced_difference','gemv2N_kernel'],
    'fused':['dot_kernel','reduce_1Block_kernel','balanced_tiny_solve','balanced_input_projection',
        'abd_preconditioner_apply_kernel','__buildMultiLevelR_optimized_new','_schwarzLocalXSym6','__collectFinalZ_new',
        'dot_kernel','reduce_1Block_kernel','balanced_tiny_solve','balanced_output_projection']}
PHASES={
    'MAS':['native_MAS']*4,
    'reference':['ZT_projection']*2+['first_8d_solve','input_projection']+['native_MAS']*4+
        ['WT_projection']*2+['second_8d_solve']+['output_projection']*2,
    'fused':['ZT_projection']*2+['first_8d_solve','input_projection']+['native_MAS']*4+
        ['WT_projection']*2+['second_8d_solve','output_projection']}

TARGETS={'ZT_projection','WT_projection','first_8d_solve','second_8d_solve'}
def union_us(rows):
    """Elapsed coverage, preserving overlaps rather than assuming serial timestamps."""
    if not rows:return 0.
    ordered=sorted((r['start'],r['end']) for r in rows);start,end=ordered[0];total=0
    for a,b in ordered[1:]:
        require(b>=a,'Negative GPU interval')
        if a>end:total+=end-start;start,end=a,b
        else:end=max(end,b)
    return (total+end-start)/1000

def main():
    folder=RAW/'trace_analysis_retry1';require(not folder.exists(),'Trace analysis exists');folder.mkdir()
    normal,n_outputs,_=audit_outputs('profile');traced,t_outputs,_=audit_outputs('trace')
    require(len(normal['samples'])==len(traced['samples'])==18 and len(n_outputs)==len(t_outputs)==30,'Missing normal/traced data')
    require(read(RAW/'stability/audit.json')['selected_calls'] is not None,'Stable batch required')
    # The selected length is for future performance tests. These 32-call trace windows are diagnostics.
    result=read(RAW/'trace/result.json');verify([result['nsys_report'],result['sqlite']])
    db=sqlite3.connect('file:'+Path(result['sqlite']['path']).as_posix()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    tables={r[0] for r in db.execute('SELECT name FROM sqlite_master WHERE type="table"')};schema={}
    for table,columns in {
        'NVTX_EVENTS':['start','end','text','textId','globalTid','eventType'],
        'CUPTI_ACTIVITY_KIND_RUNTIME':['start','end','correlationId','nameId','globalTid','returnValue'],
        'CUPTI_ACTIVITY_KIND_KERNEL':['start','end','correlationId','globalPid','deviceId','contextId','streamId','graphNodeId','shortName','demangledName'],
        'CUPTI_ACTIVITY_KIND_MEMCPY':['start','end','correlationId','graphNodeId','bytes'],
        'CUPTI_ACTIVITY_KIND_MEMSET':['start','end','correlationId','graphNodeId','bytes'],
        'StringIds':['id','value']}.items():
        require(table in tables,'Missing Nsight table: '+table);schema[table]=[dict(r) for r in db.execute('PRAGMA table_info('+table+')')]
        require(set(columns)<=set(r['name'] for r in schema[table]),'Unexpected Nsight schema: '+table)
    strings=dict(db.execute('SELECT id,value FROM StringIds'));ranges={}
    for row in db.execute('SELECT rowid,* FROM NVTX_EVENTS WHERE end>start'):
        r=dict(row);tag=r['text'] or strings.get(r['textId'],'')
        if tag.startswith('cost/'):
            require(tag not in ranges and r['eventType']==59,'Ambiguous/wrong NVTX range');ranges[tag]=r
    require(set(ranges)=={r['tag'] for r in traced['samples']},'NVTX and application windows differ')
    sample_normal={r['tag']:r for r in normal['samples']};kernels=[];memory=[];launches=[];windows=[];overhead=[]
    for measured in traced['samples']:
        tag=measured['tag'];nvtx=ranges[tag];start,end=nvtx['start'],nvtx['end'];method=measured['method'];name=measured['name']
        require(measured['calls']==32 and measured['capture_calls']==1,'Unexpected traced work')
        runtime=[dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE start>=? AND end<=? ORDER BY start',(start,end))]
        runtime=[r for r in runtime if strings.get(r['nameId'],'').startswith('cudaGraphLaunch')]
        require(len(runtime)==32 and all(r['returnValue']==0 and r['globalTid']==nvtx['globalTid'] for r in runtime),'Graph launch API mismatch')
        correlations={r['correlationId']:r for r in runtime};require(len(correlations)==32,'Duplicate launch correlation')
        ks=[dict(r) for r in db.execute('SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=? ORDER BY start',(start,end))]
        require(len(ks)==32*len(PATTERNS[method]),'Kernel trace incomplete')
        require({r['correlationId'] for r in ks}==set(correlations),'Kernel/Graph launch correlation differs')
        require(len({(r['globalPid'],r['deviceId'],r['contextId'],r['streamId']) for r in ks})==1,'Mixed process/device/context/stream')
        memories=[]
        for table,kind in [('CUPTI_ACTIVITY_KIND_MEMCPY','copy'),('CUPTI_ACTIVITY_KIND_MEMSET','set')]:
            for r in db.execute('SELECT * FROM '+table+' WHERE start>=? AND end<=? ORDER BY start',(start,end)):
                value=dict(r);value['kind']=kind;memories.append(value)
        require(len(memories)==32*(4 if method=='reference' else 3),'Memory activity trace incomplete')
        require({r['correlationId'] for r in memories}==set(correlations),'Memory/Graph correlation differs')
        window_launches=[]
        for iteration,(corr,api) in enumerate(correlations.items(),1):
            values=[r for r in ks if r['correlationId']==corr];short=[strings[r['shortName']] for r in values]
            require(short==PATTERNS[method],'Ambiguous kernel sequence: '+repr(short))
            require(all(r['graphNodeId']>0 and r['end']>r['start'] for r in values),'Missing graph node or invalid kernel duration')
            phase_times={}
            for position,(r,phase) in enumerate(zip(values,PHASES[method])):
                r['phase']=phase
                duration=(r['end']-r['start'])/1000;phase_times[phase]=phase_times.get(phase,0)+duration
                kernels.append(dict(tag=tag,name=name,method=method,repeat=measured['repeat'],iteration=iteration,position=position+1,
                    phase=phase,kernel=strings[r['demangledName']],short_name=strings[r['shortName']],start_ns=r['start'],end_ns=r['end'],
                    duration_us=duration,correlation_id=corr,graph_node_id=r['graphNodeId'],blockX=r['blockX'],registers_per_thread=r['registersPerThread']))
            ms=sorted([r for r in memories if r['correlationId']==corr],key=lambda r:r['start']);copies=[r for r in ms if r['kind']=='copy'];sets=[r for r in ms if r['kind']=='set']
            require(len(copies)==(2 if method=='reference' else 1) and len(sets)==2,'Wrong per-launch memory structure')
            n=next(c['n'] for c in read(RAW/'input_manifest.json')['cases'] if c['name']==name)
            require(all(r['bytes']==n*8 for r in copies),'Global/input copy size differs')
            for index,r in enumerate(copies):r['phase']='input_projection' if method=='reference' and index==0 else 'native_MAS'
            for r in sets:r['phase']='native_MAS'
            for r in ms:
                require(r['end']>=r['start'] and r['graphNodeId']>0,'Invalid memory event')
                memory.append(dict(tag=tag,name=name,method=method,repeat=measured['repeat'],iteration=iteration,phase=r['phase'],kind=r['kind'],
                    bytes=r['bytes'],duration_us=(r['end']-r['start'])/1000,correlation_id=corr,graph_node_id=r['graphNodeId'],start_ns=r['start'],end_ns=r['end']))
            activities=sorted(values+ms,key=lambda r:r['start'])
            span=(max(r['end'] for r in activities)-min(r['start'] for r in activities))/1000
            summed=sum((r['end']-r['start'])/1000 for r in activities);kernel_busy=sum(phase_times.values());memory_busy=summed-kernel_busy
            busy=union_us(activities);kernel_union=union_us(values);memory_union=union_us(ms)
            target=[r for r in values if r['phase'] in TARGETS];other=[r for r in activities if r.get('phase') not in TARGETS]
            solves=[r for r in values if r['phase'] in {'first_8d_solve','second_8d_solve'}]
            transpose=[r for r in values if r['phase'] in {'ZT_projection','WT_projection'}]
            row=dict(tag=tag,name=name,method=method,repeat=measured['repeat'],iteration=iteration,correlation_id=corr,
                span_us=span,kernel_busy_us=kernel_busy,memory_busy_us=memory_busy,internal_gap_us=span-busy,
                kernel_union_us=kernel_union,memory_union_us=memory_union,activity_union_us=busy,
                interval_overlap_us=summed-busy,kernel_overlap_us=kernel_busy-kernel_union,
                target_union_us=union_us(target),solve_union_us=union_us(solves),transpose_union_us=union_us(transpose),
                target_exclusive_us=busy-union_us(other))
            row.update({k+'_us':v for k,v in phase_times.items()})
            row.update({k+'_union_us':union_us([r for r in values if r['phase']==k]) for k in phase_times})
            launches.append(row);window_launches.append(row)
        window_activities=ks+memories;span=(max(r['end'] for r in window_activities)-min(r['start'] for r in window_activities))/1000
        summed_busy=sum(r['kernel_busy_us']+r['memory_busy_us'] for r in window_launches);window_union=union_us(window_activities)
        windows.append(dict(tag=tag,name=name,method=method,repeat=measured['repeat'],calls=32,span_us=span,
            kernel_busy_us=sum(r['kernel_busy_us'] for r in window_launches),memory_busy_us=sum(r['memory_busy_us'] for r in window_launches),
            activity_union_us=window_union,interval_overlap_us=summed_busy-window_union,
            activity_gap_us=span-window_union,application_event_us=measured['event_ms']*1000,
            event_minus_activity_span_us=measured['event_ms']*1000-span))
        normal_sample=sample_normal[tag];require(normal_sample['calls']==32,'Normal work differs')
        overhead.append(dict(tag=tag,name=name,method=method,repeat=measured['repeat'],normal_event_us_per_call=normal_sample['event_ms']*1000/32,
            traced_event_us_per_call=measured['event_ms']*1000/32,event_ratio=measured['event_ms']/normal_sample['event_ms'],
            host_ratio=measured['host_ms']/normal_sample['host_ms']))
    require(len(kernels)==5568 and len(launches)==576 and len(windows)==18,'Incomplete traced work')
    summary=[]
    for name,method in dict.fromkeys((r['name'],r['method']) for r in launches):
        rows=[r for r in launches if (r['name'],r['method'])==(name,method)];require(len(rows)==96,'Missing repeated launch')
        med={k:float(np.median([r.get(k,0.) for r in rows])) for k in ['span_us','kernel_busy_us','memory_busy_us','internal_gap_us',
            'kernel_union_us','memory_union_us','activity_union_us','interval_overlap_us','kernel_overlap_us','target_union_us','target_exclusive_us']+[p+'_us' for p in dict.fromkeys(PHASES[method])]}
        sum_span=sum(r['span_us'] for r in rows)
        summary.append(dict(name=name,method=method,n=96,**med,
            transpose_and_solve_share_of_span=sum(r['target_union_us'] for r in rows)/sum_span,
            solve_only_share_of_span=sum(r['solve_union_us'] for r in rows)/sum_span,
            transpose_only_share_of_span=sum(r['transpose_union_us'] for r in rows)/sum_span,
            target_exclusive_share_of_span=sum(r['target_exclusive_us'] for r in rows)/sum_span))
    tables_dir=REPORT/'tables'
    csv_new(tables_dir/'profile_CPU.csv',n_outputs);csv_new(tables_dir/'trace_CPU.csv',t_outputs)
    for filename,rows in [('trace_kernels.csv',kernels),('trace_memory.csv',memory),('trace_launches.csv',launches),('trace_windows.csv',windows),('trace_summary.csv',summary),('trace_overhead.csv',overhead)]:csv_new(tables_dir/filename,rows)
    write_new(folder/'schema.json',schema)
    write_new(folder/'summary.json',dict(passed=True,attribution_verified=True,ranges=18,launches=576,kernels=5568,memory_events=len(memory),
        normal_checked_outputs=len(n_outputs),traced_checked_outputs=len(t_outputs),
        maximum_CPU_relative=max(r['CPU_relative'] for r in n_outputs+t_outputs),summary=summary,overhead=overhead,
        performance_certified=False,inputs=tree(RAW/'profile')+tree(RAW/'trace'),source=record(Path(__file__))))
    print(dict(attribution=True,launches=576,kernels=5568,memory=len(memory),summary=summary))
if __name__=='__main__':main()
