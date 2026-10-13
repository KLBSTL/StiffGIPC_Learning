"""Independently check delivered metrics and exact native-mesh exports; build QA sheets."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageStat

ROOT=Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def check(ok,message):
    if not ok:
        raise RuntimeError(message)


def verify(session,tables,figures,output):
    check(not output.exists(),'Preserve previous verification output')
    output.mkdir(parents=True)
    summary=read(tables/'summary.json');manifest=read(figures/'render_manifest.json')
    ledger=read(session/'performance_delivery_ledger.json')
    check(summary['complete'] and ledger['complete'] and manifest['complete'],'Incomplete delivery')
    scenes=summary['selected_scenes']
    check(scenes==ledger['selected_scenes']==manifest['selected_scenes'],'Scene scope differs')
    check(len(scenes)==6 and 'Stack' not in scenes,'Six-scene approved scope required')
    check(summary['excluded_scenes']==manifest['excluded_scenes']==['Stack'],'Stack exclusion missing')
    check(manifest['camera'].get('depth_order_smoke_passed') is True,'Depth-tested renderer smoke missing')
    renderer=manifest['renderer_script']
    check(digest(Path(renderer['path']))==renderer['sha256'],'Depth renderer source changed')
    check(len(ledger['runs'])==summary['formal_runs']==90,'Formal count differs')
    raw={}
    for record in ledger['runs']:
        task=record['task'];folder=session/task['name'];result=read(folder/'result.json')
        check(result['status']=='completed' and result['recorded_frames']==100,'Incomplete native run')
        with (folder/'trace/frames.csv').open() as stream:
            rows=list(csv.DictReader(stream))
        frames=read(folder/'output/stats.json')['frames']
        check(len(rows)==len(frames)==100,'Native frame coverage differs')
        check([int(row['frame']) for row in rows]==list(range(1,101)),'Physical frame order differs')
        check(all(f.get('newton_exit')!='iteration_limit' for f in frames),'Newton limit')
        pcg=[n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
        check(pcg and not any(p.get('breakdown') or p.get('iteration_limit') for p in pcg),'PCG failure')
        native=dict(TimeTot=sum(float(row['solver_ms']) for row in rows)/1000,
                    N_tot=len(pcg),P_tot=sum(p['iterations'] for p in pcg))
        for field,value in native.items():
            check(value==record['observation'][field],f'Metric mismatch: {task["name"]}/{field}')
        for name,value in record['observation']['raw_files'].items():
            check(digest(folder/name)==value,'Raw file changed: '+str(folder/name))
        raw[task['scene_key'],task['arm'],task['repeat']]=native
    for group in summary['groups']:
        for field in ('TimeTot','N_tot','P_tot'):
            check(group[field]==statistics.median(raw[group['scene'],group['method'],r][field] for r in (1,2,3)),
                  f'Grouped metric mismatch: {group["scene"]}/{group["method"]}/{field}')
    for arm in ('old','ours'):
        ratios=[sum(raw[s,'stiff',r]['TimeTot'] for s in scenes)/sum(raw[s,arm,r]['TimeTot'] for s in scenes)
                for r in (1,2,3)]
        check(ratios==summary['aggregate_speedups'][arm]['raw'],'Aggregate raw mismatch')
        check(statistics.median(ratios)==summary['aggregate_speedups'][arm]['median'],'Aggregate median mismatch')
    for stem,expected in [('phase_table_current',12),('phase_table_frozen',12),('phase_table_both',18),
                          ('graph_table_old',6),('graph_table_current',6),('checkpoint_table',6)]:
        with (tables/(stem+'.csv')).open(encoding='utf8') as stream:
            check(len(list(csv.DictReader(stream)))==expected,'Derived table row count differs')
        check((tables/(stem+'.tex')).is_file(),'Missing TeX table')
        check((tables/(stem+'.pdf')).read_bytes().startswith(b'%PDF-'),'Invalid table PDF')
        with Image.open(tables/(stem+'.png')) as image:
            image.verify()
    continuation=read(session/'continuation_manifest.json')
    for evidence in (continuation['controller'],continuation['prior_ledger']):
        check(digest(Path(evidence['path']))==evidence['sha256'],'Continuation evidence changed')
    ids=read(session/'identities.json');file_count=0
    for program in ids['programs'].values():
        records=program['sources']+program['build_evidence']+[program['exe']]+program['dlls']+program.get('objects',[])
        for evidence in records:
            check(digest(Path(evidence['path']))==evidence['sha256'],'Program identity changed')
        file_count+=len(records)
    check(digest(Path(ids['controller']['path']))==ids['controller']['sha256'],'Original controller changed')
    check(len(manifest['figures'])==96 and len(manifest['evidence'])==306,'Unexpected visual count')
    dimensions={};pngs=[]
    for name in manifest['figures']:
        path=figures/name
        if path.suffix=='.pdf':
            check(path.read_bytes().startswith(b'%PDF-'),'Invalid figure PDF')
        else:
            with Image.open(path) as image:
                image.load();check(min(image.size)>500,'Small or partial PNG')
                check(max(ImageStat.Stat(image.convert('RGB')).stddev)>1,'Blank figure')
                dimensions[name]=list(image.size)
            pngs.append(path)
    for evidence in manifest['evidence']:
        check(evidence['scene'] in scenes,'Deferred scene illustration')
        state=Path(evidence['state']);obj=figures/evidence['obj']
        check(digest(state)==evidence['state_sha256'],'Illustrated state changed')
        check(digest(state.parent/'topology.bin')==evidence['topology_sha256'],'Topology changed')
        check(digest(obj)==evidence['obj_sha256'],'OBJ changed')
        vertices=[];faces=[]
        with obj.open(encoding='ascii') as stream:
            for line in stream:
                if line.startswith('v '):vertices.append([float(x) for x in line.split()[1:]])
                elif line.startswith('f '):faces.append([int(x)-1 for x in line.split()[1:]])
        check(np.array_equal(np.array(vertices),np.fromfile(state,dtype='<f8').reshape(-1,3)),
              'Exported geometry differs from actual world coordinates')
        topology=np.fromfile(state.parent/'topology.bin',dtype='<u4');nv,nf,_=map(int,topology[:3])
        check(len(vertices)==nv and np.array_equal(np.array(faces),topology[3:3+3*nf].reshape(-1,3)),
              'Exported connectivity differs from native geometry')
    sheets=[]
    for repeat in (1,2,3):
        selected=sorted(p for p in pngs if p.stem.endswith(f'_r{repeat}'))
        check(len(selected)==16,'Repeat illustration coverage differs')
        for batch in range(2):
            canvas=Image.new('RGB',(1800,1440),'white');draw=ImageDraw.Draw(canvas)
            for index,path in enumerate(selected[batch*8:(batch+1)*8]):
                x=(index%2)*900;y=(index//2)*360
                with Image.open(path) as image:
                    thumb=image.convert('RGB');thumb.thumbnail((890,328))
                    canvas.paste(thumb,(x+(900-thumb.width)//2,y+24))
                draw.text((x+10,y+4),path.name,fill='black')
            target=output/f'qa_r{repeat}_{batch+1}.png';canvas.save(target);sheets.append(target.name)
    for name in ('run_combination_suite.py','continue_combination_suite.py','analyze_combination_suite.py',
                 'render_combination_suite.py','render_combination_depth.py','report_combination_suite.py',
                 'verify_combination_delivery.py'):
        ast.parse((ROOT/'tools/local'/name).read_text(encoding='utf8'),filename=name)
    receipt=dict(passed=True,formal_runs=90,formal_frames=9000,verified_program_files=file_count,
                 raw_metrics_independently_recomputed=True,continuation_identity_unchanged=True,
                 exact_world_coordinate_obj_exports=306,figure_pngs=48,figure_pdfs=48,table_sets=6,
                 depth_order_smoke_passed=True,
                 selected_scenes=scenes,excluded_scenes=['Stack'],qa_contact_sheets=sheets,
                 png_dimensions=dimensions,script_sha256=digest(Path(__file__)),
                 summary_sha256=digest(tables/'summary.json'),render_sha256=digest(figures/'render_manifest.json'),
                 physical_quality_certified=False,performance_certified=False)
    (output/'verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('png_dimensions','script_sha256','summary_sha256','render_sha256')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('session','tables','figures','output'):
        parser.add_argument('--'+field,type=Path,required=True)
    args=parser.parse_args()
    verify(args.session.resolve(),args.tables.resolve(),args.figures.resolve(),args.output.resolve())
