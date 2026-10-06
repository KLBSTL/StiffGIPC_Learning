"""Single evidence catalogue; legacy artefacts stay immutable."""
import json
from datetime import datetime, timezone
from config import ROOT, read, sha

def main():
    rows=[]
    matrix=ROOT/'reports/BUNNY_COMPONENTS_MATRIX.json'
    if matrix.exists():
        for run in read(matrix)['runs']:
            result=run['result'];name=run['name']
            rows.append({'id':name,'origin':'frozen_component_matrix','scene':'bunny_cloth_bunny_l',
                         'status':'pending' if result['status']=='completed' else 'failed',
                         'completion':result['status'],'performance_certified':False,'quality_certified':False,
                         'reason':'Completed diagnostic comparison; strict quality and controlled GPU gates not certified.',
                         'frames':result['recorded_frames'],'solver_seconds':result.get('solver_seconds'),
                         'evidence':['runs/local/'+name+'/requested.json','runs/local/'+name+'/result.json']})
    for version in (55,56):
        report=ROOT/f'reports/V{version}_FIXTURES.json'
        if report.exists():rows.append({'id':f'v{version}_local_action','origin':'frozen_fixture',
           'status':'rejected','reason':'Correctness passed; no stable measured whole-linear improvement. One hotspot-gated retest allowed.',
           'performance_certified':False,'evidence':[report.relative_to(ROOT).as_posix()]})
    for result_path in sorted((ROOT/'runs/active').glob('*/result.json')):
        folder=result_path.parent;result=read(result_path);req_path=folder/'requested.json'
        req=read(req_path) if req_path.exists() else {};c=req.get('expanded_config',{})
        is_fixture='checks' in result
        state=('accepted' if result['passed'] else 'failed') if is_fixture else ('pending' if result['status']=='completed' else 'failed')
        rows.append({'id':folder.name,'origin':'active_fixture' if is_fixture else 'active_run',
                     'status':state,'status_scope':'fixture correctness only' if is_fixture else 'performance/physical-quality gates pending',
                     'scene':c.get('scene'),'frames':result.get('recorded_frames'),
                     'completion':result.get('status'),'solver_seconds':result.get('solver_seconds'),
                     'config_sha256':req.get('config_sha256'),'exe_sha256':req.get('exe_sha256',result.get('exe_sha256')),
                     'diagnostics':c.get('diagnostics'),'performance_certified':False,'quality_certified':False,
                     'evidence':[str(p.relative_to(ROOT).as_posix()) for p in
                        [req_path,folder/'resolved_config.json',folder/'build_manifest.json',result_path] if p.exists()]})
    decisions_path=ROOT/'reports/active/EXPERIMENT_DECISIONS.json'
    decisions=read(decisions_path) if decisions_path.exists() else {}
    for row in rows:
        if row['id'] in decisions:
            row.update(decisions[row['id']])
            row['evidence'].append(decisions_path.relative_to(ROOT).as_posix())
    target=ROOT/'reports/active/EXPERIMENT_INDEX.json'
    # Curated aggregates and historical decisions are evidence, too. Do not
    # silently discard them when the filesystem scan discovers new runs.
    if target.exists():
        previous=read(target)['entries'];known={r['id'] for r in previous}
        if len(known)!=len(previous):raise ValueError('Duplicate historical experiment id')
        rows=previous+[r for r in rows if r['id'] not in known]
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate experiment id')
    target.write_text(json.dumps({'schema':1,'updated_utc':datetime.now(timezone.utc).isoformat(),
         'status_vocabulary':['accepted','rejected','failed','pending'],'entries':rows},indent=2))
    print(json.dumps({'index':str(target),'entries':len(rows)}))

if __name__=='__main__':main()
