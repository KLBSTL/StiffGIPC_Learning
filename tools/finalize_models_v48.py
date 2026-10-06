"""Aggregate gates and local model-error attribution; verify frozen evidence."""
import hashlib,json
from pathlib import Path
import numpy as np
from event_model_v48 import EventModel
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
out=ROOT/'reports/FINAL_V48_20261004.json';assert not out.exists()
smoke=read(ROOT/'reports/MODELS_V48_SMOKE_R2.json');models=read(ROOT/'reports/MODELS_V48_BUNNY_R2.json')
curves=read(ROOT/'reports/CURVATURE_V48_BUNNY.json');assert smoke['passed'] and models['passed'] and curves['passed']
for report in [smoke,models,curves]:
    for name,digest in report['helper_sha256'].items():assert sha(ROOT/'tools'/name)==digest
manifest=read(ROOT/'manifests/perf_v47_local.json')
for f in manifest['files']:assert sha(ROOT/f['path'])==f['sha256']
for name,f in manifest['binaries'].items():assert sha(ROOT/name)==f['sha256']
rows=[]
for e,c in zip(models['events'],curves['events']):
    assert e['event']==c['event']
    original=read(ROOT/'runs/local/v47_bunny_graph/events'/f"{e['event']}_line_search.json")
    qdelta=sum(e['model_delta_terms'].values());error=e['total_energy_delta']-qdelta
    al_model=e['model_delta_terms']['linear_al']+e['model_delta_terms']['quadratic_al']
    rows.append({'event':e['event'],'total_quadratic_model_delta':qdelta,'total_actual_delta':e['total_energy_delta'],
        'total_model_error':error,'fem_model_error':c['fem_nonlinear_model_error'],
        'al_model_error':e['terms']['al']['energy_delta']-al_model,
        'other_model_error':error-c['fem_nonlinear_model_error']-(e['terms']['al']['energy_delta']-al_model),
        'fem_fraction_of_model_error':c['fem_nonlinear_model_error']/error,
        'other_gradient_norm_fraction':e['unseparated_other_gradient_norm']/e['b_norm'],
        'curvatures':{k:c[k] for k in ['exact_fem_curvature','projected_fem_curvature','al_curvature','unseparated_other_curvature']}})
m=EventModel(ROOT/'runs/local/v47_bunny_graph/events/e4');slopes=[]
for label,z in [('original',m.load('solution')),('accurate',np.fromfile(ROOT/'reports/REFERENCE_V47_E4.bin',dtype='<f8'))]:
    d=m.physical(z)
    for r in [0,.0078125,.015625]:
        x=m.x-r*d;ea,ga,_=m.al(x,True);ef,gf,_=m.fem(x,True)
        slopes.append({'direction':label,'r':r,'al_plus_fem_energy':ea+ef,'al_plus_fem_derivative':float(np.sum((ga+gf)*(-d)))})
checks=[c for r in models['events'] for c in r['checks']]
result={'source_binary_verified_unchanged':True,'source_files_verified':len(manifest['files']),
    'bunny_checks':len(checks),'smoke_checks':sum(len(r['checks']) for r in smoke['events']),
    'all_gates_passed':all(c['passed'] for c in checks),'max_gpu_energy_abs_error':max(c['max_abs_error'] for c in checks if 'gpu_energy' in c['name']),
    'max_curvature_gradient_fd_relative_error':max(e['gradient_fd_relative_error'] for e in curves['events']),
    'attribution':rows,'two_component_slopes':slopes,
    'input_reports_sha256':{p.name:sha(p) for p in [ROOT/'reports/MODELS_V48_SMOKE_R2.json',ROOT/'reports/MODELS_V48_BUNNY_R2.json',ROOT/'reports/CURVATURE_V48_BUNNY.json']},
    'reference_solution_sha256':sha(ROOT/'reports/REFERENCE_V47_E4.bin'),
    'helper_sha256':{p.name:sha(p) for p in [ROOT/'tools/event_model_v48.py',ROOT/'tools/analyze_models_v48.py',ROOT/'tools/curvature_models_v48.py',Path(__file__)]}}
out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
