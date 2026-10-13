"""Collect immutable diagnosis evidence; fail if native identity changed."""
from pathlib import Path
import argparse
import json
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2]
for folder in ('local','bench'):sys.path.insert(0,str(ROOT/'tools'/folder))
from config import read
from linux_runner import require
from local_identity import record,verify,verify_active_sources


def cohort(path):
    frames=read(path)['frames'][:112]
    require(len(frames)==112,'Completed common cohort required')
    newton=[n for f in frames for n in f['newton']]
    outer=[t for f in frames for t in f['toi']]
    phases={k:sum(n['profile_ms'][k] for n in newton)/1000 for k in ('assembly','pcg','line_search')}
    phases.update({k:sum(t['profile_ms'][k] for t in outer)/1000 for k in ('active_update','safe_ccd','safe_update')})
    return dict(evidence=record(path),frames=112,directions=len(newton),outer=len(outer),
        pcg=sum(n['pcg']['iterations'] for n in newton),seconds=phases,
        non_pcg_seconds=sum(v for k,v in phases.items() if k!='pcg'),
        cautions=['GIPC_PROFILE adds synchronization; diagnostic stage accounting only.',
                  'Same frame range, different trajectories; not a fixed-work causal timing comparison.',
                  'Excluded frame 113 of MAS run contains fixed-study/event overhead.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/DEEP_TOI_RESULTS_FINAL.json')
    args=parser.parse_args()
    destination=args.output.resolve()
    require(destination.parent==(ROOT/'reports').resolve(),'Report must stay in the reports directory')
    require(not destination.exists(),'Do not overwrite final diagnosis')
    identity_path=ROOT/'build/audit_corrected_frozen/snapshot_identity.json'
    identity=read(identity_path);verify_active_sources(identity)
    verify(identity['sources']+identity['build_evidence']+identity.get('objects',[])+[identity['exe']]+identity['dlls'])
    rows=[];evidence=[];run_identities=[]
    for session in ('deep_toi_controls_20261009','deep_toi_controls_remaining_20261009','deep_toi_diagonal_20261009'):
        ledger_path=ROOT/'runs'/session/'ledger.json';ledger=read(ledger_path);evidence.append(record(ledger_path))
        run_identities.append(record(ROOT/'runs'/session/'identity.json'))
        for run in ledger['runs']:
            require(run['result']['cleanup_owned_job_empty'],'Native job did not close')
            if run['result']['status']=='completed':
                require('summary' in run and 'analysis_error' not in run,'Incomplete successful-run analysis')
            rows.append(dict(session=session,name=run['task']['name'],status=run['result']['status'],
                frames=run['result']['recorded_frames'],summary=run.get('summary'),
                native_environment=run['task']['diagnostic_environment'],
                initial_input=run.get('initial_input'),trajectory=run.get('trajectory')))
    base=next(r for r in rows if r['name']=='base')['summary']['solver_seconds']
    profiles=dict(MAS=cohort(ROOT/'runs/deep_toi_native_allowed_20261009/linear113/output/stats.json'),
        diagonal=cohort(ROOT/'runs/deep_toi_diagonal_profile_20261009/toi_diagonal_profile/output/stats.json'))
    systems=[]
    for file in sorted((ROOT/'runs/deep_toi_native_allowed_20261009').glob('*/fixed/*_study.json')):
        d=read(file);require(d['system_unchanged'] and d['primary_restored_bitwise'],'Unprotected fixed study')
        systems.append(dict(evidence=record(file),frame=d['frame'],direction=d['direction'],
            operators=d['operator_mean_us'],runs=d['runs'],system_unchanged=True,primary_restored_bitwise=True))
    cpu_path=ROOT/'runs/deep_contact_common_norm_20261009/summary.json'
    cpu=read(cpu_path)
    schur_path=ROOT/'runs/deep_schur_coarse_20261009/analysis.json'
    schur=read(schur_path)
    precision_path=ROOT/'runs/deep_precision_20261009/analysis.json'
    precision=read(precision_path)
    diagnostic_ledgers=[]
    for session in ('deep_toi_native_allowed_20261009','deep_toi_diagonal_profile_20261009'):
        path=ROOT/'runs'/session/'ledger.json';ledger=read(path)
        require(all(r['result']['status']=='completed' and r['result']['cleanup_owned_job_empty']
            and 'analysis_error' not in r for r in ledger['runs']),'Unclosed or failed native diagnostic')
        diagnostic_ledgers.append(record(path))
    report=dict(native_identity=record(identity_path),native_exe=identity['exe'],controller=record(__file__),
        full_runs=rows,ledger_evidence=evidence,run_identity_evidence=run_identities,
        diagnostic_ledger_evidence=diagnostic_ledgers,profiles=profiles,fixed_systems=systems,
        scan_evidence=record(ROOT/'runs/deep_toi_scan_20261009/scan.json'),
        CPU_contact_evidence=record(cpu_path),CPU_contacts=cpu['events'],
        schur_evidence=record(schur_path),schur={k:v for k,v in schur.items() if k not in ('PCG_studies','evidence','operator_evidence')},
        schur_PCG=[{k:v for k,v in r.items() if k!='history'} for r in schur['PCG_studies']],
        precision_evidence=record(precision_path),
        precision={k:v for k,v in precision.items() if k not in ('cases','evidence','operator_evidence')},
        precision_PCG=[{k:v for k,v in r.items() if k!='history'} for r in precision['cases']],
        observed_base_seconds=base,target_2x_seconds=base/2,
        fixed_work_budget=dict(MAS_non_PCG_first112=profiles['MAS']['non_pcg_seconds'],
            diagonal_non_PCG_first112=profiles['diagonal']['non_pcg_seconds'],
            interpretation='With these measured trajectory/stage costs held fixed, eliminating all PCG/preparation cost cannot reach the observed baseline/2 budget. This is conditional diagnostic accounting, not an impossibility theorem.'),
        production_native_changes=False,quality_certified=False,performance_certified=False,
        target_2x_achieved=False)
    with destination.open('x',encoding='utf-8') as stream:json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(full_runs=[dict(name=r['name'],status=r['status'],frames=r['frames'],
        solver_seconds=(r['summary'] or {}).get('solver_seconds')) for r in rows],
        target_seconds=base/2,schur=report['schur_PCG'],budget=report['fixed_work_budget']),indent=2))


if __name__=='__main__':main()
