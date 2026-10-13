"""Bounded Windows setup/checkpoint checks; never run the long AutoDL matrix."""
import argparse
import json
import re
from pathlib import Path
import sys
from protocol import ROOT,plan,env_for,read,sha,write,require
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench'),str(ROOT/'tools/diagnostic')]
import windows_runner
from local_identity import verify,record,tree,project_map,compiler_records,norm,text
from config import digest
from run_suite import checkpoint_evidence


def local_program_identity(build):
    # Only four explicitly restored runtime caches were added after compilation.
    # Do not edit the original build snapshot or relax the general identity tool.
    before=read(build/'build_inputs_before.json');built=read(build/'windows_manifest.json')
    require(before==built['source_inventory'] and built['status']=='completed','Build inventories disagree')
    verify(before+[built['exe']]+built['objects']+built['dlls'])
    sources=sum((tree(ROOT/n) for n in ('StiffGIPC','Assets','MeshProcess')),[])+[record(ROOT/'CMakeLists.txt')]
    old={norm(r['path']):r for r in before};added=[r for r in sources if norm(r['path']) not in old]
    restore=read(ROOT/'runs/paper_suite_prepare_20261010/restored_caches.json')['files']
    approved={norm(r['target']):r for r in restore}
    require({norm(r['path']) for r in added}==set(approved),'Unexpected post-build addition')
    for r in added:
        expected=approved[norm(r['path'])]
        require(r['sha256']==expected['sha256'] and r['bytes']==expected['bytes']
            and record(Path(expected['source']))['sha256']==r['sha256'],'Runtime cache differs from frozen baseline')
    require(all(old.get(norm(r['path']))==r for r in sources if norm(r['path']) not in approved),'Compiled source or old runtime input changed')
    mapping=project_map(ROOT,build);tracking=build/'gipc.dir/Release/gipc.tlog'
    link=text(tracking/'link.read.1.tlog');log=text(build/'build.log');cl=text(tracking/'CL.command.1.tlog')
    require(set(built['translation_units'])=={r['source'] for r in mapping},'Native source coverage differs')
    for r in mapping:
        require(norm(r['object']) in norm(link),'Object absent from actual link')
        require(norm(r['source']) in norm(log if r['kind']=='CudaCompile' else cl),'Actual compile command missing')
    evidence=[build/'build_inputs_before.json',build/'windows_manifest.json',build/'build.log',build/'CMakeCache.txt',build/'gipc.vcxproj',*tracking.glob('*.tlog')]
    return dict(kind='active',exe=built['exe'],dlls=built['dlls'],source_root=str(ROOT),sources=sources,
        source_digest=digest(sources),build_evidence=[record(p) for p in evidence],objects=built['objects'],
        source_object_map=mapping,compilers=compiler_records(build),capabilities=dict(resolved_config=True,actual_velocity=True),
        runtime_input_additions=restore,
        scope='Clean native build bytes unchanged; four exact frozen-baseline runtime caches added afterward, separately audited. Original pre-build snapshot retained.')


def main(session,build,stage):
    require(sys.platform=='win32','Windows-only diagnostic entry')
    session=session.resolve();require(session.is_relative_to(ROOT/'runs'),'Workspace-local session required')
    session.mkdir(parents=True,exist_ok=True);require(not (session/(stage+'_ledger.json')).exists(),'No implicit retry')
    active=local_program_identity(build)
    base=read(ROOT/'reports/report_execution_20261007/FROZEN_REFERENCE.json')['programs']['base']
    verify(base['sources']+[base['exe']]+base['dlls'])
    identities=dict(active=active,base=base)
    identity_file=session/'local_programs.json'
    if not identity_file.exists():write(identity_file,dict(programs=identities,controller=record(__file__)))
    else:require(read(identity_file)['programs']==identities,'Build identity changed between local stages')
    stages=('seed','matched') if stage=='checkpoint' else ('smoke',)
    original_env=windows_runner.environment;rows=[]
    with windows_runner.gpu_lock(ROOT.parent/'StiffGIPC_Learning'):
        for part in stages:
            require(not (session/(part+'_ledger.json')).exists(),'Local stage exists')
            tasks=plan()['stages'][part];current=[];skipped=[];failed=set()
            for t in tasks:
                if t['scene_key'] in failed:skipped.append(t);continue
                if part=='matched':
                    cp=session/t['checkpoint_from']
                    if not (cp/'checkpoint.json').exists():skipped.append(t);failed.add(t['scene_key']);continue
                    checkpoint_evidence(cp)
                identity=identities[t['binary']]
                def env(config,out,t=t,identity=identity):
                    return env_for(t,out,identity['exe']['sha256'],session)
                windows_runner.environment=env
                # baseline execute constructs its own environment, with identical controls.
                try:result=windows_runner.execute(session,t,identity,0)
                finally:windows_runner.environment=original_env
                if result['status']=='completed' and part=='matched':
                    out=session/t['name'];cp=session/t['checkpoint_from'];restore=read(out/'output/checkpoint_restore.json')
                    require(restore['device_bytes_verified'] and restore['context_verified'],'Native restore failed')
                    for name,trace in (('vertices','state_0000.bin'),('velocities','velocity_0000.bin')):
                        require((cp/(name+'.bin')).read_bytes()==(out/'trace'/trace).read_bytes(),'Initial restore differs')
                row=dict(task=t,result=result);current.append(row);rows.append(row)
                if result['status']!='completed':failed.add(t['scene_key'])
            packet=dict(stage=part,runs=current,skipped=skipped,plan=plan(),
                complete=not skipped and len(current)==len(tasks) and all(r['result']['status']=='completed' for r in current))
            write(session/(part+'_ledger.json'),packet)
            if not packet['complete'] and part=='seed':break
    verify(active['sources']+[active['exe']]+active['dlls'])
    print(json.dumps(dict(stage=stage,complete=all(r['result']['status']=='completed' for r in rows),runs=len(rows))))
    return 0 if rows and all(r['result']['status']=='completed' for r in rows) else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True);parser.add_argument('--build',type=Path,required=True)
    parser.add_argument('--stage',choices=('smoke','checkpoint'),required=True)
    args=parser.parse_args();raise SystemExit(main(args.session,args.build.resolve(),args.stage))
