"""Independent M0 completion/identity/count/phase/origin/compile-options audit."""
import csv
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def require(value,message):
    if not value:raise ValueError(message)
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def release_cuda(project,source_root):
    ns={'m':'http://schemas.microsoft.com/developer/msbuild/2003'}
    tree=ET.parse(project)
    group=next(n for n in tree.findall('m:ItemDefinitionGroup',ns) if 'Release|x64' in n.attrib.get('Condition',''))
    node=group.find('m:CudaCompile',ns)
    defines=node.find('m:Defines',ns).text.split(';')
    defines=[d for d in defines if not d.startswith(('GIPC_ASSETS_DIR=','GIPC_OUTPUT_DIR='))]
    settings={n.tag.split('}')[-1]:n.text for n in node if n.tag.split('}')[-1]!='Defines'}
    # Each implementation necessarily includes its own verified native tree.
    # Normalize only these two exact roots; external dependency paths must match.
    if 'Include' in settings:
        for name in ('StiffGIPC','MeshProcess'):
            settings['Include']=settings['Include'].replace(str(source_root/name),'<native_source>/'+name)
    return dict(defines=defines,settings=settings)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root',type=Path,default=ROOT/'reports/innovation_m0_20261014/analysis_v2')
    args=parser.parse_args();report=args.analysis_root.resolve()
    summary=read(report/'summary.json')
    sessions=[ROOT/'runs/innovation_m0_20261014',ROOT/'runs/innovation_m0_boxes_hard_20261014']
    packet=read(sessions[0]/'seal.json')
    for identity in packet['programs'].values():
        rows=identity['sources']+[identity['exe']]+identity['dlls']+identity['build_evidence']+identity['objects']
        for row in rows:
            p=Path(row['path']);require(p.stat().st_size==row['bytes'] and sha(p)==row['sha256'],'Build/source identity changed: '+str(p))
    active=ROOT/'build/innovation_m0_b1_20261014_v2/gipc.vcxproj'
    base=ROOT/'build/innovation_m0_b0_20261014/frozen_baseline/gipc.vcxproj'
    options=[release_cuda(active,ROOT),release_cuda(base,ROOT/'baseline')]
    require(options[0]==options[1],'Release CUDA flags/macros differ beyond asset/output paths')
    require('USE_FRICTION' in options[0]['defines'] and 'USE_SNK1' in options[0]['defines'],'Wrong material/friction build')
    count=frames=restores=0;power_samples=0
    for session in sessions:
        for stage in ('smoke','prefix','matched','reference'):
            file=session/(stage+'_ledger.json')
            if not file.exists():continue
            ledger=read(file);require(ledger['complete'],'Incomplete stage')
            for row in ledger.get('runs',[]):
                folder=session/row['task']['name'];result=read(folder/'result.json')
                require(result['status']=='completed' and result['exit_code']==0 and result['cleanup_owned_job_empty'],'Failed native run')
                stat=read(folder/'output/stats.json')['frames']
                require(len(stat)==row['task']['config']['steps']==result['recorded_frames'],'Frame count mismatch')
                pcg=[n['pcg'] for f in stat for n in f['newton'] if 'pcg' in n]
                require(len(pcg)==row['observation']['N'] and sum(p['iterations'] for p in pcg)==row['observation']['P'],'N/P mismatch')
                for phase,total in row['observation']['phase_ms'].items():
                    if total is not None:
                        observed=sum(n['phase_ms'][phase] for f in stat for n in f['newton'])
                        require(math.isclose(observed,total,rel_tol=1e-12,abs_tol=1e-8),'Phase mismatch')
                power=read(folder/'power_samples.json');power_samples+=len(power['samples'])
                require(power['after']==packet['power_guid'] and all(s['guid']==packet['power_guid'] for s in power['samples']),'Observed power mismatch')
                if 'restore' in row:
                    require(row['restore']['passed'] and row['restore']['report']['context_verified'],'Restore gate failed');restores+=1
                count+=1;frames+=len(stat)
    fixed=ROOT/'runs/innovation_m0_fixed_20261014'
    require(read(fixed/'capture_gate.json')['passed'],'Fixed capture failed')
    result=read(fixed/'fixed_bunny_f14/result.json')
    require(result['status']=='completed' and result['cleanup_owned_job_empty'],'Fixed Job cleanup failed')
    require((count,frames,restores)==(60,256,40),'Ordinary matrix incomplete')
    require(summary['fixed_system']['cpu_native_residual_crosscheck_passed'],'No explicit residual crosscheck')
    require(summary['legacy_origin_metadata_mismatches']==30,'Missing original metadata diagnosis')
    require(read(sessions[1]/'origin_metadata_gate.json')['passed'],'Origin integration gate failed')
    subprocess.run(['git','diff','--exit-code','--','StiffGIPC','baseline','Assets','CMakeLists.txt','tests'],cwd=ROOT,check=True,capture_output=True)
    receipt=dict(passed=True,native_runs=count+1,native_frames=frames+1,matched_restores=restores,
        fixed_restore_verified=True,observed_power_samples=power_samples,
        native_source_and_frozen_baseline_unchanged=True,release_cuda_flags_and_material_macros_equal=True,
        input_problem_equality=True,explicit_residual_crosscheck=True,corrected_origin_integration=True,
        compile_settings=options[0],verification_command='E:/Anaconda/envs/DL/python.exe -X utf8 -B tools/bench/verify_m0.py --analysis-root reports/innovation_m0_20261014/analysis_v2',
        performance_certified=False,physical_quality_certified=False)
    output=report/'data_validation.json'
    require(not output.exists(),'Fresh verification receipt required')
    output.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='compile_settings'}))


if __name__=='__main__':main()
