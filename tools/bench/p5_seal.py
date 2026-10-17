"""Preserve this bounded GPU cost round and all corrections byte-for-byte."""
import ast
import json
from pathlib import Path
import re
import shutil
import subprocess

from m3_numeric import ROOT,need
from local_identity import record,verify,tree
from linux_runner import write_new

REPORT=ROOT/'reports/innovation_p5_gpu_20261015'
RAW=ROOT/'runs/innovation_p5_gpu_20261015'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def equivalent(recorded,path):
    actual=record(path);return (actual['bytes'],actual['sha256'])==(recorded['bytes'],recorded['sha256'])


def main():
    need(not (REPORT/'DELIVERY_SEAL.json').exists(),'Round already sealed')
    need(read(REPORT/'tables/validation.json')['passed'],'Tables incomplete')
    parent=read(ROOT/'reports/innovation_m3_20261015/DELIVERY_SEAL.json')
    todo=ROOT/'docs/research/INNOVATION_TODO_20261014.md';prior=REPORT/'prior_plan'/todo.name
    remapped=[]
    for r in parent['files']:
        if Path(r['path']).resolve()==todo.resolve():
            need(equivalent(r,prior),'Prior TODO differs from M3 seal');remapped.append(dict(original=r,archive=record(prior)))
        else:verify([r])
    source_paths=sorted((ROOT/'tools/bench').glob('p5_*.py'))+[ROOT/'tools/local/build_p5_probe.py',ROOT/'tools/local/run_p5_probe.py']
    source_paths+=sorted((ROOT/'tools/bench/p5_gpu').glob('*'))
    text_paths=source_paths+[REPORT/'REPORT.md',REPORT/'SOURCE_AUDIT.md',REPORT/'PLAN.md',todo]
    for p in text_paths:
        text=p.read_text(encoding='utf-8-sig')
        need(all(x.rstrip()==x for x in text.splitlines()),'Whitespace in '+str(p))
        if p.suffix=='.py':ast.parse(text,filename=str(p))
    missing=[]
    for p in (REPORT/'REPORT.md',REPORT/'SOURCE_AUDIT.md'):
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            if not target.startswith('https://') and not (p.parent/target).exists():missing.append(target)
    need(not missing,'Report references missing: '+str(missing))
    tests=ROOT/'build/p5_final_python_tests.log';content=tests.read_text(encoding='utf-8-sig')
    need('Ran 78 tests' in content and content.rstrip().endswith('OK'),'Final tests not passed')
    # Freeze current sources and bind every executed older runner/validator
    # revision to its retained archive instead of mislabeling it as current.
    mapping=[]
    for source in source_paths:
        destination=REPORT/'sources_v2'/source.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
        need(not destination.exists(),'Source snapshot exists');shutil.copy2(source,destination)
        need(equivalent(record(source),destination),'Source copy differs');mapping.append(dict(original=record(source),archive=record(destination)))
    archived=list((REPORT/'prior_runner_source').glob('*.py'))+[REPORT/'prior_validation_source/p5_validate.py']
    executed=[]
    for name in ('fixture_positive','fixture_deficient','mixed_cost','fixture_memcheck'):
        launch=read(RAW/name/'launch.json');need(launch['passed'] and launch['wall_seconds']<180,'Invalid Job')
        need(all(x['power_guid']==launch['power_guid'] for x in launch['samples']),'Power drift')
        candidates=source_paths+archived
        matches=[p for p in candidates if equivalent(launch['runner'],p)]
        need(matches,'Executed runner version not preserved');executed.append(dict(run=name,runner=launch['runner'],copy=record(matches[0])))
    build=ROOT/'build/p5_gpu_v2_20261015';identity=read(build/'identity.json')
    verify(identity['sources']+identity['evidence']+[identity['exe']])
    build_text=(build/'build.log').read_text(encoding='utf-8-sig',errors='replace')
    need('p5_pod_cost.cu' in build_text and 'compute_86' in build_text and '--use_fast_math' not in build_text,
         'Actual standalone CUDA build flags differ')
    diff=subprocess.run(['git','diff','--check','--','docs/research/INNOVATION_TODO_20261014.md'],cwd=ROOT,capture_output=True,timeout=20)
    need(diff.returncode==0,'New tracked document whitespace failure')
    capture=read(ROOT/'runs/innovation_m3_20261015/seal.json')
    verify(capture['identity']['sources']+capture['identity']['frozen_source']['original_sources']+[capture['identity']['exe']])
    # The current stdout log is still open. It changes after this script
    # prints its final result; including it would make the seal self-invalid.
    logs=[p for p in (ROOT/'build').glob('p5_*.log') if p.is_file() and 'seal' not in p.name]
    write_new(REPORT/'delivery_checks.json',dict(passed=True,Python_tests=78,memcheck_errors=0,
        text_syntax_whitespace_links_passed=True,parent_M3_files_verified_or_archived=len(parent['files']),
        production_sources_and_binary_unchanged=True,executed_runner_revisions_preserved=executed,
        parent_mutable_file_remapping=remapped,source=record(Path(__file__))))
    files=tree(RAW)+tree(REPORT)+[record(p) for p in source_paths+logs]+[record(todo)]+tree(build)
    # Retain the failed configure as evidence without relying on live files
    # outside this workspace (system compiler temporaries are not archived).
    files+=tree(ROOT/'build/p5_gpu_20261015')
    write_new(REPORT/'DELIVERY_SEAL.json',dict(schema='p5_gpu_cost_delivery.v1',files=files,
        source_snapshot=mapping,executed_runner_revisions=executed,parent_mutable_file_remapping=remapped,
        probe_exe=identity['exe'],production_exe=capture['identity']['exe'],Python_tests=78,
        GPU_jobs=4,expected_rejections=1,memcheck_errors=0,production_solver_modified=False,
        native_GPU_PCG_integrated=False,performance_certified=False,Github_uploaded=False,
        decision='Stop expanding direct 128-sample rebuild; setup exceeds the one-target conditional budget'))
    print(json.dumps(dict(sealed=True,files=len(files),sources=len(mapping),Python_tests=78,memcheck_errors=0)),flush=True)


if __name__=='__main__':main()
