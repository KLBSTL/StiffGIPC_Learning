"""Final immutable evidence seal, with executed-version and parent TODO mapping."""
import ast
import json
from pathlib import Path
import re
import shutil
import subprocess

from m3_numeric import ROOT,need
from local_identity import record,verify,tree
from linux_runner import write_new

REPORT=ROOT/'reports/innovation_p5_cgstats_20261015'
RAW=ROOT/'runs/innovation_p5_cgstats_20261015'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def same(saved,path):
    r=record(path);return (r['bytes'],r['sha256'])==(saved['bytes'],saved['sha256'])


def main():
    need(not (REPORT/'DELIVERY_SEAL.json').exists(),'Already sealed')
    audit=read(REPORT/'AUDIT.json');need(audit['passed'] and audit['target_solutions_residual_rechecked']==12,'Audit incomplete')
    verify([audit['source']])
    parent_path=ROOT/'reports/innovation_p5_gpu_20261015/DELIVERY_SEAL.json';parent=read(parent_path)
    todo=ROOT/'docs/research/INNOVATION_TODO_20261014.md';prior=REPORT/'prior_plan'/todo.name;remapped=[]
    for r in parent['files']:
        if Path(r['path']).resolve()==todo.resolve():
            need(same(r,prior),'Parent TODO archive differs');remapped.append(dict(original=r,archive=record(prior)))
        else:verify([r])
    plan=read(RAW/'plan.json');executed=[]
    for r in plan['dependencies']:
        if same(r,Path(r['path'])):continue
        archive=REPORT/'executed_sources/run_cgstats_cpu_v1.py'
        need(Path(r['path']).name=='run_cgstats_cpu.py' and same(r,archive),'Executed CPU dependency is missing')
        executed.append(dict(original=r,archive=record(archive)))
    sources=[ROOT/'tools/bench'/name for name in ('cgstats_numeric.py','cgstats_numeric_test.py','run_cgstats_cpu.py',
        'prepare_cgstats_gpu.py','validate_cgstats_gpu.py','cgstats_audit_export.py','cgstats_seal.py')]
    sources += [ROOT/'tools/local/build_cgstats_probe.py',ROOT/'tools/local/run_cgstats_probe.py']
    sources += sorted((ROOT/'tools/bench/cgstats_gpu').glob('*'))
    docs=[REPORT/'REPORT.md',REPORT/'SOURCE_AUDIT.md',REPORT/'PLAN.md',todo]
    for p in sources+docs:
        text=p.read_text(encoding='utf-8-sig');need(all(x.rstrip()==x for x in text.splitlines()),'Trailing whitespace: '+str(p))
        if p.suffix=='.py':ast.parse(text,filename=str(p))
    for p in docs[:-1]:
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            need(link.startswith('https://') or (p.parent/link).exists(),'Missing report target: '+link)
    snapshot=[]
    for p in sources:
        dest=REPORT/'sources'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        need(not dest.exists(),'Snapshot exists');shutil.copy2(p,dest);need(same(record(p),dest),'Source snapshot changed')
        snapshot.append(dict(original=record(p),archive=record(dest)))
    inputs=[]
    for name in ('inputs','fixture_positive_inputs','fixture_negative_inputs'):
        m=read(RAW/name/'manifest.json');verify(m['records']);inputs.extend(m['records'])
    for name in ('fixture_positive','fixture_negative','mixed_cost','fixture_memcheck'):
        launch=read(RAW/name/'launch.json');need(launch['passed'] and launch['cleanup_owned_job_empty'],'GPU job incomplete')
        verify([launch['runner']]);verify(launch['identity']['sources']+launch['identity']['evidence']+[launch['identity']['exe']])
    build=ROOT/'build/cgstats_gpu_v2_20261015';identity=read(build/'identity.json')
    verify(identity['sources']+identity['evidence']+[identity['exe']])
    log=(build/'build.log').read_text(encoding='utf-8-sig',errors='replace')
    need('compute_86' in log and '--use_fast_math' not in log and '/utf-8' in log,'Actual compiler flags differ')
    tests=ROOT/'build/cgstats_all_python_tests.log';text=tests.read_text(encoding='utf-8-sig')
    need('Ran 84 tests' in text and text.rstrip().endswith('OK'),'Tests not passed')
    production=read(ROOT/'runs/innovation_m3_20261015/seal.json')['identity']
    verify(production['sources']+production['frozen_source']['original_sources']+[production['exe']])
    check=subprocess.run(['git','diff','--check','--','docs/research/INNOVATION_TODO_20261014.md'],cwd=ROOT,capture_output=True,timeout=20)
    need(check.returncode==0,'New tracked document whitespace failure')
    write_new(REPORT/'delivery_checks.json',dict(passed=True,Python_tests=84,memcheck_errors=0,
        new_source_syntax_whitespace_and_links_passed=True,parent_files_verified_or_remapped=len(parent['files']),
        executed_source_remapping=executed,parent_mutable_file_remapping=remapped,
        production_source_and_binary_unchanged=True,source=record(Path(__file__))))
    logs=[p for p in (ROOT/'build').glob('cgstats_*.log') if 'seal' not in p.name]
    records=tree(RAW)+tree(REPORT)+[record(p) for p in sources+logs]+[record(todo)]+inputs+tree(build)+tree(ROOT/'build/cgstats_gpu_20261015')
    # De-duplicate shared immutable input references, detecting any mid-seal change.
    unique={}
    for r in records:
        key=str(Path(r['path']).resolve()).casefold()
        need(key not in unique or unique[key]==r,'File changed during seal');unique[key]=r
    files=sorted(unique.values(),key=lambda r:r['path'].casefold())
    write_new(REPORT/'DELIVERY_SEAL.json',dict(schema='cgstats_delivery.v1',files=files,parent=record(parent_path),
        probe_exe=identity['exe'],production_exe=production['exe'],source_snapshot=snapshot,
        executed_source_remapping=executed,parent_mutable_file_remapping=remapped,
        CPU_source_histories=6,CPU_target_solutions=12,Decimal50_checks=6,GPU_jobs=4,expected_rejections=1,
        Python_tests=84,memcheck_errors=0,approximate_statistics_Gram=True,
        native_PCG_implemented=False,production_solver_modified=False,performance_certified=False,Github_uploaded=False))
    print(json.dumps(dict(sealed=True,files=len(files),sources=len(snapshot),Python_tests=84,memcheck_errors=0)),flush=True)


if __name__=='__main__':main()
