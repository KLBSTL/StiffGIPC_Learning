"""Seal only completed build/smoke evidence; representative gate remains pending."""
import ast
import json
from pathlib import Path
import re
import shutil
from m3_numeric import ROOT,need
from local_identity import record,verify,tree,freeze_active_identity
from linux_runner import write_new
from windows_runner import gpu_query

RAW=ROOT/'runs/innovation_p5_native_history_20261016'
REPORT=ROOT/'reports/innovation_p5_native_history_20261016'
BUILD=ROOT/'build/native_history_20261016'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    need(not (REPORT/'BUILD_AND_SMOKE_SEAL.json').exists(),'Stage already sealed')
    packet=read(RAW/'seal.json');identity=packet['identity'];verify(identity['sources']+[identity['exe']]+identity['dlls'])
    verify([packet['runner'],packet['protocol']])
    need(read(RAW/'small_ledger.json')['complete'],'Small native stage incomplete')
    smoke=read(REPORT/'SMOKE_AUDIT.json');need(smoke['passed'] and len(smoke['checks'])==3,'Smoke audit missing')
    verify(smoke['inputs']+[smoke['source']])
    verification=read(ROOT/'build/native_history_verification_v2/verification.json')
    need(verification['passed'] and verification['cleanup_owned_job_empty'],'CTest invalid')
    verify([verification['source']])
    tests=(ROOT/'build/native_history_python_tests.log').read_text(encoding='utf-8-sig')
    need('Ran 90 tests' in tests and tests.rstrip().endswith('OK'),'Python tests incomplete')
    for label in ('smoke','fault'):
        r=read(RAW/label/'history_validation.json');need(r['passed'] and r['result']['cleanup_owned_job_empty'],'Native Job invalid')
        verify(r['studies'])
    # Preserve full native build bytes for subsequent implementation rounds.
    identity_path=RAW/'compiled_identity.json';write_new(identity_path,identity)
    frozen=freeze_active_identity(ROOT,BUILD,BUILD/'Release/gipc.exe',BUILD/'build.log',
        ROOT/'build/native_history_snapshot_20261016',from_verified_identity=identity_path)
    parent_path=ROOT/'reports/innovation_p5_qrfree_20261016/DELIVERY_SEAL.json';parent=read(parent_path);mapped=[]
    for r in parent['files']:
        path=Path(r['path']);current=record(path)
        if current==r:continue
        archive=REPORT/'prior_sources'/path.relative_to(ROOT)
        need(archive.is_file(),'Unmapped parent mutation: '+str(path));saved=record(archive)
        need((saved['bytes'],saved['sha256'])==(r['bytes'],r['sha256']),'Parent source archive differs')
        mapped.append(dict(original=r,archive=saved))
    need(len(mapped)==5,'Unexpected parent source mutation count')
    preflight=gpu_query(0)
    write_new(REPORT/'REPRESENTATIVE_PENDING.json',dict(pending=True,current_GPU=preflight,
        recommended_free_mib=4608,reason='Preserve existing memory budget; previous representative peaks 2156/2328MiB plus bank',
        representative_launched=False,native_rank8_validated=False))
    sources=[ROOT/p for p in ['CMakeLists.txt','tests/native_history_sink_test.cpp',
        'StiffGIPC/linear_system/solver/pcg_solver.h','StiffGIPC/linear_system/solver/pcg_solver.cu',
        'StiffGIPC/linear_system/solver/pcg_fixed_study.inl','StiffGIPC/linear_system/solver/native_history_sink.h',
        'StiffGIPC/linear_system/solver/pcg_native_history_study.inl','StiffGIPC/linear_system/solver/pcg_native_history_runs.inl',
        'tools/local/run_native_history.py','tools/local/verify_native_history.py',
        'tools/bench/analyze_native_history.py','tools/bench/audit_native_history_smoke.py','tools/bench/seal_native_history_stage1.py']]
    for p in sources:
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'))
        dest=REPORT/'sources'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        need(record(p)['sha256']==record(dest)['sha256'],'Source copy differs')
    target=REPORT/'BUILD_AND_SMOKE_SEAL.json'
    for p in REPORT.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
            need(link.startswith('https://') or (p.parent/link).exists() or (p.parent/link)==target,'Missing report link')
    records=tree(RAW)+tree(REPORT)+tree(BUILD)+tree(ROOT/'build/native_history_snapshot_20261016')
    records+=tree(ROOT/'build/native_history_verification')+tree(ROOT/'build/native_history_verification_v2')
    records += [record(p) for p in sources]+[record(ROOT/'docs/research/INNOVATION_TODO_20261014.md'),record(parent_path)]
    records += [record(p) for p in (ROOT/'build').glob('native_history*.log') if 'stage1_seal' not in p.name]
    unique={r['path']:r for r in records};files=[unique[k] for k in sorted(unique)];verify(files)
    write_new(target,dict(schema='native_history_build_smoke.v1',files=files,parent=record(parent_path),
        parent_files_verified=len(parent['files']),parent_mutable_mapping=mapped,
        binary=frozen['exe'],CTest=27,Python_tests=90,native_success_runs=1,native_expected_failure_runs=1,
        independent_native_banks_checked=3,representative_validation_pending=True,native_rank8_validated=False,
        native_graph_history=False,production_threshold_changed=False,performance_certified=False))
    print(json.dumps(dict(sealed=True,files=len(files),parent_mappings=len(mapped),CTest=27,Python_tests=90,
        representative_validation_pending=True)),flush=True)


if __name__=='__main__':main()
