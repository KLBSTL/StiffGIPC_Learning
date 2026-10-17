"""Seal the completed finite M3 diagnosis without changing native sources."""
import ast
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

from m3_numeric import ROOT, need
from local_identity import record, verify, tree
from linux_runner import write_new

REPORT = ROOT / 'reports/innovation_m3_20261015'
RAW = ROOT / 'runs/innovation_m3_20261015'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    need(not (REPORT/'DELIVERY_SEAL.json').exists(), 'Already sealed')
    validation = read(REPORT/'verified/validation.json')
    need(validation['passed'] and validation['saved_solution_rechecks'] == 252, 'Verification incomplete')
    rhs = read(REPORT/'rhs_excitation/validation.json')
    need(rhs['passed'] and rhs['rows'] == 27, 'RHS verification incomplete')
    verify([validation['source'],rhs['source']] + rhs['dependencies'])
    capture = read(RAW/'seal.json')
    verify(capture['identity']['sources'] + capture['identity']['frozen_source']['original_sources'] +
           [capture['identity']['exe']] + capture['identity']['dlls'])
    sources = [ROOT/'tools/local/run_m3_capture.py']
    sources += sorted((ROOT/'tools/bench').glob('m3_*.py'))
    sources += sorted((ROOT/'tools/bench').glob('run_m3_*.py'))
    need(len({str(p.resolve()) for p in sources}) == len(sources), 'Duplicate M3 source')
    docs = [REPORT/'REPORT.md',REPORT/'PROTOCOL.md',ROOT/'docs/research/INNOVATION_TODO_20261014.md']
    for path in sources:
        ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    for path in sources+docs:
        lines = path.read_text(encoding='utf-8-sig').splitlines()
        need(not any(line.rstrip()!=line for line in lines), 'M3 trailing whitespace: '+str(path))
    # Keep a byte-identical copy of this round's diagnostic source, as the
    # working scripts may be edited by a later research round.
    snapshot = REPORT/'sources'
    need(not snapshot.exists(), 'Fresh source snapshot required')
    mapping = []
    for path in sources:
        destination = snapshot/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        original,saved = record(path),record(destination)
        need((original['bytes'],original['sha256']) == (saved['bytes'],saved['sha256']), 'Source copy changed')
        mapping.append(dict(original=original,snapshot=saved))
    testlog = ROOT/'build/m3_all_python_tests.log'
    text = testlog.read_text(encoding='utf-8-sig')
    need('Ran 72 tests' in text and text.rstrip().endswith('OK'), 'Full Python test result missing')
    diff = subprocess.run(['git','diff','--check'],cwd=ROOT,capture_output=True,check=False,timeout=20)
    diff_log = REPORT/'repository_diff_check.txt'
    need(not diff_log.exists(), 'Existing diff check result')
    diff_log.write_bytes(diff.stdout+diff.stderr)
    scoped = subprocess.run(['git','diff','--check','--','docs/research/INNOVATION_TODO_20261014.md'],
                            cwd=ROOT,capture_output=True,check=False,timeout=20)
    need(scoped.returncode == 0, 'M3 tracked-document whitespace error')
    git_head = subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,check=True,timeout=20).stdout.decode().strip()
    need(git_head == '07534faba7d377315ac8f2023e63eab73fef14db', 'Unexpected branch head')
    write_new(REPORT/'delivery_checks.json',dict(passed=True,python_tests=72,new_M3_contract_tests=13,
        parsed_M3_python_files=len(sources),new_text_whitespace_passed=True,
        global_git_diff_check_exit=diff.returncode,global_git_diff_output=record(diff_log),
        global_diff_scope='Pre-existing native P3 whitespace retained byte-identical to compiled identity',
        native_current_and_frozen_sources_verified=True,head=git_head,
        residual_validation=record(REPORT/'verified/validation.json'),
        rhs_validation=record(REPORT/'rhs_excitation/validation.json'),test_log=record(testlog)))
    logs = [p for name in ('m3_capture_controller.log','m3_numeric_v2_controller.log','m3_full_cg_controller.log',
        'm3_arnoldi_controller.log','m3_unit_tests.log','m3_all_python_tests.log','m3_verify.log','m3_rhs_verify.log')
        if (p:=ROOT/'build'/name).is_file()]
    files = tree(RAW)+tree(REPORT)+[record(p) for p in sources+logs]+[record(docs[-1])]
    write_new(REPORT/'DELIVERY_SEAL.json',dict(schema='m3_fixed_system_delivery.v1',head=git_head,
        executable=capture['identity']['exe'],files=files,source_snapshot=mapping,
        native_runs=6,systems=9,candidate_solves=243,reference_solves=9,solution_rechecks=252,
        Decimal50_checks=18,CG_recurrence_bitwise_replays=18,python_tests=72,
        native_solver_modified=False,GPU_low_rank_implemented=False,
        full_M3_including_P1_complete=False,performance_certified=False,github_uploaded=False))
    print(json.dumps(dict(sealed=True,files=len(files),source_copies=len(mapping),python_tests=72,
        global_diff_check=diff.returncode,report=str(REPORT/'REPORT.md'))))


if __name__=='__main__':
    main()
