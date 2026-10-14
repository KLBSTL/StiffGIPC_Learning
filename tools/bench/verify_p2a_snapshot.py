"""Verify archived P2-A source against its pinned base, independently of HEAD."""
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
BASE='a4b83c040161eeef6149c93ae6167644fdf19384'
sys.path[:0]=[str(ROOT/'tools/local'),str(ROOT/'tools/bench')]
from config import read
from linux_runner import require
from local_identity import verify

def main():
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT).decode('utf-8').replace('\r\n','\n')
    require(not git('diff',BASE,'--name-only','--','baseline','StiffGIPC/fem',
        'StiffGIPC/linear_system','StiffGIPC/cuda_tools'),'Protected source differs from pinned base')
    for path in ('StiffGIPC/solver/line_search_acceptance.h','StiffGIPC/solver/ipc_options.h',
                 'StiffGIPC/solver/ipc_pcg_forcing.h'):
        require(git('show',BASE+':'+path)==(ROOT/path).read_text(encoding='utf-8'),
                'Acceptance/stopping/forcing source differs')
    path='StiffGIPC/abd_system/abd_system_function/cal_abd_energy.cu'
    before=git('show',BASE+':'+path);after=(ROOT/path).read_text(encoding='utf-8')
    require(before[before.index('__global__'):before.index('Float ABDSystem::cal_abd_kinetic_energy')]
        ==after[after.index('__global__'):after.index('void ABDSystem::enqueue_kinetic_energy')],
        'ABD mathematical kernels changed')
    path='StiffGIPC/core/GIPC.cu';pattern=r'bool GIPC::lineSearch\([\s\S]*?^}'
    before=re.search(pattern,git('show',BASE+':'+path),re.M).group()
    after=re.search(pattern,(ROOT/path).read_text(encoding='utf-8'),re.M).group()
    expected=before.replace('    wait_device();',
        '    { gipc::CostScope cost_wait("ipc.line_search.entry_wait",false); wait_device(); }')
    require(expected==after,'Line-search semantics differ beyond observational wait scope')
    report=ROOT/'reports/innovation_p2a_20261014'
    receipt=read(report/'analysis_v2/data_validation.json')
    require(receipt['passed'] and receipt['verified_runs']==79 and receipt['verified_frames']==384,
        'Final evidence receipt is invalid')
    # Local sealing is meaningful only at the original absolute artifact paths.
    delivery=read(report/'DELIVERY_SEAL.json')
    local=all(Path(row['path']).is_file() and Path(row['path']).is_relative_to(ROOT)
              for row in delivery['files'])
    if local:verify(delivery['files'])
    print(json.dumps(dict(passed=True,pinned_base=BASE,physics_source_checked=True,
        local_delivery_files_verified=local,source_check_only_on_other_checkouts=not local)))

if __name__=='__main__':main()
