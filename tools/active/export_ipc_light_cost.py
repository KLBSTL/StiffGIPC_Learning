"""CPU-only export of the already completed finite Nsight captures."""
import json
import subprocess
import sys
from config import ROOT,read

NSYS='C:/Program Files/NVIDIA Corporation/Nsight Systems 2025.3.2/target-windows-x64/nsys.exe'

def command(argv,log):
    with log.open('xb') as output:
        output.write((json.dumps(argv)+'\n').encode())
        result=subprocess.run(argv,stdout=output,stderr=subprocess.STDOUT,timeout=120)
    if result.returncode:raise RuntimeError(f'Export failed ({result.returncode}): {log}')

def main():
    plan=read(ROOT/'configs/active/ipc_light_cost_20261005.json')
    for task in plan['runs']:
        if task['config'].get('profile')!='node':continue
        run=ROOT/'runs/active'/task['name'];result=read(run/'result.json')
        if result['status']!='completed':continue
        database=run/'nsight.sqlite';analysis=run/'activity_analysis.json'
        command([NSYS,'export','--type=sqlite','--output='+str(database),str(run/'nsight.nsys-rep')],run/'export_v2.log')
        command([sys.executable,str(ROOT/'tools/active/analyze_ipc_light_cost.py'),'--sqlite',str(database),'--output',str(analysis)],run/'analysis_v2.log')
        command([NSYS,'stats','--report','nvtx_gpu_proj_sum','--format','csv','--output='+str(run/'nsys'),str(database)],run/'stats_export.log')
        print(json.dumps({'run':run.name,'exported':True}),flush=True)

if __name__=='__main__':main()
