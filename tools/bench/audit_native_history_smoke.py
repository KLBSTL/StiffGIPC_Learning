"""Real native six-step bank audit, independent of the recorder's index logic."""
import json
from pathlib import Path
import numpy as np
from m3_numeric import ROOT,FrozenSystem,need,rel
from cgstats_numeric import statistical_gram
from cpu_fixed_reference import decimal_residual
from local_identity import record,verify
from linux_runner import write_new


def main():
    raw=ROOT/'runs/innovation_p5_native_history_20261016';report=ROOT/'reports/innovation_p5_native_history_20261016'
    ledger=json.loads((raw/'small_ledger.json').read_text());need(ledger['complete'],'Small native runs incomplete')
    prefix=raw/'smoke/fixed/f1_n1';system=FrozenSystem(prefix)
    study=json.loads(Path(str(prefix)+'_native_history_study.json').read_text());n=len(system.full_b)
    checks=[];inputs=[];solutions=[]
    for row in study['runs']:
        x=np.fromfile(row['solution_file'],dtype='<f8');need(len(x)==n,'Native solution shape')
        metric=np.linalg.norm(system.full_b-system.full_A@x)/np.linalg.norm(system.full_b)
        need(abs(metric-row['true_relative_residual'])<1e-8,'Native SpMV versus full original CSR residual')
        solutions.append(x);inputs.append(record(row['solution_file']))
        if not row['capture']:continue
        steps=row['iterations'];width=row['history_width'];ids=np.fromfile(row['indices_file'],dtype='<i4')
        history=np.fromfile(row['history_file'],dtype='<f8').reshape(n,width,order='F')
        scalar=np.fromfile(row['scalars_file'],dtype='<f8').reshape(steps,3)
        need(steps<=128 and np.array_equal(ids,np.arange(steps)) and width==steps,'Small complete bank missing updates')
        need(np.array_equal(history[:,0],np.zeros(n)),'First saved iterate must be pre-update x0')
        H,_=statistical_gram(scalar[:,0],scalar[:,1],ids,curvature=scalar[:,2])
        need(np.all(scalar[:-1,1]>1e-4*scalar[0,1]) and scalar[-1,1]<=1e-4*scalar[0,1],
             'Native previous-rho stop changed')
        delta=np.diff(np.column_stack([history,x]),axis=1);energy=np.einsum('ij,ij->j',delta,system.full_A@delta)
        expected=scalar[:,0]*scalar[:,1];diagonal_error=rel(energy,expected)
        need(diagonal_error<1e-7,'Actual iterate increment energy does not match alpha/rho')
        S=x[:,None]-history;G=S.T@(system.full_A@S);G=.5*(G+G.T)
        _,dec=decimal_residual(system.full_A,system.full_b,x)
        need(abs(dec['relative_l2']-metric)<1e-9,'Independent Decimal50 native residual')
        checks.append(dict(pair=row['pair'],iterations=steps,history_width=width,
            native_relative_l2=row['true_relative_residual'],CPU_relative_l2=float(metric),Decimal50_relative_l2=dec['relative_l2'],
            increment_energy_relative=diagonal_error,statistics_Gram_relative=rel(H,G),rank8_available=width>=8))
        inputs += [record(row[k]) for k in ('indices_file','history_file','scalars_file')]
    differences=[rel(x,solutions[0]) for x in solutions];verify(inputs)
    write_new(report/'SMOKE_AUDIT.json',dict(passed=True,checks=checks,all_solution_relative_to_first=differences,
        inputs=inputs,source=record(Path(__file__)),native_execution='host',native_graph_history=False,
        representative_validation_pending=True,rank8_gate_tested=False,performance_certified=False))
    print(json.dumps(dict(passed=True,checks=checks,maximum_solution_difference=max(differences))),flush=True)


if __name__=='__main__':main()
