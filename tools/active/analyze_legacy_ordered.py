"""Independent CPU reference and bounded default-PCG cost gate; no GPU work."""
import itertools
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from config import ROOT, read, sha, matches_requested
from ipc_benchmark import write
from verify_systems import matrix_from_snapshot, accurate_reference, unpack

TAG='ipc_legacy_ordered_20261005'

def relative(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))

def load_operator(prefix,meta):
    def binary(suffix,dtype='<f8'):
        return np.fromfile(str(prefix)+suffix,dtype)
    locals=[v for v in meta['local_preconditioners'] if v['kind']=='MAS_full_owned_buffers']
    assert len(locals)==1
    local=locals[0]
    assert not any(local.get(k,False) for k in ('wide_apply','inverse64','cholesky'))
    nodes,mapped,levels,_,clusters,*_=local['dimensions'];offset=local['offset']*3
    part=binary('_mas_d_partId_map_real.bin','<i4')[:mapped]
    real=binary('_mas_d_real_map_partId.bin','<i4')[:nodes]
    coarse=binary('_mas_d_coarseTable.bin','<i4').reshape(-1,6)[:nodes,:levels-1]
    starts=binary('_mas_d_restriction_starts.bin','<i4')
    indices=binary('_mas_d_restriction_nodes.bin','<i4')
    assert len(starts)==clusters+1 and starts[0]==0 and starts[-1]==len(indices)
    assert np.all(starts[1:]>=starts[:-1]) and np.all((indices>=0)&(indices<nodes))
    valid=part>=0
    assert np.all(part[valid]<nodes) and np.all((real>=0)&(real<mapped))
    assert np.all((coarse>=mapped)&(coarse<clusters))
    assert np.all(np.bincount(part[valid],minlength=nodes)==1)
    assert np.array_equal(real[part[valid]],np.flatnonzero(valid))
    # Actual prolongation transpose, retaining every coarse multiplicity.
    rows=np.r_[real,coarse.ravel()]
    cols=np.r_[np.arange(nodes),np.repeat(np.arange(nodes),levels-1)]
    permutation=np.lexsort((cols,rows));expected_rows=rows[permutation];expected_cols=cols[permutation]
    expected_starts=np.r_[0,np.cumsum(np.bincount(expected_rows,minlength=clusters))]
    assert np.array_equal(starts,expected_starts) and np.array_equal(indices,expected_cols)
    restriction=sparse.csr_matrix((np.ones(len(indices)),indices,starts),shape=(clusters,nodes))
    bank=16;count=clusters//bank
    allocation=local['buffers']['d_precondMatMas']['count']
    inverse=unpack(Path(str(prefix)+'_mas_d_precondMatMas.bin'),'<f4',allocation,bank)[:count]
    assert np.isfinite(inverse).all()
    spectrum=np.linalg.eigvalsh((inverse+inverse.transpose(0,2,1))*.5)
    inverse_asymmetry=np.linalg.norm(inverse-inverse.transpose(0,2,1),axis=(1,2))/np.maximum(np.linalg.norm(inverse,axis=(1,2)),1e-300)
    abd=[]
    for item in meta['local_preconditioners']:
        if item['kind']=='abd':
            abd.append((3*item['offset'],binary('_'+item['buffer']+'.bin').reshape(-1,12,12).transpose(0,2,1)))
    diag=binary('_diag_inverse.bin').reshape(-1,3,3).transpose(0,2,1) if 'diag_inverse' in meta['buffers'] else None
    def apply(vector,rounded=False):
        output=np.einsum('nij,nj->ni',diag,vector.reshape(-1,3)).ravel() if diag is not None else np.zeros_like(vector)
        for at,mat in abd:
            length=len(mat)*12
            output[at:at+length]=np.einsum('nij,nj->ni',mat,vector[at:at+length].reshape(-1,12)).ravel()
        input_fem=vector[offset:offset+3*nodes].reshape(nodes,3)
        if rounded:input_fem=input_fem.astype(np.float32).astype(np.float64)
        restricted=restriction@input_fem
        if rounded:restricted=restricted.astype(np.float32).astype(np.float64)
        local_z=np.einsum('nij,nj->ni',inverse,restricted.reshape(count,bank*3)).reshape(clusters,3)
        output[offset:offset+3*nodes]=(restriction.T@local_z).ravel()
        return output
    return apply,{'nodes':nodes,'mapped_nodes':mapped,'clusters':clusters,'levels':levels,'scalar_offset':offset,
        'restriction_contributors':len(indices),'transpose_multiset_exact':True,
        'legacy_inverse_min_eigenvalue':float(spectrum.min()),
        'legacy_inverse_nonpositive_blocks':int(np.count_nonzero(spectrum[:,0]<=0)),
        'legacy_inverse_max_relative_asymmetry':float(inverse_asymmetry.max()),
        'spectrum_scope':'Recorded FP32 local inverse in FP64; not a proof of full GPU M SPD'},(part,coarse,restriction)

def main():
    folder=ROOT/'reports/active';plan=read(ROOT/f'configs/active/{TAG}.json')
    protocol=read(folder/f'{TAG}_protocol.json');gates=protocol['numerical_gates']
    systems=[];run_checks=[];stage=None;unavailable=[]
    for spec in plan['runs']:
        run=ROOT/'runs/active'/spec['name'];requested=read(run/'requested.json');result=read(run/'result.json')
        config=requested['expanded_config'];resolved=read(run/'resolved_config.json')
        assert matches_requested(config,spec['config'])
        if result['status']!='completed':
            unavailable.append({'run':run.name,'status':result['status'],'recorded_frames':result.get('recorded_frames'),
                                'required_frames':config['steps'],'memory_budget_mib':requested['memory_budget_mib'],
                                'scope':'No selected system data; no retry or reserve relaxation'})
            continue
        assert result['finite']
        assert result['recorded_frames']==config['steps']
        assert resolved['legacy_restrict']==config['legacy_restrict']
        assert resolved['fixed_legacy_restrict_study']==config['fixed_legacy_restrict_study']
        assert resolved['ipc_stopping']['termination']=='legacy' and resolved['pcg_rho_tol']==1e-4
        assert sha(Path(requested['command'][0]))==requested['exe_sha256']
        pcg=[n['pcg'] for f in read(run/'output/stats.json')['frames'] for n in f['newton'] if 'pcg' in n]
        assert not any(p.get('iteration_limit') or p.get('breakdown') for p in pcg)
        run_checks.append({'run':run.name,'completed':True,'exe_sha256':requested['exe_sha256'],
                           'production_iterations':[p['iterations'] for p in pcg]})
        if not config['diagnostics']:continue
        prefix=run/f"fixed/f{config['steps']}_n1";study=read(Path(str(prefix)+'_study.json'))
        if config['fixed_mas_stage_study']:
            probe=study['mas_stage_probe'];assert probe['stage_replays']==16
            restoration={k:study[k] for k in ('system_unchanged','full_snapshot_unchanged_including_scratch','primary_restored_bitwise','z_restored_bitwise','graph_signature_restored')}
            assert all(restoration.values())
            local=next(l for l in study['system']['local_preconditioners'] if l['kind']=='MAS_full_owned_buffers')
            nodes,mapped,levels,_,clusters,*_=local['dimensions']
            part=np.fromfile(str(prefix)+'_mas_d_partId_map_real.bin','<i4')[:mapped]
            coarse=np.fromfile(str(prefix)+'_mas_d_coarseTable.bin','<i4').reshape(-1,6)[:nodes,:levels-1]
            vector=np.fromfile(probe['input_r_file'],'<f8').reshape(nodes,3).astype(np.float32)
            reference=np.zeros((clusters,3));valid=part>=0;reference[np.flatnonzero(valid)]=vector[part[valid]]
            for level in range(levels-1):np.add.at(reference,coarse[:,level],vector.astype(np.float64))
            reference=reference.astype(np.float32)
            stage_records=[]
            for item in probe['stages']:
                dtype='<f4' if item['output_dtype']=='float32' else '<f8'
                outputs=[np.fromfile(rec['file'],dtype) for rec in item['outputs']]
                same=all(np.array_equal(outputs[0].view(np.uint8),v.view(np.uint8)) for v in outputs[1:])
                assert same and all(np.isfinite(v).all() for v in outputs)
                record={'stage':item['stage'],'four_outputs_bitwise_equal':same}
                if item['stage']=='restrict':
                    output=outputs[0].reshape(clusters,3)
                    record['fine_cpu_bitwise_equal']=np.array_equal(output[:mapped].view(np.uint8),reference[:mapped].view(np.uint8))
                    record['coarse_cpu_relative_l2']=relative(output[mapped:],reference[mapped:])
                    assert record['fine_cpu_bitwise_equal'] and record['coarse_cpu_relative_l2']<=gates['ordered_coarse_cpu_relative_l2_max']
                stage_records.append(record)
            stage={'restoration':restoration,'records':stage_records};continue
        assert config['fixed_legacy_restrict_study'] and 'diagnostic_error' not in study
        restoration={k:study[k] for k in ('system_unchanged','primary_restored_bitwise','z_restored_bitwise','graph_signature_restored')}
        assert all(restoration.values()) and study['rho_tolerance']==1e-4
        meta,a,b=matrix_from_snapshot(prefix);_,cpu=accurate_reference(a,b);assert cpu['passed']
        apply,mapping,_=load_operator(prefix,meta)
        probes=[];first={}
        for item in study['preconditioner_probes']:
            vector=np.fromfile(item['input_file'],'<f8');output=np.fromfile(item['output_file'],'<f8')
            assert np.isfinite(output).all()
            reference=apply(vector);error=relative(output,reference)
            assert error<=gates['full_M_cpu_relative_l2_max']
            quad=float(vector@output)
            assert (quad>0 if item['probe']!=1 else not np.count_nonzero(output))
            key=(item['arm'],item['probe'])
            if item['repeat']==1:first[key]=(vector,output)
            equal=np.array_equal(output.view(np.uint8),first[key][1].view(np.uint8))
            if item['arm']=='ordered':assert equal
            probes.append({'arm':item['arm'],'probe':item['probe'],'repeat':item['repeat'],
                'cpu_exact_inverse_relative_l2':error,'cpu_rounded_input_relative_l2':relative(output,apply(vector,True)),
                'quadratic':quad,'bitwise_equal_to_first':equal})
        symmetry=[]
        for arm in ('atomic','ordered'):
            for u,v in itertools.combinations((0,2,3),2):
                x,mx=first[arm,u];y,my=first[arm,v]
                asymmetry=float(abs(x@my-y@mx)/max(np.sqrt((x@mx)*(y@my)),1e-300))
                symmetry.append({'arm':arm,'pair':[u,v],'scaled_bilinear_asymmetry':asymmetry})
                assert asymmetry<=gates['bilinear_symmetry_relative_max']
        pairs=study['legacy_restriction_pairs'];assert len(pairs)==12 and len(study['warmups'])==4
        cost=[];work=[]
        for mode in ('host','graph'):
            atomic=[p for p in pairs if p['mode']==mode and p['arm']=='atomic']
            ordered=[p for p in pairs if p['mode']==mode and p['arm']=='ordered']
            maximum_iter=max(p['iterations'] for p in atomic);maximum_residual=max(p['true_relative_residual'] for p in atomic)
            for item in ordered:
                assert item['iterations']<=maximum_iter+max(2,int(np.ceil(.1*maximum_iter)))
                assert item['true_relative_residual']<=1.05*maximum_residual+1e-12
            for item in atomic+ordered:
                solution=np.fromfile(item['solution_file'],'<f8')
                residual=float(np.linalg.norm(b-a@solution)/max(np.linalg.norm(b),1e-300))
                assert np.isclose(residual,item['true_relative_residual'],rtol=1e-10,atol=1e-12)
                assert item['finite'] and item['rho_initial']>0 and item['rho_stop']>0
                if mode=='graph':assert item['graph_cache_hit']
            ratios=[next(p for p in atomic if p['pair']==q['pair'])['optimistic_full_linear_ms']/q['optimistic_full_linear_ms'] for q in ordered]
            solve_ratios=[next(p for p in atomic if p['pair']==q['pair'])['solve_ms']/q['solve_ms'] for q in ordered]
            speedup=float(np.median(ratios));reduction=1-1/speedup
            cost.append({'mode':mode,'paired_optimistic_linear_speedups':ratios,'paired_median_speedup':speedup,
                'median_cost_reduction':reduction,'solver_only_paired_median_speedup':float(np.median(solve_ratios)),
                'passes_15_percent_optimistic_cost_gate':reduction>=.15,
                'atomic_median_solve_ms':float(np.median([p['solve_ms'] for p in atomic])),
                'ordered_median_solve_ms':float(np.median([p['solve_ms'] for p in ordered]))})
            work.append({'mode':mode,'atomic_iterations':[p['iterations'] for p in atomic],
                'ordered_iterations':[p['iterations'] for p in ordered],
                'atomic_true_residual_range':[min(p['true_relative_residual'] for p in atomic),maximum_residual],
                'ordered_true_residual_range':[min(p['true_relative_residual'] for p in ordered),max(p['true_relative_residual'] for p in ordered)]})
        systems.append({'run':run.name,'system':mapping,'cpu_linear_reference':cpu,'restoration':restoration,
            'probes':probes,'symmetry':symmetry,'work':work,'cost':cost,
            'prepare_with_map_ms':study['prepare_with_map_ms'],'map_prepare_ms':study['map_prepare_ms'],
            'atomic_prepare_estimate_ms':study['atomic_prepare_estimate_ms'],'graph_warmups':study['warmups']})
    assert len(systems)+len(unavailable)==4 and stage is not None
    assert not (ROOT/'runs/active/.gpu.lock').exists()
    passes=not unavailable and all(c['passes_15_percent_optimistic_cost_gate'] for s in systems for c in s['cost'])
    report={'completed_system_numerical_checks_passed':True,'full_fixed_system_coverage_passed':not unavailable,
        'unavailable_systems':unavailable,'stage_checks':stage,'run_checks':run_checks,
        'systems':systems,'cost_gate_passed':passes,'promotion_allowed':False,'default_changed':False,
        'quality_certified':False,'old_quality_protocol_unchanged':True,'production_tolerances_changed':False,
        'decision':'Needs actual full-stage and scene validation' if passes else 'Seal execution candidate: no evidence meeting 15% complete-linear benefit; no long tests',
        'scope':'Completed-system diagnostic, not proof of full M SPD or physical trajectory quality. Cost subtracts map work from common synchronized preparation; omits distribution and cold capture. This is a diagnostic estimate, not a mathematical bound under different synchronization schedules.',
        'protocol_sha256':sha(folder/f'{TAG}_protocol.json'),'plan_sha256':sha(ROOT/f'configs/active/{TAG}.json')}
    write(folder/f'{TAG}_analysis.json',report)
    print(json.dumps({'completed_system_numerical_checks_passed':True,'unavailable_systems':unavailable,'cost_gate_passed':passes,
        'systems':[{'run':s['run'],'map_ms':s['map_prepare_ms'],'cost':s['cost']} for s in systems]}))

if __name__=='__main__':main()
