"""Explicit experiment configuration. No ambient GIPC variables are inherited."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = {
    'scene': 'bunny_cloth_bunny_l', 'steps': 35, 'dt': .01, 'timeout_seconds': 120,
    'backend': 'ipc', 'execution': 'host', 'mas': 'legacy', 'pcg_graph_chunk': 1,
    'ipc_newton_tol': .01, 'toi_remaining_fraction_tol': .01,
    'toi_trial_velocity_tol': .05, 'pcg_rho_tol': 1e-4,
    'refit': False, 'batch': False, 'reuse': False,
    'choose_start': False, 'restart_guard': False, 'mu_mode': 'diagonal',
    'mu_coordinates': 'auto',
    'diagnostics': [], 'cost_frames': '1-3,24-26,33-35',
    'fixed_frames': '2,25,34', 'fixed_directions': '1', 'trace_velocity': True,
    'profile': 'none', 'mas_restrict':'serial', 'fixed_restrict_study':False,
    'mas_factor_action':'auto', 'fixed_factor_study':False,
    'fixed_graph_chunk_study':False,
    'full_step_exit_probe':None,
    'ipc_termination':'legacy', 'ipc_cumulative_tol':None, 'ipc_min_updates':6,
    'ipc_residual_rel_tol':.03, 'ipc_residual_floor':1e-30,
    'ipc_residual_shadow':False, 'ipc_terminal_audit_frame':0,
    'ipc_residual_cpu_audit':False,
    'fixed_mas_stage_study':False,
    'legacy_restrict':'atomic', 'fixed_legacy_restrict_study':False,
    'cost_events':True,
    # Independent execution experiments. They never inherit the combined preset.
    'mas_fused_dot':False, 'fixed_mas_dot_study':False,
    'discrete_bvh_refit':False, 'discrete_bvh_rebuild_interval':8,
    'discrete_bvh_validate':False,
    'spmv_fused_quadratic':False, 'fixed_spmv_quadratic_study':False,
    'bvh_eligibility':False, 'bvh_eligibility_validate':False,
    'bounded_ccd':False, 'bounded_ccd_validate':False,
    'contact_pool':False, 'contact_pool_validate':False,
    'edge_query_order':'raw', 'edge_order_probe_frames':None,
}
PRESETS = {
    'base': {}, 'host': {}, 'graph': {'execution': 'conditional_graph'},
    'refit': {'refit': True}, 'batch': {'batch': True}, 'reuse': {'reuse': True},
    'combined': {'execution': 'conditional_graph', 'refit': True, 'batch': True, 'reuse': True},
    # A named report-compatible stopping experiment, using IPC contact/material
    # assembly. This is distinct from the AL/Robust `toi` backend below.
    'toi_compensated': {'backend':'ipc', 'execution':'conditional_graph',
                        'ipc_termination':'compensated', 'ipc_cumulative_tol':.001,
                        'ipc_residual_rel_tol':.03, 'ipc_min_updates':6},
    'cholesky': {'mas': 'cholesky'},
    'toi': {'backend': 'toi_al', 'execution': 'conditional_graph', 'refit': True,
            'batch': True, 'reuse': True, 'mas': 'cholesky', 'choose_start': True, 'restart_guard': True},
}

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def expand(data):
    data = dict(data)
    preset = data.pop('preset', 'host')
    if preset not in PRESETS:
        raise ValueError('Unknown preset: ' + preset)
    unknown = data.keys() - DEFAULT.keys()
    if unknown:
        raise ValueError('Unknown configuration keys: ' + str(unknown))
    c = DEFAULT | PRESETS[preset] | data
    chunk=c['pcg_graph_chunk']
    if type(chunk) is not int or chunk not in (1,4):
        raise ValueError('PCG Graph chunk must be the integer 1 or 4')
    if chunk==4 and c['execution']!='conditional_graph':
        raise ValueError('PCG Graph chunk 4 requires conditional_graph execution')
    if c['edge_query_order'] not in ('raw','leaf'):
        raise ValueError('Edge query order must be raw or leaf')
    probe=c['edge_order_probe_frames']
    if probe is not None:
        parts=probe.split(',') if isinstance(probe,str) else []
        if not parts or any(p not in ('1','41','57') for p in parts) or len(set(parts))!=len(parts):
            raise ValueError('Edge probe frames must be a unique subset of 1,41,57')
        if any(int(p)>c['steps'] for p in parts) or 'edge_order' not in c['diagnostics']:
            raise ValueError('Edge probe requires selected frames in budget and edge_order diagnostic')
    elif 'edge_order' in c['diagnostics']:
        raise ValueError('Edge order diagnostic requires explicit probe frames')
    if c['backend']!='ipc' and (c['edge_query_order']!='raw' or probe is not None):
        raise ValueError('Edge query order and probe are IPC-only')
    for key in ('mas_fused_dot','fixed_mas_dot_study','discrete_bvh_refit','discrete_bvh_validate',
                'spmv_fused_quadratic','fixed_spmv_quadratic_study','bvh_eligibility','bvh_eligibility_validate',
                'bounded_ccd','bounded_ccd_validate',
                'contact_pool','contact_pool_validate',
                'ipc_residual_shadow','ipc_residual_cpu_audit','fixed_graph_chunk_study'):
        if not isinstance(c[key],bool):raise ValueError(key+' requires a boolean')
    interval=c['discrete_bvh_rebuild_interval']
    if isinstance(interval,bool) or not isinstance(interval,int) or not 1<=interval<=1024:
        raise ValueError('Discrete BVH rebuild interval must be an integer in [1,1024]')
    if c['discrete_bvh_validate'] and not c['discrete_bvh_refit']:
        raise ValueError('Discrete BVH validation requires the refit candidate')
    if c['bvh_eligibility_validate'] and not c['bvh_eligibility']:
        raise ValueError('BVH eligibility validation requires the eligibility candidate')
    if c['bvh_eligibility'] and c['backend']!='ipc':
        raise ValueError('BVH eligibility experiments are scoped to IPC')
    if c['bounded_ccd_validate'] and not c['bounded_ccd']:
        raise ValueError('Bounded CCD validation requires the candidate')
    if c['bounded_ccd'] and c['backend']!='ipc':
        raise ValueError('Bounded CCD is scoped to the second IPC swept query')
    if c['contact_pool_validate'] and not c['contact_pool']:
        raise ValueError('Contact pool validation requires the candidate')
    if c['contact_pool'] and c['backend']!='ipc':
        raise ValueError('Contact pool is scoped to IPC line-search queries')
    if (c['mas_fused_dot'] or c['fixed_mas_dot_study'] or c['discrete_bvh_refit']) and c['backend']!='ipc':
        raise ValueError('New execution components are scoped to the IPC experiments')
    if (c['mas_fused_dot'] or c['fixed_mas_dot_study']) and c['mas']!='legacy':
        raise ValueError('MAS fused dot experiments require the legacy MAS path')
    if (c['spmv_fused_quadratic'] or c['fixed_spmv_quadratic_study']) and c['backend']!='ipc':
        raise ValueError('SpMV quadratic experiments are scoped to IPC')
    if preset=='toi_compensated' and (c['backend']!='ipc' or c['ipc_termination']!='compensated'):
        raise ValueError('toi_compensated requires IPC compensated stopping; use a different preset for other modes')
    if c['ipc_cumulative_tol'] is None:c['ipc_cumulative_tol']=c['ipc_newton_tol']
    if c['ipc_termination'] not in ('legacy','movement_only','gated','compensated'):
        raise ValueError('Invalid IPC termination')
    if isinstance(c['ipc_min_updates'],bool) or not isinstance(c['ipc_min_updates'],int) or not 1<=c['ipc_min_updates']<=10000:
        raise ValueError('Invalid IPC minimum updates')
    if not 0<c['ipc_cumulative_tol']<1 or not 0<c['ipc_residual_rel_tol']<float('inf') or not 0<c['ipc_residual_floor']<float('inf'):
        raise ValueError('Invalid IPC residual budget')
    if (c['ipc_residual_shadow'] or c['ipc_termination'] in ('gated','compensated')) and c['ipc_residual_rel_tol']<c['ipc_cumulative_tol']:
        raise ValueError('Residual tolerance must cover the cumulative tolerance')
    if c['ipc_residual_cpu_audit'] and not (c['ipc_residual_shadow'] or c['ipc_termination'] in ('gated','compensated')):
        raise ValueError('CPU residual audit requires residual observation')
    if isinstance(c['ipc_terminal_audit_frame'],bool) or not isinstance(c['ipc_terminal_audit_frame'],int) or c['ipc_terminal_audit_frame'] not in (0,c['steps']):
        raise ValueError('Terminal audit must be the final requested physical frame')
    if c['ipc_terminal_audit_frame'] and (not c['ipc_residual_shadow'] or c['ipc_termination']!='legacy'):
        raise ValueError('Terminal audit requires legacy stopping with shadow observations')
    if c['backend']!='ipc' and (c['ipc_termination']!='legacy' or c['ipc_residual_shadow'] or c['ipc_terminal_audit_frame']):
        raise ValueError('IPC stopping options cannot change AL TOI')
    if c['mas_factor_action']=='auto':
        c['mas_factor_action']='factor_inverse' if c['mas']=='cholesky' else 'triangular'
    if c['mu_coordinates']=='auto':
        c['mu_coordinates']='world_block' if c['backend']=='toi_al' and c['mu_mode']=='diagonal' else 'generalized'
    if not 1 <= c['steps'] <= 300 or not 1 <= c['timeout_seconds'] <= 600:
        raise ValueError('Finite frame/time budget required (<=300 frames, <=600 seconds)')
    if c['backend'] not in ('ipc', 'toi_al') or c['execution'] not in ('host', 'conditional_graph'):
        raise ValueError('Unsupported solver/backend')
    if c['mas'] not in ('legacy', 'cholesky') or c['mu_mode'] not in ('diagonal', 'mass'):
        raise ValueError('Unsupported MAS/contact mode')
    if c['mu_coordinates'] not in ('generalized','world_block'):
        raise ValueError('Unsupported penalty coordinates')
    if c['mu_coordinates']=='world_block' and (c['backend']!='toi_al' or c['mu_mode']!='diagonal'):
        raise ValueError('World block penalty requires TOI diagonal mode')
    if c['mas_restrict'] not in ('serial','warp') or (c['mas_restrict']=='warp' and c['mas']!='cholesky'):
        raise ValueError('Warp restriction requires stable Cholesky MAS')
    if c['mas_factor_action'] not in ('triangular','factor_inverse') or (c['mas_factor_action']=='factor_inverse' and c['mas']!='cholesky'):
        raise ValueError('Factor inverse action requires stable Cholesky MAS')
    if set(c['diagnostics']) - {'cost', 'operator_probe', 'fixed', 'state_window', 'audit', 'physics', 'substeps','outer_probe','edge_order','structure_probe','bvh_query_probe'}:
        raise ValueError('Unknown diagnostic')
    if not isinstance(c['cost_events'],bool) or (not c['cost_events'] and 'cost' not in c['diagnostics']):
        raise ValueError('CPU/NVTX-only cost mode requires cost diagnostics and a boolean flag')
    if c['full_step_exit_probe'] is not None:
        target=c['full_step_exit_probe']
        fields=target.split(':') if isinstance(target,str) else []
        if len(fields)!=2 or not all(f.isascii() and f.isdecimal() for f in fields):
            raise ValueError('Exit probe requires frame:outer')
        if not 1<=int(fields[0])<=c['steps'] or int(fields[1])<0 or c['backend']!='toi_al' or 'outer_probe' not in c['diagnostics']:
            raise ValueError('Exit probe requires an observed TOI subproblem in the frame budget')
    if 'operator_probe' in c['diagnostics'] and 'cost' not in c['diagnostics']:
        raise ValueError('operator_probe requires cost')
    if set(c['diagnostics']) & {'structure_probe','bvh_query_probe'}:
        if c['backend']!='ipc' or 'cost' not in c['diagnostics'] or c['profile']!='none':
            raise ValueError('Structure/query probes require standalone IPC cost diagnostics')
    if c['fixed_restrict_study'] and ('fixed' not in c['diagnostics'] or c['mas']!='cholesky'):
        raise ValueError('Restriction pair study requires fixed-system Cholesky diagnostics')
    if c['fixed_factor_study'] and ('fixed' not in c['diagnostics'] or c['mas']!='cholesky'):
        raise ValueError('Factor action study requires fixed-system Cholesky diagnostics')
    if c['fixed_factor_study'] and c['fixed_restrict_study']:
        raise ValueError('Study one MAS hypothesis at a time')
    if c['fixed_mas_stage_study'] and ('fixed' not in c['diagnostics'] or c['mas']!='legacy'
        or c['fixed_factor_study'] or c['fixed_restrict_study'] or c['backend']!='ipc'):
        raise ValueError('MAS stage probe requires fixed legacy IPC without other studies')
    if c['fixed_mas_stage_study'] and (set(c['diagnostics'])!={'fixed'}
        or c['fixed_frames']!=str(c['steps']) or c['fixed_directions']!='1'):
        raise ValueError('MAS stage probe requires exactly the final frame and first direction')
    if c['legacy_restrict'] not in ('atomic','ordered'):
        raise ValueError('Invalid legacy restriction')
    if (c['legacy_restrict']=='ordered' or c['fixed_legacy_restrict_study']) and (c['mas']!='legacy' or c['backend']!='ipc'):
        raise ValueError('Ordered legacy restriction is an IPC legacy MAS candidate')
    if c['fixed_legacy_restrict_study'] and (set(c['diagnostics'])!={'fixed'} or c['fixed_mas_stage_study']
        or c['fixed_frames']!=str(c['steps']) or c['fixed_directions']!='1' or c['legacy_restrict']!='atomic'):
        raise ValueError('Legacy pair study requires one final system and atomic production')
    if c['fixed_mas_dot_study'] and (set(c['diagnostics'])!={'fixed'} or c['fixed_mas_stage_study']
        or c['fixed_factor_study'] or c['fixed_restrict_study'] or c['fixed_legacy_restrict_study']
        or c['fixed_frames']!=str(c['steps']) or c['fixed_directions']!='1' or c['legacy_restrict']!='atomic'):
        raise ValueError('MAS dot study requires one final system and no other fixed study')
    if c['fixed_spmv_quadratic_study'] and (set(c['diagnostics'])!={'fixed'} or c['backend']!='ipc'
        or c['fixed_mas_dot_study'] or c['fixed_mas_stage_study'] or c['fixed_factor_study']
        or c['fixed_restrict_study'] or c['fixed_legacy_restrict_study']
        or c['fixed_frames']!=str(c['steps']) or c['fixed_directions']!='1'):
        raise ValueError('SpMV quadratic study requires one final system and no other fixed study')
    if c['fixed_graph_chunk_study']:
        other_studies=('fixed_restrict_study','fixed_factor_study','fixed_mas_stage_study',
                       'fixed_legacy_restrict_study','fixed_mas_dot_study','fixed_spmv_quadratic_study')
        if ('fixed' not in c['diagnostics'] or c['mas']!='legacy'
            or c['execution']!='conditional_graph' or c['pcg_graph_chunk']!=1
            or c['profile']!='none' or any(c[key] for key in other_studies)):
            raise ValueError('Graph chunk study requires fixed legacy MAS with conditional_graph K1 primary, no other study or profile')
    if c['profile'] not in ('none','graph','node') or (c['profile']!='none' and 'cost' not in c['diagnostics']):
        raise ValueError('Nsight graph/node profiling requires selected cost ranges')
    for key in ('dt', 'ipc_newton_tol', 'toi_remaining_fraction_tol', 'toi_trial_velocity_tol', 'pcg_rho_tol'):
        if not 0 < c[key] < float('inf'):
            raise ValueError('Invalid tolerance: ' + key)
    return c

def matches_requested(saved, requested):
    """Compare without rewriting/hash-normalizing historical request evidence.

    The two defaults changed on user request. Saved expanded values retain
    the historical effective settings when the original request omitted them;
    explicitly requested values and every other field must still match.
    """
    expected=expand(requested)
    if 'full_step_exit_probe' not in saved and expected['full_step_exit_probe'] is None:
        expected.pop('full_step_exit_probe')
    if 'fixed_mas_stage_study' not in saved and not expected['fixed_mas_stage_study']:
        expected.pop('fixed_mas_stage_study')
    for key,default in [('legacy_restrict','atomic'),('fixed_legacy_restrict_study',False)]:
        if key not in saved and expected[key]==default:expected.pop(key)
    if 'cost_events' not in saved and expected['cost_events']:expected.pop('cost_events')
    for key in ('mas_fused_dot','fixed_mas_dot_study','discrete_bvh_refit','discrete_bvh_rebuild_interval','discrete_bvh_validate',
                'spmv_fused_quadratic','fixed_spmv_quadratic_study','bvh_eligibility','bvh_eligibility_validate',
                'bounded_ccd','bounded_ccd_validate','contact_pool','contact_pool_validate'):
        if key not in saved and expected[key]==DEFAULT[key]:expected.pop(key)
    for key in ('ipc_termination','ipc_cumulative_tol','ipc_min_updates','ipc_residual_rel_tol','ipc_residual_floor','ipc_residual_shadow','ipc_terminal_audit_frame','ipc_residual_cpu_audit'):
        if key not in saved and key not in requested:expected.pop(key,None)
    for key in ('mas_factor_action','mu_coordinates'):
        if key not in requested or requested[key]=='auto':
            if key in saved:
                allowed=('triangular','factor_inverse') if key=='mas_factor_action' else ('generalized','world_block')
                if saved[key] not in allowed:return False
                expected[key]=saved[key]
            elif key=='mu_coordinates':expected.pop(key)
    if 'mu_coordinates' not in saved and requested.get('mu_coordinates')=='generalized':
        expected.pop('mu_coordinates')
    for key in ('edge_query_order','edge_order_probe_frames','pcg_graph_chunk','fixed_graph_chunk_study'):
        if key not in saved and expected[key]==DEFAULT[key]:expected.pop(key)
    return saved==expected

def environment(c, out):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIPC_')}
    env.update({
        'GIPC_SCENE': c['scene'], 'GIPC_STEPS': str(c['steps']), 'GIPC_DT': str(c['dt']),
        'GIPC_LEGACY_RESTRICT':c['legacy_restrict'],
        'GIPC_MAS_FUSED_DOT':str(int(c['mas_fused_dot'])),
        'GIPC_SPMV_FUSED_QUADRATIC':str(int(c['spmv_fused_quadratic'])),
        'GIPC_DISCRETE_BVH_REFIT':str(int(c['discrete_bvh_refit'])),
        'GIPC_DISCRETE_BVH_REBUILD_INTERVAL':str(c['discrete_bvh_rebuild_interval']),
        'GIPC_DISCRETE_BVH_VALIDATE':str(int(c['discrete_bvh_validate'])),
        'GIPC_BVH_ELIGIBILITY':str(int(c['bvh_eligibility'])),
        'GIPC_BVH_ELIGIBILITY_VALIDATE':str(int(c['bvh_eligibility_validate'])),
        'GIPC_BOUNDED_CCD':str(int(c['bounded_ccd'])),
        'GIPC_BOUNDED_CCD_VALIDATE':str(int(c['bounded_ccd_validate'])),
        'GIPC_CONTACT_POOL':str(int(c['contact_pool'])),
        'GIPC_CONTACT_POOL_VALIDATE':str(int(c['contact_pool_validate'])),
        'GIPC_EDGE_QUERY_ORDER':c['edge_query_order'],
        'GIPC_NEWTON_TOL': str(c['ipc_newton_tol']), 'GIPC_PCG_TOL': str(c['pcg_rho_tol']),
        'GIPC_IPC_TERMINATION':c['ipc_termination'], 'GIPC_IPC_CUMULATIVE_TOL':str(c['ipc_cumulative_tol']),
        'GIPC_IPC_MIN_UPDATES':str(c['ipc_min_updates']), 'GIPC_IPC_RESIDUAL_REL_TOL':str(c['ipc_residual_rel_tol']),
        'GIPC_IPC_RESIDUAL_FLOOR':str(c['ipc_residual_floor']), 'GIPC_IPC_RESIDUAL_SHADOW':str(int(c['ipc_residual_shadow'])),
        'GIPC_IPC_TERMINAL_AUDIT_FRAME':str(c['ipc_terminal_audit_frame']),
        'GIPC_IPC_RESIDUAL_CPU_AUDIT':str(int(c['ipc_residual_cpu_audit'])),
        'GIPC_TOI_REMAINING_FRACTION_TOL': str(c['toi_remaining_fraction_tol']),
        'GIPC_TOI_ROBUST_VELOCITY_TOL': str(c['toi_trial_velocity_tol']),
        'GIPC_CONTACT_BACKEND': c['backend'], 'GIPC_PCG_EXECUTION': c['execution'],
        'GIPC_PCG_GRAPH_CHUNK': str(c['pcg_graph_chunk']),
        'GIPC_FIXED_GRAPH_CHUNK_STUDY': str(int(c['fixed_graph_chunk_study'])),
        'GIPC_PCG_PRECONDITIONER': 'mas', 'GIPC_MAS_CHOLESKY': str(int(c['mas'] == 'cholesky')),
        'GIPC_MAS_WIDE_APPLY': '0', 'GIPC_MAS_INVERSE64': '0', 'GIPC_PCG_FUSED_DIAG_UPDATE': '0',
        'GIPC_MAS_RESTRICT_MODE':c['mas_restrict'],
        'GIPC_MAS_FACTOR_ACTION':c['mas_factor_action'],
        'GIPC_ACCEL_SUITE': '0', 'GIPC_CCD_PAIR_LIMIT': '10000000',
        'GIPC_TOI_POLICY': 'robust', 'GIPC_TOI_INNER_EXIT': 'native',
        'GIPC_TOI_CHOOSE_START': str(int(c['choose_start'])),
        'GIPC_TOI_RESTART_FULL_STEP_GUARD': str(int(c['restart_guard'])),
        'GIPC_TOI_MU_MODE': c['mu_mode'], 'GIPC_TOI_MU_SCALE': '1',
        'GIPC_TOI_MU_COORDINATES':c['mu_coordinates'],
        'GIPC_TOI_REDUCED_SLACK': '0', 'GIPC_TOI_SAFE_INJECTIVITY': 'auto',
        'GIPC_TOI_VOLUME_BOUND': 'normalized', 'GIPC_TOI_TRIAL_INJECTIVITY': '0',
        'GIPC_TOI_REUSE_INITIAL_ASSEMBLY': '1', 'GIPC_DEFER_STATS': '1',
        'GIPC_TRACE_STRIDE': '1', 'GIPC_TRACE_PHYSICS': str(int('physics' in c['diagnostics'])),
        'GIPC_TRACE_VELOCITY': str(int(c['trace_velocity'])),
        'GIPC_TRACE_DIR': str(out / 'trace'), 'GIPC_DUMP_STATE': str(out / 'final.bin'),
        'GIPC_OUTPUT_PATH': str(out / 'output'), 'GIPC_RESOLVED_CONFIG': str(out / 'resolved_config.json'),
        'GIPC_CCD_BVH_REFIT': str(int(c['refit'])), 'GIPC_BATCHED_ENERGY': str(int(c['batch'])),
        'GIPC_ENERGY_REUSE': str(int(c['reuse'])),
    })
    if c['edge_order_probe_frames'] is not None:
        env.update(GIPC_EDGE_ORDER_PROBE_FRAMES=c['edge_order_probe_frames'],
                   GIPC_EDGE_ORDER_PROBE_FILE=str(out/'edge_order_probe.jsonl'))
    if 'cost' in c['diagnostics']:
        env.update(GIPC_COST_TRACE=str(out / 'cost.jsonl'), GIPC_COST_FRAMES=c['cost_frames'],
                   GIPC_COST_OPERATOR_PROBE=str(int('operator_probe' in c['diagnostics'])),
                   GIPC_COST_EVENTS=str(int(c['cost_events'])))
    if 'structure_probe' in c['diagnostics']:
        env.update(GIPC_LINEAR_STRUCTURE_PROBE='1')
    if 'bvh_query_probe' in c['diagnostics']:
        frames=set()
        for item in c['cost_frames'].split(','):
            limits=list(map(int,item.split('-')))
            if len(limits)==1:limits.append(limits[0])
            if len(limits)!=2 or not 1<=limits[0]<=limits[1]<=c['steps']:
                raise ValueError('Query probe range exceeds frame budget')
            frames.update(range(limits[0],limits[1]+1))
        env.update(GIPC_BVH_QUERY_PROBE='1',GIPC_BVH_QUERY_PROBE_FRAMES=','.join(map(str,sorted(frames))),
                   GIPC_BVH_QUERY_PROBE_FILE=str(out/'bvh_query_probe.jsonl'))
    if 'outer_probe' in c['diagnostics']:
        env.update(GIPC_TOI_OUTER_PROBE=str(out/'outer_probe.jsonl'),GIPC_TOI_OUTER_PROBE_FRAMES=c['cost_frames'])
    if c['full_step_exit_probe'] is not None:
        env['GIPC_TOI_FULL_STEP_EXIT_PROBE']=c['full_step_exit_probe']
    if 'fixed' in c['diagnostics']:
        env.update(GIPC_FIXED_STUDY_DIR=str(out / 'fixed'), GIPC_FIXED_STUDY_FRAMES=c['fixed_frames'],
                   GIPC_FIXED_STUDY_DIRECTIONS=c['fixed_directions'], GIPC_FIXED_STUDY_COMPACT='1', GIPC_MAS_SNAPSHOT='1')
        env['GIPC_FIXED_RESTRICT_STUDY']=str(int(c['fixed_restrict_study']))
        env['GIPC_FIXED_FACTOR_STUDY']=str(int(c['fixed_factor_study']))
        env['GIPC_FIXED_MAS_STAGE_STUDY']=str(int(c['fixed_mas_stage_study']))
        env['GIPC_FIXED_LEGACY_RESTRICT_STUDY']=str(int(c['fixed_legacy_restrict_study']))
        env['GIPC_FIXED_MAS_DOT_STUDY']=str(int(c['fixed_mas_dot_study']))
        env['GIPC_FIXED_SPMV_QUADRATIC_STUDY']=str(int(c['fixed_spmv_quadratic_study']))
    if 'state_window' in c['diagnostics']:
        env.update(GIPC_STATE_WINDOW_DIR=str(out/'state_window'),GIPC_MAS_SNAPSHOT='1')
    if 'audit' in c['diagnostics']:
        env.update(GIPC_AUDIT_PCG='1', GIPC_AUDIT_REFIT='1', GIPC_AUDIT_ENERGY='1')
    if 'audit' in c['diagnostics'] or 'substeps' in c['diagnostics']:
        env['GIPC_TRACE_SUBSTEPS'] = str(out / 'trace/substeps')
    return env
