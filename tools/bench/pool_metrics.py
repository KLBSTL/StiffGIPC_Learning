"""Frozen CPU metrics extracted from the pre-cleanup benchmark analyzers.
No solver invocation, checkpoint replay or inferred velocity. Source: archive/pre-cleanup-20261006.
"""
import collections
import math
import statistics
import numpy as np
from config import read, sha

PHASES = ("assembly", "pcg", "ccd", "line_search", "state_update")
POOL_SUM_FIELDS = ("attempts", "reused_queries", "old_queries", "validation_calls",
                   "validation_passed", "validation_failed", "pairs_compared",
                   "nonempty_validation_calls", "production_overflow_queries")
POOL_DETAIL_FIELDS = ("prepare_calls", "capture_passes", "generations", "pool_pairs", "vf_pairs", "ee_pairs")

def require(value, message):
    if not value:
        raise ValueError(message)

def metrics(folder,frame_limit=None):
    req=read(folder/'requested.json');result=read(folder/'result.json');frames=read(folder/'output/stats.json')['frames']
    if frame_limit is not None:
        if not isinstance(frame_limit,int) or not 1<=frame_limit<=result['recorded_frames']:
            raise ValueError('Prefix metrics require completed exported frames')
        frames=frames[:frame_limit]
    raw=np.fromfile(folder/'trace/topology.bin',dtype='<u4');nv,nf,nt=map(int,raw[:3]);faces=raw[3:3+3*nf].reshape(-1,3);tets=raw[3+3*nf:].reshape(-1,4)
    x0=np.fromfile(folder/'trace/state_0000.bin',dtype='<f8').reshape(-1,3)
    abd=read(folder/'trace/metadata.json')['abd_point_num'];in_tet=np.zeros(nv,bool);in_tet[tets.ravel()]=True
    cloth_faces=faces[~in_tet[faces].any(axis=1)]
    edges=np.unique(np.sort(np.concatenate([cloth_faces[:,[0,1]],cloth_faces[:,[1,2]],cloth_faces[:,[2,0]]]),axis=1),axis=0)
    l0=np.linalg.norm(x0[edges[:,0]]-x0[edges[:,1]],axis=1)
    boundary=np.fromfile(folder/'trace/boundary_types.bin',dtype='<i4');mass=np.fromfile(folder/'trace/masses.bin',dtype='<f8')
    fixed=boundary!=0
    scene=read(folder/'output/scene.json');objs=scene['objects']
    if all(o.get('fixed_mode')=='all' for o in objs if o['dimension']==3 and o['body_type']=='ABD'):
        # Only mark ABD fixed if there actually are ABD vertices and all such objects are fixed.
        fixed[:abd]=True
    def volume(x):
        q=x[tets];return np.einsum('ij,ij->i',q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))/6
    v0=volume(x0);fm=np.all(tets>=abd,axis=1);am=np.all(tets<abd,axis=1)
    rows=[]
    for i in range(1,len(frames)+1):
        x=np.fromfile(folder/f'trace/state_{i:04d}.bin',dtype='<f8').reshape(-1,3)
        stretch=np.linalg.norm(x[edges[:,0]]-x[edges[:,1]],axis=1)/l0
        j=volume(x)/v0 if nt else np.array([])
        rows.append({'frame':i,'max_stretch':float(stretch.max()) if len(stretch) else 1,
            'p99_stretch':float(np.quantile(stretch,.99)) if len(stretch) else 1,
            'fixed_drift_m':float(np.linalg.norm(x[fixed]-x0[fixed],axis=1).max()) if fixed.any() else 0,
            'finite':bool(np.isfinite(x).all()),'fem_min_J':float(j[fm].min()) if fm.any() else None,
            'fem_nonpositive':int((j[fm]<=0).sum()),'fem_negative_volume':float(np.maximum(-j[fm],0).dot(abs(v0[fm]))) if fm.any() else 0,
            'abd_min_J':float(j[am].min()) if am.any() else None})
    pcg=[n['pcg'] for f in frames for n in f['newton'] if 'pcg' in n]
    observations=[n['ipc_residual'] for f in frames for n in f['newton'] if 'ipc_residual' in n]
    return {'name':folder.name,'scene':req['expanded_config']['scene'],'config':req['expanded_config'],
        'status':result['status'],'seconds':result.get('solver_seconds'),'frames':rows,
        'max_stretch':max(r['max_stretch'] for r in rows),'p99_stretch':max(r['p99_stretch'] for r in rows),
        'fixed_drift_m':max(r['fixed_drift_m'] for r in rows),'finite':all(r['finite'] for r in rows),
        'directions':len(pcg),'pcg':sum(p['iterations'] for p in pcg),
        'pcg_failures':sum(bool(p.get('iteration_limit') or p.get('breakdown')) for p in pcg),
        'exits':dict(collections.Counter(f.get('newton_exit') for f in frames)),
        'residual_observations':observations,'history_vetoes':sum(bool(r['history_veto']) for r in observations),
        'observe_host_ms':sum(r['observe_host_ms'] for r in observations),
        'terminal_assembly_host_ms':sum(f.get('ipc_terminal_assembly_host_ms',0) for f in frames),
        'exit_assembly_ms':sum(f.get('ipc_exit_assembly_ms',0) for f in frames),
        'phase_ms':{k:sum(f.get('phase_ms',{}).get(k,0) for f in frames) for k in ['assembly','pcg','ccd','line_search','state_update']},
        'requested_sha256':sha(folder/'requested.json'),'velocity_frames':len(list((folder/'trace').glob('velocity_*.bin')))}

def state_comparison(a, b):
    """Report missing exports explicitly; never infer actual velocity from x."""
    meta = read(a / 'trace/metadata.json')
    nv = np.fromfile(a / 'trace/state_0000.bin', dtype='<f8').size // 3
    raw = np.fromfile(a / 'trace/topology.bin', dtype='<u4')
    assert np.array_equal(raw, np.fromfile(b / 'trace/topology.bin', dtype='<u4'))
    nf, nt = map(int, raw[1:3])
    tets = raw[3+3*nf:].reshape(nt, 4)
    abd = meta['abd_point_num']
    fem = np.zeros(nv, bool)
    fem[tets.ravel()] = True
    fem[:abd] = False
    groups = {'cloth': ~fem & (np.arange(nv) >= abd), 'FEM': fem, 'ABD': np.arange(nv) < abd}
    result = {}
    for kind, pattern in (('position', 'state_*.bin'), ('velocity', 'velocity_*.bin')):
        files = sorted((a / 'trace').glob(pattern))
        other = sorted((b / 'trace').glob(pattern))
        if not files or not other:
            result[kind] = {'available': False, 'reason': 'Actual export absent in one or both programs',
                            'run_exported_frames': len(files), 'reference_exported_frames': len(other)}
            continue
        assert [f.name for f in files] == [f.name for f in other], 'Incomplete paired state exports'
        values = {k: [] for k, mask in groups.items() if mask.any()}
        for p, q in zip(files, other):
            x, y = np.fromfile(p, dtype='<f8'), np.fromfile(q, dtype='<f8')
            assert x.size == y.size == 3*nv and np.isfinite(x).all() and np.isfinite(y).all()
            delta = (x-y).reshape(nv, 3)
            for k in values:
                error = np.linalg.norm(delta[groups[k]], axis=1)
                values[k].append((float(np.sqrt(np.mean(error**2))), float(error.max())))
        result[kind] = {k: {'max_frame_rms': max(v[0] for v in rows), 'max_vertex': max(v[1] for v in rows),
                           'frames': len(rows)} for k, rows in values.items()}
    return result

def window(frames, csv_rows, low, high):
    require(high >= low and high <= len(frames), "Incomplete requested window")
    chosen = frames[low-1:high]
    times = {int(r["frame"]): float(r["solver_ms"]) for r in csv_rows}
    require(all(i in times for i in range(low, high+1)), "Missing frame timing")
    require(all(all(k in f.get("phase_ms", {}) for k in PHASES) for f in chosen), "Missing phase timing")
    phase = {k: sum(f["phase_ms"][k] for f in chosen) for k in PHASES}
    elapsed = sum(times[i] for i in range(low, high+1))
    terminal_available = all("ipc_exit_assembly_ms" in f for f in chosen)
    terminal = sum(f["ipc_exit_assembly_ms"] for f in chosen) if terminal_available else None
    pcgs = [n["pcg"] for f in chosen for n in f["newton"] if "pcg" in n]
    alphas = [n["alpha"] for f in chosen for n in f["newton"] if "alpha" in n]
    return {"frames": [low, high], "solver_ms": elapsed, "phase_ms": phase,
            "exit_assembly_ms": terminal,
            "unclassified_ms": elapsed-sum(phase.values())-(terminal or 0),
            "unclassified_includes_unreported_exit_assembly": not terminal_available,
            "directions": len(pcgs), "pcg_iterations": sum(p["iterations"] for p in pcgs),
            "pcg_iterations_per_direction": {"min": min((p["iterations"] for p in pcgs), default=None),
                "max": max((p["iterations"] for p in pcgs), default=None),
                "median": statistics.median(p["iterations"] for p in pcgs) if pcgs else None},
            "accepted_alpha_count": len(alphas), "alpha_sum": sum(alphas),
            "accepted_alphas_by_frame": [[n["alpha"] for n in f["newton"] if "alpha" in n] for f in chosen],
            "newton_exits": dict(collections.Counter(f.get("newton_exit") for f in chosen)),
            "native_candidate_active_frames": [low+i for i, f in enumerate(chosen)
                if f.get("contact_geometry", {}).get("native_narrow_self_pairs", 0)
                or f.get("contact_geometry", {}).get("native_narrow_ground_pairs", 0)],
            "newton_active_frames": [low+i for i, f in enumerate(chosen) if any(n.get("active_pairs", 0) > 0 for n in f["newton"])],
            "native_self_pair_peak": max((f.get("contact_geometry", {}).get("native_narrow_self_pairs", 0) for f in chosen), default=0),
            "native_ground_pair_peak": max((f.get("contact_geometry", {}).get("native_narrow_ground_pairs", 0) for f in chosen), default=0),
            "newton_active_pair_peak": max((n.get("active_pairs", 0) for f in chosen for n in f["newton"]), default=0),
            "physical_contact_certified": False}

def pool_evidence(frames, expected_on, validate_on):
    rows = [f.get("contact_pool") for f in frames]
    required = ("requested", "validate", *POOL_SUM_FIELDS, *POOL_DETAIL_FIELDS, "fallback_reasons", "pool_bytes_peak")
    missing = [{"frame": i+1, "fields": list(required) if not isinstance(row, dict)
                else [k for k in required if k not in row]} for i, row in enumerate(rows)
               if not isinstance(row, dict) or any(k not in row for k in required)]
    result = {"complete_contract": bool(rows) and not missing, "missing": missing, "passed": False,
              "same_state_energy_alpha_support": None}
    if missing or not rows:
        return result
    count_fields = (*POOL_SUM_FIELDS, *POOL_DETAIL_FIELDS, "pool_bytes_peak")
    require(all(type(r[k]) is int and r[k] >= 0 for r in rows for k in count_fields), "Invalid contact-pool counters")
    sums = {k: sum(r[k] for r in rows) for k in (*POOL_SUM_FIELDS, *POOL_DETAIL_FIELDS)}
    reasons = collections.Counter()
    for r in rows:
        reasons.update(r["fallback_reasons"])
    result.update(totals=sums, fallback_reasons=dict(reasons),
                  pool_bytes_peak=max((r["pool_bytes_peak"] for r in rows), default=0),
                  raw_frame_evidence=rows,
                  production_overflow_scope="Measured only by validation reference.count > caller capacity; when validate=false this is not the total production overflow count")
    flags = all(r["requested"] == expected_on and r["validate"] == validate_on for r in rows)
    valid = sums["validation_failed"] == 0
    if expected_on:
        valid &= sums["reused_queries"] > 0
    else:
        valid &= sums["reused_queries"] == 0
    if validate_on:
        valid &= (sums["validation_calls"] > 0 and sums["validation_calls"] == sums["validation_passed"]
                  and sums["pairs_compared"] > 0 and sums["nonempty_validation_calls"] > 0)
        valid &= sums["validation_calls"] == sums["reused_queries"]
        energies = [n for f in frames for entry in f.get("newton", [])
                    for n in entry.get("contact_pool_energy", [])]
        energy_frames = [r.get("energy_audit") for r in rows]
        energy_contract = all(isinstance(r, dict) and all(k in r for k in
            ("calls", "failed", "max_relative_error", "max_barrier_relative_error")) for r in energy_frames)
        energy_ok = bool(energy_contract and energies)
        if energy_contract:
            expected_energy = sums["validation_passed"]-sums["production_overflow_queries"]
            energy_ok &= 0 < expected_energy and sum(r["calls"] for r in energy_frames) == len(energies) == expected_energy
            energy_ok &= sum(r["failed"] for r in energy_frames) == 0
            energy_ok &= all(math.isfinite(r[k]) and 0 <= r[k] <= 1e-10 for r in energy_frames
                             for k in ("max_relative_error", "max_barrier_relative_error"))
            for frame, counts, audit in zip(frames, rows, energy_frames):
                frame_energy = [e for n in frame.get("newton", []) for e in n.get("contact_pool_energy", [])]
                energy_ok &= (audit["calls"] == len(frame_energy) == counts["validation_passed"]-counts["production_overflow_queries"])
        keys = ("alpha", "candidate_energy", "legacy_energy", "relative_error", "barrier_relative_error",
                "armijo_branch_equal", "production_restored", "passed")
        for e in energies:
            energy_ok &= all(k in e for k in keys)
            if all(k in e for k in keys):
                energy_ok &= all(isinstance(e[k], (int, float)) and math.isfinite(e[k]) for k in keys[:5])
                energy_ok &= e["passed"] and e["armijo_branch_equal"] and e["production_restored"]
                energy_ok &= abs(e["relative_error"]) <= 1e-10 and abs(e["barrier_relative_error"]) <= 1e-10
        result["same_state_energy_alpha_support"] = {"passed": bool(energy_ok), "records": energies,
            "frame_summaries": energy_frames, "overflow_queries_excluded": sums["production_overflow_queries"],
            "expected_energy_calls": sums["validation_passed"]-sums["production_overflow_queries"],
            "tolerance": 1e-10, "tolerance_is_physical_relaxation": False}
        valid &= energy_ok
    result["passed"] = bool(flags and valid)
    return result

def export_evidence(run, steps, kind):
    topology = np.fromfile(run / "trace/topology.bin", dtype="<u4")
    require(len(topology) >= 3, "Topology header missing")
    vertices = int(topology[0])
    expected = [f"{kind}_{i:04d}.bin" for i in range(steps+1)]
    actual = sorted(p.name for p in (run / "trace").glob(kind+"_*.bin"))
    problems = []
    for name in expected:
        path = run / "trace" / name
        if not path.exists():
            problems.append({"file": name, "reason": "missing"})
            continue
        if path.stat().st_size != vertices*24:
            problems.append({"file": name, "reason": "wrong_byte_count", "actual": path.stat().st_size, "expected": vertices*24})
        elif not np.isfinite(np.fromfile(path, dtype="<f8")).all():
            problems.append({"file": name, "reason": "nonfinite"})
    return {"available": bool(actual), "passed": actual == expected and not problems,
            "expected_frames": steps+1, "actual_frames": len(actual), "vertices": vertices,
            "expected_bytes_per_frame": vertices*24, "problems": problems}

def material_window(rows, low, high):
    selected = rows[low-1:high]
    def minimum(key):
        values = [r[key] for r in selected if r[key] is not None]
        return min(values) if values else None
    return {"frames": [low, high], "max_stretch": max(r["max_stretch"] for r in selected),
            "p99_stretch": max(r["p99_stretch"] for r in selected),
            "fixed_drift_m": max(r["fixed_drift_m"] for r in selected),
            "fem_min_J": minimum("fem_min_J"), "abd_min_J": minimum("abd_min_J"),
            "fem_nonpositive_sum": sum(r["fem_nonpositive"] for r in selected),
            "fem_negative_volume_peak": max(r["fem_negative_volume"] for r in selected)}

def initial_inputs(a, b):
    """The off/on execution flags may differ; physical inputs may not."""
    files = ("trace/topology.bin", "trace/masses.bin", "trace/boundary_types.bin", "trace/body_ids.bin",
             "trace/state_0000.bin", "trace/velocity_0000.bin", "trace/metadata.json", "output/scene.json")
    evidence = {}
    for name in files:
        pa, pb = a / name, b / name
        present = pa.exists() and pb.exists()
        left, right = (sha(pa), sha(pb)) if present else (None, None)
        evidence[name] = {"off_sha256": left, "on_sha256": right, "equal": present and left == right}
    return {"passed": all(r["equal"] for r in evidence.values()), "files": evidence}
