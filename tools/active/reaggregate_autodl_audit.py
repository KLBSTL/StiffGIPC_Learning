"""Recheck configuration and reuse an unchanged completed AutoDL CPU CCD audit.

Reads the original audit, native CCD JSON/stdout, and complete retained traces.
Never invokes a validator, simulator, subprocess or GPU. The old audit is not
modified. A fresh output file records every original failure and source identity.
"""
import argparse
import copy
import datetime
import json
from pathlib import Path

import analyze_autodl_factor_v2 as analyzer
import audit_autodl_factor as original_wrapper
from config import ROOT, read, sha


CONFIG_PREFIX = 'Configuration/execution: '
INVENTORY_FIELDS = ('coverage_passed', 'all_accepted_states_finite', 'accepted_states_checked',
                    'accepted_segments', 'bridge_count', 'stationary_bridges', 'expected_validator_paths',
                    'fixed_vertices', 'fixed_abd_vertices', 'max_fixed_drift_m',
                    'max_fixed_abd_drift_m', 'fixed_diagnostic_passed')
FINAL_FIELDS = ('all_54_accepted_path_audits_passed', 'actual_validator_reports',
                'actual_validator_paths_checked', 'actual_collision_flags_from_available_reports',
                'collision_flag_total_covers_all_54_full_windows')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def no_links(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        require(not part.is_symlink() and not getattr(part, 'is_junction', lambda: False)(),
                'Symlink/junction rejected: ' + str(part))
    return path.resolve()


def exact_file(path):
    path = no_links(path)
    require(path.is_file(), 'Missing regular evidence file: ' + str(path))
    return path


def recheck_record(root, audit_dir, old, task, batch, bundle, matrix, provenance,
                   validator, seconds, references):
    """Only the configuration-prefix findings may be superseded by v2 checks."""
    record = copy.deepcopy(old)
    record.update(original_configuration_passed=old.get('configuration_passed'),
                  original_configuration=old.get('configuration'), original_failures=list(old['failures']),
                  configuration_passed=False, accepted_path_audit_passed=False,
                  reused_native_ccd_verified=False, retained_trace_identity_verified=False,
                  reaggregation_evidence_verified=False, cleared_configuration_failures=[])
    name = task['name']; run = root / 'runs/active' / name
    verified = {}
    try:
        require(all(old.get(key) == task[key] for key in ('scene_key', 'role', 'repeat'))
                and old['run'] == name, 'Old audit run identity differs from the current plan')
        require(isinstance(old['failures'], list) and all(isinstance(v, str) for v in old['failures']),
                'Malformed original failure list')
        record_path = exact_file(audit_dir / (name + '.record.json'))
        require(read(record_path) == old, 'Original audit row differs from its saved record')
        verified['record_sha256'] = sha(record_path)
        for suffix, field, hash_field in (('.stdout.log', 'stdout', 'stdout_sha256'),
                                           ('.identity.json', 'identity_json', 'identity_sha256')):
            path = exact_file(audit_dir / (name + suffix))
            require(no_links(Path(old[field])) == path, 'Original evidence path does not match its exact audit directory')
            digest = sha(path)
            if hash_field in old:
                require(digest == old[hash_field], 'Original evidence SHA mismatch: ' + path.name)
            else:
                require(suffix == '.identity.json' and read(path).get('available') is False,
                        'Missing original evidence SHA: ' + path.name)
            verified[hash_field] = digest
        identity_path = audit_dir / (name + '.identity.json')
        previous_inventory = read(identity_path)
        require(previous_inventory.get('available') is not False,
                'Original accepted-state inventory was unavailable; cannot certify reused CCD')
        # This is a read-only inventory, not another CCD solve. It hashes every
        # accepted state, physical endpoint, topology and initial-state input.
        current_inventory = original_wrapper.inspect_trace(run, task['expanded_config']['steps'], seconds)
        require(current_inventory == previous_inventory, 'Retained trace/state identity or observations changed since the CCD audit')
        require(all(old.get(field) == previous_inventory[field] for field in INVENTORY_FIELDS),
                'Saved audit observations differ from the verified inventory')
        for field in INVENTORY_FIELDS:
            record[field] = current_inventory[field]
        identity = current_inventory['initial_identity_sha256']
        reference = references.setdefault(task['scene_key'], identity)
        require(old.get('same_scene_initial_identity') == (reference == identity),
                'Same-scene identity observation changed')
        record['retained_trace_identity_verified'] = True
        ccd_path = exact_file(audit_dir / (name + '.ccd.json'))
        require(no_links(Path(old['native_ccd_json'])) == ccd_path and old.get('validator_invoked') is True,
                'Original native CCD path/invocation differs')
        require(old.get('command') == [str(validator), str(run / 'trace'), str(ccd_path), 'substeps', '--stable-nh1'],
                'Original native CCD command differs from the verified trace/validator')
        require(sha(ccd_path) == old['native_ccd_sha256'] and read(ccd_path) == old['ccd'],
                'Native CCD JSON content/SHA differs from the original audit')
        verified['native_ccd_sha256'] = sha(ccd_path)
        ccd = old['ccd']
        record['validator_coverage_passed'] = bool(ccd.get('scope') == 'accepted substeps'
            and ccd.get('paths_checked') == current_inventory['expected_validator_paths']
            and ccd.get('stopped_after_first_collision') is False)
        record['ccd_observation_passed'] = bool(old.get('validator_exit_code') == 0
            and not old.get('validator_timed_out') and ccd.get('passed') is True and ccd.get('finite') is True
            and ccd.get('conservative_collision_flags') == 0 and ccd.get('abd_tet_inversions') == 0)
        record['reused_native_ccd_verified'] = True
        # Revalidate configuration only after the old numerical evidence is
        # bound to the unchanged trace, program, source and ledger identities.
        try:
            record['configuration'] = analyzer.validate_run(task, batch, root, bundle, matrix, provenance)
            record['configuration_passed'] = True
            cleared = [v for v in old['failures'] if v.startswith(CONFIG_PREFIX)]
            record['cleared_configuration_failures'] = cleared
            record['failures'] = [v for v in old['failures'] if not v.startswith(CONFIG_PREFIX)]
        except (ValueError, KeyError, OSError, StopIteration, AssertionError) as error:
            message = CONFIG_PREFIX + str(error)
            if message not in record['failures']:
                record['failures'].append(message)
        # Preserve every original non-configuration failure, even if an old
        # field looked inconsistent; never silently repair numerical findings.
        if not record['validator_coverage_passed']:
            message = 'Native validator did not cover every inventoried accepted path/bridge'
            if message not in record['failures']: record['failures'].append(message)
        if not record['ccd_observation_passed']:
            message = 'Native independent CCD/finite/ABD/ground observation failed'
            if message not in record['failures']: record['failures'].append(message)
        record['reaggregation_evidence_verified'] = True
        record['accepted_path_audit_passed'] = bool(not record['failures'] and record['configuration_passed']
            and record['coverage_passed'] and record['validator_coverage_passed'] and record['ccd_observation_passed'])
    except (ValueError, KeyError, OSError, StopIteration, AssertionError, TimeoutError) as error:
        record['failures'].append('Reaggregation evidence: ' + str(error))
        record['accepted_path_audit_passed'] = False
    record['reaggregation_verified_files'] = verified
    return record


def summarize(records):
    verified = [row for row in records if row.get('reused_native_ccd_verified')
                and row.get('retained_trace_identity_verified')]
    return {'all_54_accepted_path_audits_passed': len(records) == 54
                and all(row['accepted_path_audit_passed'] for row in records),
            'actual_validator_reports': len(verified),
            'actual_validator_paths_checked': sum(row['ccd'].get('paths_checked', 0) for row in verified),
            'actual_collision_flags_from_available_reports': sum(row['ccd'].get('conservative_collision_flags', 0)
                                                                  for row in verified),
            'collision_flag_total_covers_all_54_full_windows': len(verified) == 54
                and all(row['coverage_passed'] and row['validator_coverage_passed'] for row in verified)}


def reaggregate(root, audit_dir, output, seconds=60):
    root, audit_dir, output = no_links(root), no_links(audit_dir), no_links(output)
    require(audit_dir.is_relative_to(root) and output.is_relative_to(root) and not output.is_relative_to(audit_dir),
            'Use an audit inside the root and a fresh output outside the original audit directory')
    require(not output.exists() and 1 <= seconds <= 120, 'Fresh output and finite inventory budget required')
    audit_path = exact_file(audit_dir / 'audit.json'); original_sha = sha(audit_path); old = read(audit_path)
    require(old.get('planned_runs') == old.get('recorded_runs') == 54 and len(old.get('runs', [])) == 54
            and all(field in old for field in FINAL_FIELDS), 'Original audit has not finalized all 54 runs')
    require(no_links(Path(old['download_root'])) == root, 'Original audit belongs to a different trace root')
    wrapper = exact_file(root / 'tools/active/audit_autodl_factor.py')
    require(sha(wrapper) == old['wrapper_sha256'] == sha(Path(original_wrapper.__file__)),
            'Original wrapper source differs from the code that produced the audit')
    analyzer_path = exact_file(root / 'tools/active/analyze_autodl_factor_v2.py')
    require(sha(analyzer_path) == sha(Path(analyzer.__file__)), 'Imported v2 analyzer differs from target root')
    validator = exact_file(Path(old['validator']))
    require(validator.is_relative_to(root) and sha(validator) == old['validator_sha256'],
            'Original native CPU validator bytes changed')
    validator_source = exact_file(root / 'tools/validator/diagnose_first_path.cpp')
    require(sha(validator_source) == old['validator_source_sha256'], 'CPU validator source changed since the original audit')
    plan_path = exact_file(root / 'configs/active/autodl_window.json')
    matrix_path = exact_file(root / 'reports/active/AUTODL_WINDOW_BATCH.json')
    bundle_path = exact_file(root / 'autodl_bundle.json')
    require(sha(plan_path) == old['plan_sha256'] and sha(matrix_path) == old['matrix_sha256']
            and sha(bundle_path) == old['bundle_sha256'], 'Original plan/ledger/bundle changed')
    plan, matrix, bundle = read(plan_path), read(matrix_path), read(bundle_path)
    tasks = analyzer.prepare_tasks(root, plan)
    require([row['name'] for row in tasks] == [row['run'] for row in old['runs']], 'Original audit order differs from full plan')
    provenance = analyzer.verify_download_provenance(root, plan, matrix, bundle, plan_path)
    require(provenance['passed'], 'Current source/executable/ordered-ledger provenance failed: '
            + '; '.join(provenance.get('failures', [])))
    batches = {row['name']: row for row in matrix['runs']}
    require(len(batches) == len(matrix['runs']) == 54 and set(batches) == {row['name'] for row in tasks},
            'Missing or duplicate current window results')
    source_identities = {str(path): sha(path) for path in
                        (audit_path, wrapper, analyzer_path, validator, validator_source, plan_path, matrix_path, bundle_path)}
    report = {key: copy.deepcopy(value) for key, value in old.items() if key != 'runs' and key not in FINAL_FIELDS}
    report.update(schema_version=2, download_provenance=provenance, physical_certified=False,
                  performance_certified=False, runs=[],
                  reanalysis={'original_audit': str(audit_path), 'original_audit_sha256': original_sha,
                    'original_summary': {key: old[key] for key in FINAL_FIELDS},
                    'original_wrapper_sha256': old['wrapper_sha256'], 'v2_analyzer_sha256': sha(analyzer_path),
                    'reaggregator_sha256': sha(Path(__file__)), 'validator_invoked_this_reanalysis': False,
                    'gpu_invoked': False, 'inventory_timeout_seconds_per_run': seconds,
                    'scope': 'Recheck all retained trace identities and v2 configuration checks; reuse unchanged '
                             'native CCD observations. Original audit and non-configuration failures remain preserved.'})
    references = {}
    for task, previous in zip(tasks, old['runs']):
        record = recheck_record(root, audit_dir, previous, task, batches[task['name']], bundle, matrix,
                               provenance, validator, seconds, references)
        report['runs'].append(record)
        print(json.dumps({'run': task['name'], 'configuration_passed': record['configuration_passed'],
                          'evidence_verified': record['reaggregation_evidence_verified'],
                          'accepted_path_audit_passed': record['accepted_path_audit_passed']}), flush=True)
    require(all(sha(Path(path)) == digest for path, digest in source_identities.items()),
            'Global source/audit evidence changed during reaggregation; output not written')
    # Every old small record and native output is rehashed after the complete
    # pass so a concurrently changing audit cannot become a certified reuse.
    suffixes = {'record_sha256': '.record.json', 'stdout_sha256': '.stdout.log',
                'identity_sha256': '.identity.json', 'native_ccd_sha256': '.ccd.json'}
    for row in report['runs']:
        for field, digest in row['reaggregation_verified_files'].items():
            require(sha(exact_file(audit_dir / (row['run'] + suffixes[field]))) == digest,
                    'Original per-run evidence changed during reaggregation')
    report.update(summarize(report['runs']))
    report['reanalysis'].update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_audit_document_unchanged=True,
        all_reused_evidence_verified=all(row['reaggregation_evidence_verified'] for row in report['runs']),
        cleared_configuration_failure_count=sum(len(row['cleared_configuration_failures']) for row in report['runs']))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inventory-timeout-seconds', type=int, default=60)
    args = parser.parse_args()
    report = reaggregate(args.root, args.audit_dir, args.output, args.inventory_timeout_seconds)
    print(json.dumps({key: report[key] for key in FINAL_FIELDS}))
    return int(not report['all_54_accepted_path_audits_passed'])


if __name__ == '__main__':
    raise SystemExit(main())
