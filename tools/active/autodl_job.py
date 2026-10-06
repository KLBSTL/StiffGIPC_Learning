"""Record a detached, finite build or declared run stage without altering plans."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['build', 'smoke', 'window1', 'window2', 'window3'])
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Linux execution only')
    folder = ROOT / 'jobs'
    folder.mkdir(exist_ok=True)
    status_path = folder / (args.stage + '.json')
    if status_path.exists():
        raise FileExistsError(status_path)
    command = [sys.executable, str(ROOT / 'tools/active/autodl_linux.py')]
    if args.stage == 'build':
        command += ['build', '--gpu', '0', '--arch', '89', '--jobs', '2']
    elif args.stage == 'smoke':
        command = ['xvfb-run', '-a', '-s', '-screen 0 640x480x24'] + command + ['run', '--gpu', '0', '--stage', 'smoke']
    else:
        command = ['xvfb-run', '-a', '-s', '-screen 0 640x480x24', sys.executable,
                   str(ROOT / 'tools/active/autodl_window_chunks.py'), '--gpu', '0', '--repeat', args.stage[-1]]
    now = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    record = {'status': 'running', 'started_utc': now(), 'pid': os.getpid(), 'command': command}
    status_path.write_text(json.dumps(record, indent=2))
    env = dict(os.environ)
    env['PATH'] = '/usr/local/cuda/bin:' + env.get('PATH', '')
    env['PYTHONUNBUFFERED'] = '1'
    try:
        with (folder / (args.stage + '.log')).open('xb') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        record.update(status='completed' if result.returncode == 0 else 'failed', exit_code=result.returncode)
    except BaseException as error:
        record.update(status='failed', error=str(error))
        raise
    finally:
        record['ended_utc'] = now()
        status_path.write_text(json.dumps(record, indent=2))
    return int(record['status'] != 'completed')


if __name__ == '__main__':
    raise SystemExit(main())
