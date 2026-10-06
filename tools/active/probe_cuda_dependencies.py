"""Read-only NVCC dependency probe using the last actual compiler invocation."""
import argparse
import json
import re
import subprocess
from pathlib import Path
from build_support import actual_compile_commands

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / 'builds/active'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='sources/stiff_active/StiffGIPC/app/gl_main.cu')
    args = parser.parse_args()
    source = (ROOT / args.source).resolve()
    row = actual_compile_commands(BUILD, BUILD / 'provenance')[str(source).lower()]
    command = row['command'].replace('--use-local-env', '').replace('--compile', '--generate-dependencies')
    out = BUILD / 'dependency_probe'
    out.mkdir(exist_ok=True)
    dep = out / (source.stem + '.d')
    command = re.sub(r'(?<!\S)-o\s+(?:"[^"]+"|\S+)', '-o "' + dep.as_posix() + '"', command)
    (out / (source.stem + '.command.json')).write_text(json.dumps({'command': command, 'source_compile': row}, indent=2))
    result = subprocess.run(command, cwd=BUILD / 'inherited-v50', capture_output=True, text=True)
    (out / (source.stem + '.log')).write_text(result.stdout + result.stderr)
    print(json.dumps({'returncode': result.returncode, 'dependency_file': str(dep), 'output': (result.stdout + result.stderr)[-3000:]}))
    if dep.exists():
        print('\n'.join(x for x in dep.read_text().splitlines() if 'MASPreconditioner' in x or 'PCG_SOLVER' in x))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
