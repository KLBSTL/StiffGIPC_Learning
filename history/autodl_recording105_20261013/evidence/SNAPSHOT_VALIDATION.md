# History snapshot validation

The exact native solver was already clean-built and fully measured on AutoDL;
the archiving operation made no native source changes.

Commands run from this branch:

```powershell
& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B tools/remote/verify_history_snapshot_20261013.py
# From tools/bench, with TEMP and TMP set to the history checkout's tmp/history_cpu_contracts:
& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B contracts_test.py
# From tools/paper_suite:
& 'E:/Anaconda/envs/DL/python.exe' -X utf8 -B contracts_test.py
git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --cached --check -- HISTORY_SNAPSHOT.md README.md .gitattributes tools/remote/verify_history_snapshot_20261013.py
git diff --cached --exit-code -- baseline/
```

Results:

- All 355 native/input files and CMake match the actual observer snapshot.
- All 466 original transfer files match their staged Git blob SHA256.
- 124 Python sources compiled before adding the two historical launch scripts;
  those two scripts were separately compiled after copying.
- All 18 numerical datasets retain exact source content, including gzip.
- 23 bench CPU contracts and 16 paper-suite CPU contracts passed.
- New documentation/verifier whitespace checks passed; the exact sealed source
  retains its historical whitespace and CRLF bytes.
- Frozen `baseline/` has no staged changes.
- No private-key/token-shaped content, generated binaries or >=100 MiB files
  found among staged changed files at the recorded scan.

The first bench test attempt encountered Windows sandbox temporary-directory
permissions. The full 23-test suite was rerun successfully using an explicit
workspace TEMP/TMP directory. Both logs are preserved. No GPU simulation was
rerun for the Git archive.
