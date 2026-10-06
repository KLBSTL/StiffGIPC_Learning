# Two scenes, L/M, four execution arms

This controller declares exactly 16 serial runs, each from zero for 100 frames.
Scenes are `cloth_hang_l`, `cloth_sphere7_l`, `cloth_hang_m` and
`cloth_sphere7_m`. Arms are frozen Stiff, active IPC host, active IPC conditional
Graph and active IPC Graph with the selected execution components.
Native materials, full CCD, legacy stopping (.01/min6), PCG rho 1e-4 and K1
remain unchanged. A single sample per arm is diagnostic, not certification.

`run.py` reuses the full-evaluation binary, runner, resource checks and metrics.
Its new seal verifies the original files and explicitly records only reviewed
derived input caches. Generate needed M caches with the original METIS library
before sealing and copy matching bytes to both asset trees; do not replace
existing files. The preparation receipt and original pre-addition seal proof
are mandatory provenance for this recorded session.

```sh
python tools/resolution_eval/run.py seal \
  --parent-seal reports/autodl_full_20261006/full_eval_seal.json \
  --pre-cache-receipt reports/autodl_resolution_20261007/cache_preparation_v2/original_seal_verified.json \
  --cache-preparation-receipt reports/autodl_resolution_20261007/cache_preparation_v2/result.json \
  --output reports/autodl_resolution_20261007/resolution_seal.json
python tools/resolution_eval/run.py init --seal <seal> --session <fresh-session>
xvfb-run -a python tools/resolution_eval/run.py run-next --seal <seal> --session <session> --gpu 0
python tools/resolution_eval/report.py --session <finished-session> \
  --previous <prior-compact-results.json> --output <fresh-report-directory>
python -m unittest discover -s tools/resolution_eval -p '*_test.py' -v
```

`run-next` runs at most one declared task and completes CPU analysis before
advancing. It retains the 120-second native budget and existing resource guards;
failures are not retried. The offline report verifies the task ledger, metadata
and pair identities before reporting ratios; it never starts a simulation.
Output directories and prior evidence are preserved, not overwritten.

Graph gains use host/Graph; other component gains use Graph/combined; total gains
use Stiff/combined. All are net observed ratios, including any workload/state
differences. The recorded results, test logs, exact seals, cache-generation
wrapper/commands, input hashes and complete raw trajectories are retained in the
verified archive referenced by the public gain review. No native solver edits
are part of this controller.
