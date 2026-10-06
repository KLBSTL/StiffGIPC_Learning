# Selected-frame offline profile analyzer

`tools/local/analyze_profile.py` reads an existing Nsight Systems SQLite export and native cost JSONL. It performs no GPU work, imports no solver or launcher, and opens SQLite read-only. The default selected physical frame is 57. This is diagnostic accounting, not a performance certificate.

## Provenance

The implementation is standalone. It adapts interval union/clipping, completed NVTX decoding, unique runtime correlation, and exact kernel-symbol classification from these donor files; it does not import the old project at runtime.

| Donor under `../stiff_toi_cudagraph_20260929/` | SHA256 |
|---|---|
| `tools/active/contact_linear_profile.py` | `f70bd5af57a4ed9e6bcbd9e387de66adf4ad28d42e48c372067d562d146f9376` |
| `tools/active/analyze_ipc_light_cost.py` | `651774f6989a96ffd37023af27e35b86b0e1294355e6e78eadf0646afe94428a` |

Kernel allowlist additions were checked against current native collision, matrix-conversion, SpMV, ABD-preconditioner and PCG declarations. Unknown symbols remain individually listed. CUB reductions without an identified operand have their own unresolved-operand category. There is no fallback that silently labels an unknown kernel as a known operation.

## Capture contract and accounting

- Every native cost record must be for the requested frame, `sample_kind=production`, `measurement_mode=nvtx_cpu_only`, `nvtx_available=true`, and `gpu_events_enabled=false`. Exactly one native `ipc.physical_frame` with `parent_scope_id=0` must exist.
- Exactly one completed physical-frame NVTX range must remain after collapsing exact/nested ranges on the same CPU thread. Other overlapping physical ranges fail as ambiguous. NVTX has no numeric frame ID here; keep the launcher's configuration/result evidence to establish its correspondence to native frame 57.
- A real selected-window kernel activity is required. A graph-level export without kernel nodes is rejected; use the node trace for operator attribution.
- For a GPU activity interval `I` and normalized frame window `W`, measured work is `I intersect W`. `sum_ms` is the sum of these clipped durations; `union_ms` is the length of their union. Concurrent GPU activities can make the sum exceed the union. Each activity has exactly one category; category union lengths must not be added to produce total covered time.
- CPU NVTX stages report inclusive **union** envelopes. Parent and child CPU scopes are never summed into a component total. Runtime API, explicit wait, and memcpy API intervals remain separate from GPU work and are not added to it.
- A unique runtime correlation can locate a CPU replay-launch envelope, but cannot recover the internal Graph node's captured CostScope. Therefore `node_internal_nvtx_operator_unattributed` conservatively retains all Graph-node intervals even when kernel-symbol classification succeeds. This is different from unknown kernel symbols.
- The uncovered time between first and last graph-node execution includes dependencies, scheduling and unobserved work. It is neither measured removable CPU overhead nor a predicted speedup.
- Asynchronous work can cross the CPU frame boundaries. This report describes a selected timeline window, not proof of causal frame ownership or full capture completeness. Boundary-crossing and outside-window activities are counted. Missing activity tables are explicitly listed.

Input SHA256 values are checked again after analysis. The result records analyzer and donor identities. Existing output files are rejected; output creation is exclusive.

## Invocation

From `E:/university_class/ComputerGraphics/GIPC/StiffGIPC_Learning`:

```powershell
& 'E:/Anaconda/envs/DL/python.exe' tools/local/analyze_profile.py --sqlite '<trace.sqlite>' --cost-jsonl '<cost.jsonl>' --output '<new-analysis.json>'
```

The command prints only selected frame, CPU NVTX wall time, GPU sum/union, unknown-kernel union, unresolved internal Graph-scope union, and output path. Detailed symbols/categories and limitations are stored in JSON.

## CPU validation

```powershell
& 'E:/Anaconda/envs/DL/python.exe' -m unittest analyze_profile_contracts_test -v
```

Working directory: `tools/local`. Six tests passed on 2026-10-06. They cover nested-window normalization, interval union, duplicate/nested NVTX ranges, CPU wait/GPU separation, unknown symbols, unresolved Graph attribution despite a known replay owner, missing NVTX-frame failure, mismatched/non-native cost rejection, input immutability, and exclusive output creation. Synthetic GPU activity totals are 50 ms summed and 40 ms union, alongside a separate 100 ms CPU frame envelope and 30 ms CPU wait; no total combines these quantities.

No Nsight capture or GPU experiment was launched by this analyzer implementation task. After the parent produced a verified capture, the offline command also processed `runs/local_step4_profile_exports_20261006/fixed_off.sqlite` with `runs/local_step4_profile_fixed_off_node_20261006/cost.jsonl` and wrote a new `PROFILE_OFF.json` in this report directory.

That real frame-57 export passed the analyzer contract: 266 native cost records, one canonical completed NVTX frame, no boundary-crossing or outside-window GPU records. It reports 161.738546 ms CPU NVTX wall, 112.626809 ms summed GPU activity, and 112.626585 ms GPU interval union. The 34.220970 ms of unclassified symbols remain individually visible, including the named edge-triangle intersection and FEM assembly kernels; these are not claimed to be overhead. Graph-node activity has 34.625212 ms union without unique runtime correlations, so its internal NVTX scope attribution remains unresolved. CPU `graph.final_readback` is an inclusive waiting envelope and cannot be added to the GPU cost.
