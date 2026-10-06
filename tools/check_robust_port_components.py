import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
assert data["passed"], "existing AL/volume checks failed"
tests = data.get("port_tests", [])
assert {t['case'] for t in tests} == {'release_age_gpu','reactivation_gpu',
    'pt_frame_snapshot','ee_frame_snapshot','ground_frame_snapshot','singular_geometry_guard','movable_mu_diagonal'}, "Robust port lifecycle/snapshot GPU fixtures missing"
assert all(t["passed"] for t in tests), tests
print("PASS 7 port GPU fixtures and existing AL/warm/volume fixtures")
