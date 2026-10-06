"""Inspect the actual exported endpoints of v31's first flagged PT path."""
from decimal import Decimal, localcontext
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
diagnostic = json.loads((ROOT / "reports/v31_c_friction_first_ccd.json").read_text())
hit = diagnostic["first_collision"]
assert hit["kind"] == "vertex_face"
trace = ROOT / "runs/local/local_v31r2_warm_c_friction_diagnose60/trace"
body = np.fromfile(trace / "body_ids.bin", dtype="<i4")


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


endpoints = []
with localcontext() as context:
    context.prec = 60
    for filename in [hit["from"], hit["to"]]:
        points = np.fromfile(trace / "substeps" / filename, dtype="<f8").reshape(-1, 3)[hit["ids"]]
        p, a, b, c = [[Decimal.from_float(float(value)) for value in vertex] for vertex in points]
        ab, ac, ap = sub(b, a), sub(c, a), sub(p, a)
        normal = cross(ab, ac)
        twice_area = dot(normal, normal).sqrt()
        d00, d01, d11 = dot(ab, ab), dot(ab, ac), dot(ac, ac)
        d20, d21 = dot(ap, ab), dot(ap, ac)
        denominator = d00 * d11 - d01 * d01
        u = (d11 * d20 - d01 * d21) / denominator
        v = (d00 * d21 - d01 * d20) / denominator
        barycentric = [Decimal(1) - u - v, u, v]
        endpoints.append({"state": filename, "signed_face_plane_distance_m": float(dot(ap, normal) / twice_area),
                          "unsigned_face_plane_distance_m": float(abs(dot(ap, normal) / twice_area)),
                          "triangle_twice_area_m2": float(twice_area), "projected_barycentric": [float(x) for x in barycentric],
                          "projection_inside_triangle": all(0 <= x <= 1 for x in barycentric)})
output = {"first_collision": hit, "all_ids_are_cloth": bool((body[hit["ids"]] == -1).all()), "endpoints": endpoints,
          "calculation": "60-digit Decimal arithmetic on exact stored binary64 endpoint values",
          "scope": "Endpoint plane-distance inspection only; no continuous-path verdict, no relaxation of the 1e-9 CCD tolerance or zero-flag gate."}
(ROOT / "reports/v31_c_friction_first_path_geometry.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
print(json.dumps(output))
