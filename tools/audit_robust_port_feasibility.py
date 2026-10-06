"""Read-only donor audit and a fixed-input check of the public EE friction map.

This evaluates the algebra transcribed from verified source, not the CUDA binary.
No donor file is changed and no historical timing is reclassified as qualified.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DONOR = ROOT.parent / "externals/history-Robust"
OLD = ROOT.parent / "stiffGIPC/barrier-free/source"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    header = DONOR / "src/backends/cuda/contact_system/al_contact_function.h"
    util = DONOR / "src/backends/cuda/utils/friction_utils.h"
    body = header.read_text(encoding="utf-8").split("void EE_friction_gradient_hessian", 1)[1]
    assert "edge_edge_tan_rel_dx" in body
    assert "point_triangle_jacobi(basis, gamma, J);" in body
    helper = util.read_text(encoding="utf-8")
    for term in ["(1.0 - gamma[0]) * basis.transpose()", "gamma[0] * basis.transpose()",
                 "(gamma[1] - 1.0) * basis.transpose()", "-gamma[1] * basis.transpose()"]:
        assert term in helper

    # Two nonparallel edges, with closest parameters in the interior. Previous
    # geometry, tangent basis and normal force are fixed during friction solve.
    previous = np.array([[0., 0., 0.], [1., 0., 0.], [.25, -.6, .01], [.25, .4, .01]])
    gamma = np.array([.25, .6])
    basis = np.array([[1., 0.], [0., 1.], [0., 0.]])
    weights_ee = np.array([1 - gamma[0], gamma[0], gamma[1] - 1, -gamma[1]])
    weights_pt = np.array([1., -1 + gamma.sum(), -gamma[0], -gamma[1]])
    j_ee = np.hstack([w * basis.T for w in weights_ee])
    j_public = np.hstack([w * basis.T for w in weights_pt])
    movement = np.array([[.005, .002, 0.], [.002, -.001, 0.],
                         [-.001, .001, 0.], [.001, -.002, 0.]]).ravel()
    u = j_ee @ movement
    eps_vh = .0001
    assert np.linalg.norm(u) > eps_vh * 10
    # Above the smoothing threshold friction_energy is mu*lambda*norm(u).
    mu_lambda = .2 * 3.
    def energy(q):
        return mu_lambda * np.linalg.norm(j_ee @ q)
    g2 = mu_lambda * u / np.linalg.norm(u)
    h2 = mu_lambda * (np.eye(2) / np.linalg.norm(u)
                     - np.outer(u, u) / np.linalg.norm(u) ** 3)
    gradient_public = j_public.T @ g2
    gradient_correct = j_ee.T @ g2
    differences = []
    for step in [1e-5, 1e-6, 1e-7, 1e-8]:
        fd = np.array([(energy(movement + step * d) - energy(movement - step * d)) / (2 * step)
                       for d in np.eye(12)])
        differences.append({"step_m": step,
                            "public_gradient_relative_error": float(np.linalg.norm(fd - gradient_public) / np.linalg.norm(fd)),
                            "edge_edge_gradient_relative_error": float(np.linalg.norm(fd - gradient_correct) / np.linalg.norm(fd))})
    assert differences[-1]["edge_edge_gradient_relative_error"] < 1e-7
    assert min(d["public_gradient_relative_error"] for d in differences) > .1
    h_correct = j_ee.T @ h2 @ j_ee
    h_public = j_public.T @ h2 @ j_public
    data = {"schema": "robust-port-source-audit-v1", "date": "2026-10-01",
            "scope": "CPU algebra check of verified source formulas; original CUDA kernel not executed",
            "public_header": {"path": str(header), "sha256": sha(header)},
            "helper": {"path": str(util), "sha256": sha(util)},
            "historical_header_same_bytes": sha(OLD / "src/backends/cuda/contact_system/al_contact_function.h") == sha(header),
            "previous_vertices_m": previous.tolist(), "closest_edge_parameters": gamma.tolist(),
            "eps_vh_m": eps_vh, "tangential_motion_m": u.tolist(),
            "weights_public_pt": weights_pt.tolist(), "weights_correct_ee": weights_ee.tolist(),
            "finite_differences": differences,
            "jacobian_sandwich_relative_difference": float(np.linalg.norm(h_public - h_correct) / np.linalg.norm(h_correct)),
            "sandwich_scope": "Both maps use the exact Hessian of the unsmoothed norm energy as the same 2D input; this is not an execution of Robust's projected Hessian function.",
            "finding": "Public AL EE friction uses PT Jacobian, which is inconsistent with its EE energy on this regular fixed-input example.",
            "limits": "Does not establish frequency or impact in any scene, explain all slowdowns, or prove historical binary used an unchanged header at build time."}
    out = ROOT / "reports/ROBUST_PORT_SOURCE_AUDIT_20261001.json"
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"finding": data["finding"], "finite_differences": differences,
                      "jacobian_sandwich_relative_difference": data["jacobian_sandwich_relative_difference"]}))


if __name__ == "__main__":
    main()
