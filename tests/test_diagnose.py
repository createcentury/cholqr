import torch
from cholqr import diagnose


def test_diagnose_cpu_quick():
    torch.set_float32_matmul_precision("highest")
    d = diagnose(device="cpu", quick=True)
    for k in ("torch", "cholqr", "device", "device_name", "float32_matmul_precision", "gram_eps_over_u",
              "tf32_in_effect", "first_breakdown_kappa", "default_max_orth", "default_ok", "robust_available",
              "robust_orth_at_1e9", "timing_ms", "runtime_s"):
        assert k in d
    assert all(1 <= v <= 100 for v in d["gram_eps_over_u"].values())
    assert not d["tf32_in_effect"]
    assert d["default_ok"] and d["default_max_orth"] < 1e-4
    assert d["robust_available"] and d["robust_orth_at_1e9"] < 1e-4
    first = d["first_breakdown_kappa"]["1024x64"]
    assert first["cqr2"] is not None and first["scqr3"] is not None and first["cqr2"] < first["scqr3"]
