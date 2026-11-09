import subprocess

from minids import gpu


def test_parses_first_gpu_temperature(monkeypatch):
    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="55\n", stderr=""))
    assert gpu.gpu_temperature_c() == 55.0


def test_missing_nvidia_smi_returns_none(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu.subprocess, "run", boom)
    assert gpu.gpu_temperature_c() is None


def test_garbage_output_returns_none(monkeypatch):
    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="N/A\n", stderr=""))
    assert gpu.gpu_temperature_c() is None
