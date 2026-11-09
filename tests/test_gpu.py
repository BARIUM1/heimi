import subprocess

import pytest

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


def test_gpu_status_parses_all_fields(monkeypatch):
    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="55, 3493, 16303, 7\n", stderr=""))
    assert gpu.gpu_status() == {"temp_c": 55.0, "mem_used_gb": pytest.approx(3493 / 1024), "mem_total_gb": pytest.approx(16303 / 1024), "util_pct": 7.0}


def test_gpu_status_none_on_failure(monkeypatch):
    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(gpu.subprocess, "run", boom)
    assert gpu.gpu_status() is None
