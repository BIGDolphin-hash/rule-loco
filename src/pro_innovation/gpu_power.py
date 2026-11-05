"""Lightweight NVIDIA board-power sampling for evaluation runs."""

from __future__ import annotations

import atexit
import csv
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Mapping, Sequence


@dataclass(frozen=True)
class GpuPowerSample:
    timestamp_seconds: float
    power_watts: float


def summarize_power_samples(
    samples: Sequence[GpuPowerSample],
    *,
    started_at: float,
    ended_at: float,
) -> dict[str, float | int | None]:
    """Summarize samples within one measured phase.

    Energy uses the phase-average sampled board power multiplied by the exact
    phase duration.  This makes startup and inference energy directly additive.
    """

    duration = max(0.0, float(ended_at) - float(started_at))
    selected = [
        sample
        for sample in samples
        if started_at <= sample.timestamp_seconds <= ended_at
    ]
    if not selected:
        return {
            "sample_count": 0,
            "duration_seconds": duration,
            "average_power_watts": None,
            "peak_power_watts": None,
            "energy_joules": None,
            "energy_wh": None,
        }
    average = sum(sample.power_watts for sample in selected) / len(selected)
    peak = max(sample.power_watts for sample in selected)
    energy_joules = average * duration
    return {
        "sample_count": len(selected),
        "duration_seconds": duration,
        "average_power_watts": average,
        "peak_power_watts": peak,
        "energy_joules": energy_joules,
        "energy_wh": energy_joules / 3600.0,
    }


def combine_power_phases(
    phases: Sequence[Mapping[str, float | int | None]],
) -> dict[str, float | int | None]:
    """Combine non-overlapping phase summaries into one energy total."""

    available = [phase for phase in phases if phase.get("energy_joules") is not None]
    duration = sum(float(phase.get("duration_seconds") or 0.0) for phase in phases)
    sample_count = sum(int(phase.get("sample_count") or 0) for phase in phases)
    if not available:
        return {
            "sample_count": sample_count,
            "duration_seconds": duration,
            "average_power_watts": None,
            "peak_power_watts": None,
            "energy_joules": None,
            "energy_wh": None,
        }
    energy_joules = sum(float(phase["energy_joules"]) for phase in available)
    peaks = [
        float(phase["peak_power_watts"])
        for phase in available
        if phase.get("peak_power_watts") is not None
    ]
    measured_duration = sum(
        float(phase.get("duration_seconds") or 0.0) for phase in available
    )
    return {
        "sample_count": sample_count,
        "duration_seconds": duration,
        "average_power_watts": (
            energy_joules / measured_duration if measured_duration > 0.0 else None
        ),
        "peak_power_watts": max(peaks) if peaks else None,
        "energy_joules": energy_joules,
        "energy_wh": energy_joules / 3600.0,
    }


class NvidiaGpuPowerMonitor:
    """Sample whole-board NVIDIA GPU power through one persistent nvidia-smi."""

    def __init__(self, *, gpu_index: int = 0, interval_ms: int = 200) -> None:
        if gpu_index < 0:
            raise ValueError("gpu index must be non-negative")
        if interval_ms < 50:
            raise ValueError("GPU power sampling interval must be at least 50 ms")
        self.gpu_index = int(gpu_index)
        self.interval_ms = int(interval_ms)
        self.error: str | None = None
        self.started_at: float | None = None
        self._samples: list[GpuPowerSample] = []
        self._lock = Lock()
        self._first_sample = Event()
        self._process: subprocess.Popen[str] | None = None
        self._reader: Thread | None = None
        self._registered_with_atexit = False

    @property
    def samples(self) -> tuple[GpuPowerSample, ...]:
        with self._lock:
            return tuple(self._samples)

    def start(self) -> bool:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            self.error = "nvidia-smi is unavailable"
            return False
        command = [
            executable,
            f"--id={self.gpu_index}",
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            self.error = f"cannot start nvidia-smi: {exc}"
            return False
        self.started_at = perf_counter()
        self._reader = Thread(
            target=self._read_samples,
            name="gpu-power-sampler",
            daemon=True,
        )
        self._reader.start()
        atexit.register(self.stop)
        self._registered_with_atexit = True
        wait_seconds = max(2.0, self.interval_ms / 1000.0 * 5.0)
        if not self._first_sample.wait(timeout=wait_seconds):
            self.error = "nvidia-smi produced no power samples"
            self.stop()
            return False
        return True

    def _read_samples(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        for line in self._process.stdout:
            value = line.strip()
            if not value:
                continue
            try:
                watts = float(value)
            except ValueError:
                continue
            sample = GpuPowerSample(perf_counter(), watts)
            with self._lock:
                self._samples.append(sample)
            self._first_sample.set()

    def stop(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3.0)
        if self._reader is not None and self._reader.is_alive():
            self._reader.join(timeout=3.0)
        if self._registered_with_atexit:
            atexit.unregister(self.stop)
            self._registered_with_atexit = False

    def payload(
        self,
        phases: Mapping[str, tuple[float, float]],
    ) -> dict[str, object]:
        samples = self.samples
        phase_payload = {
            name: summarize_power_samples(
                samples,
                started_at=boundaries[0],
                ended_at=boundaries[1],
            )
            for name, boundaries in phases.items()
        }
        total = combine_power_phases(tuple(phase_payload.values()))
        available = bool(samples) and all(
            phase["sample_count"] for phase in phase_payload.values()
        )
        return {
            "available": available,
            "gpu_index": self.gpu_index,
            "sampling_interval_ms": self.interval_ms,
            "scope": "whole_gpu_board_power",
            "includes_other_gpu_processes": True,
            "error": self.error,
            **phase_payload,
            "total": total,
        }

    def write_csv(
        self,
        path: str | Path,
        *,
        phases: Mapping[str, tuple[float, float]],
    ) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        origin = self.started_at or (
            self.samples[0].timestamp_seconds if self.samples else 0.0
        )
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("elapsed_seconds", "power_watts", "phase"))
            for sample in self.samples:
                phase_name = "between_phases"
                for name, (started_at, ended_at) in phases.items():
                    if started_at <= sample.timestamp_seconds <= ended_at:
                        phase_name = name
                        break
                writer.writerow(
                    (
                        f"{sample.timestamp_seconds - origin:.6f}",
                        f"{sample.power_watts:.6f}",
                        phase_name,
                    )
                )
