from __future__ import annotations

import unittest

from pro_innovation.gpu_power import (
    GpuPowerSample,
    NvidiaGpuPowerMonitor,
    combine_power_phases,
    summarize_power_samples,
)


class GpuPowerTests(unittest.TestCase):
    def test_phase_summary_reports_average_peak_and_energy(self) -> None:
        summary = summarize_power_samples(
            (
                GpuPowerSample(10.2, 100.0),
                GpuPowerSample(10.4, 200.0),
                GpuPowerSample(11.2, 999.0),
            ),
            started_at=10.0,
            ended_at=11.0,
        )

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["duration_seconds"], 1.0)
        self.assertEqual(summary["average_power_watts"], 150.0)
        self.assertEqual(summary["peak_power_watts"], 200.0)
        self.assertEqual(summary["energy_joules"], 150.0)
        self.assertAlmostEqual(float(summary["energy_wh"]), 150.0 / 3600.0)

    def test_phase_totals_are_energy_weighted(self) -> None:
        total = combine_power_phases(
            (
                {
                    "sample_count": 2,
                    "duration_seconds": 2.0,
                    "average_power_watts": 100.0,
                    "peak_power_watts": 120.0,
                    "energy_joules": 200.0,
                    "energy_wh": 200.0 / 3600.0,
                },
                {
                    "sample_count": 3,
                    "duration_seconds": 1.0,
                    "average_power_watts": 200.0,
                    "peak_power_watts": 220.0,
                    "energy_joules": 200.0,
                    "energy_wh": 200.0 / 3600.0,
                },
            )
        )

        self.assertEqual(total["sample_count"], 5)
        self.assertEqual(total["duration_seconds"], 3.0)
        self.assertAlmostEqual(float(total["average_power_watts"]), 400.0 / 3.0)
        self.assertEqual(total["peak_power_watts"], 220.0)
        self.assertEqual(total["energy_joules"], 400.0)

    def test_monitor_rejects_too_short_sampling_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 50 ms"):
            NvidiaGpuPowerMonitor(interval_ms=10)


if __name__ == "__main__":
    unittest.main()
