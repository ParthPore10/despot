"""Regression checks for RNG isolation in the compiled Pocman harness.

Build with `make`, then run `python3 -m unittest test_harness.py` here.
"""

import csv
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


EXECUTABLE = Path(__file__).resolve().parent / "pocman_k"
TIMING_FIELDS = {"planning_wall_s", "planning_cpu_s", "search_cpu_s"}


@unittest.skipUnless(
    EXECUTABLE.is_file() and os.access(EXECUTABLE, os.X_OK),
    "Build the Pocman experiment executable with make first",
)
class HarnessRngTests(unittest.TestCase):
    def test_probe_order_does_not_change_decisions_or_reference_beliefs(self):
        # A tiny positive budget makes ConstructTree complete exactly one trial.
        # This removes CPU scheduling noise while preserving deep rollouts and
        # scenario-stream assignment. Depth 1 would miss libc++'s unseeded
        # random_shuffle bug: it can change gaps/actions when K order changes.
        with tempfile.TemporaryDirectory(prefix="pocman-k-order-") as temporary:
            results = []
            for name, ks in [("forward", "8,32"), ("reverse", "32,8")]:
                output = Path(temporary) / name
                subprocess.run(
                    [
                        str(EXECUTABLE),
                        "--mode", "probe",
                        "--k", ks,
                        "--seed", "1",
                        "--steps", "3",
                        "--time", "0.000000001",
                        "--depth", "8",
                        "--belief-particles", "2500",
                        "--repeats", "3",
                        "--probe-every", "1",
                        "--reference-k", "16",
                        "--output", str(output),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                with (output / "probes.csv").open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 18)
                for row in rows:
                    self.assertEqual(int(row["trials"]), 1)
                results.append(
                    sorted(
                        tuple(sorted((key, value) for key, value in row.items()
                                     if key not in TIMING_FIELDS))
                        for row in rows
                    )
                )
            # Equality at later steps also checks that probe plans do not alter
            # the independently seeded trajectory or its belief updates.
            self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
