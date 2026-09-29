#!/usr/bin/env python3
"""Run a serial, seeded Pocman scenario sweep and summarize its CSV records."""

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PRESETS = {
    "pilot": dict(ks=[16, 32, 64, 128, 256, 512], seeds=[11, 22, 33],
                  steps=40, time=0.05, depth=30, belief_particles=5000,
                  repeats=5, probe_every=10),
    "study": dict(ks=[16, 32, 64, 128, 256, 512, 1024], seeds=list(range(100, 130)),
                  steps=200, time=0.2, depth=90, belief_particles=50000,
                  repeats=20, probe_every=20),
}


def integers(value):
    try:
        items = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not items or len(items) != len(set(items)):
        raise argparse.ArgumentTypeError("values must be nonempty and unique")
    return items


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="pilot")
    parser.add_argument("--ks", type=integers)
    parser.add_argument("--seeds", type=integers)
    for option in ("steps", "depth", "belief-particles", "repeats", "probe-every"):
        parser.add_argument("--" + option, type=int)
    parser.add_argument("--time", type=float, help="tree-search CPU seconds per decision")
    parser.add_argument("--prune", type=float, default=0.01,
                        help="fixed regularization penalty per policy node (default 0.01)")
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--reference-k", type=int,
                        help="controller K for the fixed-belief probe trajectory; default max K")
    parser.add_argument("--output", type=Path, help="new directory; never overwrites an existing run")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--no-analysis", action="store_true")
    args = parser.parse_args()
    for key, value in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.ks = sorted(args.ks)
    if args.reference_k is None:
        args.reference_k = max(args.ks)
    if min(args.ks) < 1 or min(args.seeds) < 0 or max(args.seeds) > 2**32 - 1:
        parser.error("K must be positive and seeds must be uint32 integers")
    for key in ("steps", "depth", "belief_particles", "repeats", "probe_every", "reference_k"):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if args.depth < 2 or args.repeats < 2:
        parser.error("depth and repeats must be at least 2")
    if args.belief_particles < 2500:
        parser.error("use at least 2500 belief particles to avoid stock automatic particle splitting")
    if not (0 < args.time < float("inf")) or not (0 <= args.prune < float("inf")):
        parser.error("time must be finite and positive; prune finite and nonnegative")
    if not 0 < args.discount < 1:
        parser.error("discount must be strictly between 0 and 1")
    return args


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_csv(paths, output):
    header = None
    with output.open("w", newline="") as target:
        writer = csv.writer(target)
        for path in paths:
            with path.open(newline="") as source:
                reader = csv.reader(source)
                incoming = next(reader)
                if header is None:
                    header = incoming
                    writer.writerow(header)
                elif incoming != header:
                    raise RuntimeError(f"CSV header mismatch in {path}")
                writer.writerows(reader)


def main():
    args = parse_args()
    if not args.skip_build:
        subprocess.run(["make", "-C", str(HERE), "-j2"], check=True)
    binary = HERE / "pocman_k"
    if not binary.is_file():
        raise RuntimeError(f"missing executable: {binary}")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output = (args.output or HERE / "results" / (args.preset + "-" + stamp)).resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = output / "raw"
    raw.mkdir()
    settings = vars(args).copy()
    settings["output"] = str(output)
    manifest = dict(settings=settings, started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                    platform=platform.platform(), python=sys.version,
                    binary_sha256=digest(binary), status="running", commands=[],
                    model="FullPocman", lower_bound="SMART", upper_bound="SMART",
                    sampling="systematic resampling with explicitly seeded Fisher-Yates permutation; K samples are not nested",
                    seed_scheme="independent hashed domains for world, belief, search and rollout",
                    notes="Serial runs; fixed search CPU budget; wall planning includes setup and cleanup.")
    # Hash all sources so a run remains identifiable even in an untracked checkout.
    sources = list((ROOT / "src").rglob("*.cpp")) + list((ROOT / "include").rglob("*.h"))
    sources += list((ROOT / "examples/cpp_models/pocman/src").glob("*.cpp"))
    sources += list((ROOT / "examples/cpp_models/pocman/src").glob("*.h"))
    sources += list(HERE.glob("*.cpp")) + list(HERE.glob("*.py")) + [HERE / "Makefile"]
    manifest["source_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in sorted(sources)}
    manifest_path = output / "manifest.json"

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    save_manifest()
    common = ["--steps", str(args.steps), "--time", str(args.time), "--depth", str(args.depth),
              "--prune", str(args.prune), "--discount", str(args.discount),
              "--belief-particles", str(args.belief_particles), "--repeats", str(args.repeats),
              "--probe-every", str(args.probe_every), "--reference-k", str(args.reference_k)]
    jobs = [("episode", seed, str(k)) for seed in args.seeds for k in args.ks]
    jobs += [("probe", seed, ",".join(map(str, args.ks))) for seed in args.seeds]
    # Reduce systematic ordering/thermal bias without running timing jobs concurrently.
    random.Random(20260928).shuffle(jobs)
    generated = {"steps.csv": [], "episodes.csv": [], "probes.csv": []}
    start = time.monotonic()
    try:
        for index, (mode, seed, ks) in enumerate(jobs, 1):
            name = f"{mode}-seed{seed}" + (f"-k{ks}" if mode == "episode" else "")
            directory = raw / name
            directory.mkdir()
            command = [str(binary), "--mode", mode, "--seed", str(seed), "--k", ks,
                       "--output", str(directory)] + common
            manifest["commands"].append(command)
            save_manifest()
            print(f"[{index}/{len(jobs)}] {name} ({time.monotonic()-start:.1f}s elapsed)", flush=True)
            with (directory / "run.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"{name} failed ({result.returncode}); see {directory / 'run.log'}")
            for filename in generated:
                if (directory / filename).is_file():
                    generated[filename].append(directory / filename)
        for filename, paths in generated.items():
            if not paths:
                raise RuntimeError(f"no records generated for {filename}")
            merge_csv(paths, output / filename)
        manifest.update(status="complete", elapsed_s=time.monotonic()-start,
                        completed_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        save_manifest()
        if not args.no_analysis:
            subprocess.run([sys.executable, str(HERE / "analyze.py"), str(output)], check=True)
        print(f"Results: {output}")
    except BaseException as exc:
        manifest.update(status="failed", error=str(exc), elapsed_s=time.monotonic()-start)
        save_manifest()
        raise


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
