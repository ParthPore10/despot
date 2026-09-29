#!/usr/bin/env python3
"""Analyze a Pocman scenario-count sweep with no required Python dependencies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


METRICS = (
    "planning_wall_s", "planning_cpu_s", "search_cpu_s", "tree_nodes",
    "policy_nodes", "expanded_nodes", "trials", "initial_gap", "final_gap",
    "belief_particles",
)
INTEGER_FIELDS = {
    "seed", "k", "step", "steps", "repeat", "planner_seed", "action",
    "observation", "terminal", "tree_nodes", "policy_nodes", "expanded_nodes",
    "trials", "belief_particles",
}
REQUIRED = {
    "episodes": ("seed", "k", "steps", "return", "discounted_return", "terminal"),
    "steps": ("seed", "k", "step", "action", "observation", "reward", "terminal") + METRICS,
    "probes": ("seed", "step", "k", "repeat", "planner_seed", "action") + METRICS,
}
STABILITY_METRICS = (
    "pairwise_agreement", "modal_share", "entropy_bits", "agreement_to_max_k_mode",
)


class InputError(ValueError):
    """A missing or inconsistent experiment input."""


def read_table(path: Path, table: str) -> list[dict]:
    """Parse known fields strictly, retaining optional columns for future versions."""
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise InputError(f"{path.name}: missing CSV header")
            if len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise InputError(f"{path.name}: duplicate CSV column names")
            missing = set(REQUIRED[table]) - set(reader.fieldnames)
            if missing:
                raise InputError(f"{path.name}: missing columns: {', '.join(sorted(missing))}")
            rows = []
            for line, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise InputError(f"{path.name}:{line}: row length does not match header")
                parsed = dict(row)
                for field in REQUIRED[table]:
                    try:
                        value = int(row[field]) if field in INTEGER_FIELDS else float(row[field])
                    except (ValueError, TypeError) as exc:
                        raise InputError(f"{path.name}:{line}: invalid {field}={row[field]!r}") from exc
                    if not math.isfinite(value):
                        raise InputError(f"{path.name}:{line}: {field} must be finite")
                    if field == "k" and value <= 0:
                        raise InputError(f"{path.name}:{line}: k must be positive")
                    if field == "terminal" and value not in (0, 1):
                        raise InputError(f"{path.name}:{line}: terminal must be 0 or 1")
                    if (field in INTEGER_FIELDS - {"seed", "planner_seed", "observation"}
                            or field.endswith("_s")) and value < 0:
                        raise InputError(f"{path.name}:{line}: {field} must be nonnegative")
                    parsed[field] = value
                rows.append(parsed)
    except OSError as exc:
        raise InputError(f"Cannot read {path}: {exc}") from exc
    if not rows:
        raise InputError(f"{path.name}: no data rows; experiment is incomplete")
    return rows


def load_inputs(run_directory: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    try:
        manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"Cannot read manifest.json: {exc}") from exc
    if not isinstance(manifest, dict):
        raise InputError("manifest.json must contain a JSON object")
    if manifest.get("status") != "complete":
        raise InputError(f"manifest.json: run status is {manifest.get('status')!r}, expected 'complete'")
    tables = {name: read_table(run_directory / f"{name}.csv", name) for name in REQUIRED}
    validate_inputs(tables["episodes"], tables["steps"], tables["probes"])
    validate_manifest(manifest, tables)
    return manifest, tables["episodes"], tables["steps"], tables["probes"]


def validate_manifest(manifest: dict, tables: dict) -> None:
    settings = manifest.get("settings", {})
    if not isinstance(settings, dict):
        raise InputError("manifest.json: settings must be a JSON object")
    for field, column in (("seeds", "seed"), ("ks", "k")):
        declared = settings.get(field)
        if declared is not None:
            if not isinstance(declared, list) or not declared or any(type(value) is not int for value in declared):
                raise InputError(f"manifest.json: settings.{field} must be a nonempty list of integers")
            actual = {row[column] for row in tables["episodes"]}
            if actual != set(declared):
                raise InputError(f"episodes.csv: {column} values {sorted(actual)} disagree with manifest {declared}")
    if "repeats" in settings:
        expected = settings["repeats"]
        if type(expected) is not int or expected < 2:
            raise InputError("manifest.json: settings.repeats must be an integer >= 2")
        counts = Counter((row["seed"], row["step"], row["k"]) for row in tables["probes"])
        if any(count != expected for count in counts.values()):
            raise InputError(f"probes.csv: some fixed beliefs do not have the declared {expected} repeats")
    if "steps" in settings:
        horizon = settings["steps"]
        if type(horizon) is not int or horizon < 1:
            raise InputError("manifest.json: settings.steps must be a positive integer")
        if any(row["steps"] > horizon or (row["steps"] < horizon and not row["terminal"])
               for row in tables["episodes"]):
            raise InputError("episodes.csv: episode length disagrees with declared horizon or ended early without terminal")
        if any(row["step"] >= horizon for row in tables["probes"]):
            raise InputError("probes.csv: belief step exceeds the declared horizon")
    if "discount" in settings:
        discount = settings["discount"]
        if not isinstance(discount, (int, float)) or not 0 < discount < 1:
            raise InputError("manifest.json: settings.discount must be between 0 and 1")
        returns = defaultdict(float)
        for row in tables["steps"]:
            returns[row["seed"], row["k"]] += discount ** row["step"] * row["reward"]
        if any(not math.isclose(row["discounted_return"], returns[row["seed"], row["k"]],
                               rel_tol=1e-6, abs_tol=1e-6) for row in tables["episodes"]):
            raise InputError("episodes.csv: discounted return disagrees with step rewards and declared discount")


def validate_inputs(episodes: list[dict], steps: list[dict], probes: list[dict]) -> None:
    """Detect truncated runs and mismatched joins before producing conclusions."""
    episode_map = {}
    for row in episodes:
        key = row["seed"], row["k"]
        if key in episode_map:
            raise InputError(f"episodes.csv: duplicate episode seed={key[0]}, k={key[1]}")
        if row["steps"] < 1:
            raise InputError(f"episodes.csv: episode {key} has no steps")
        episode_map[key] = row
    seeds = {seed for seed, _ in episode_map}
    ks = {k for _, k in episode_map}
    missing = {(seed, k) for seed in seeds for k in ks} - set(episode_map)
    if missing:
        raise InputError(f"episodes.csv: incomplete seed × K sweep; missing {sorted(missing)}")
    step_groups = defaultdict(list)
    for row in steps:
        key = row["seed"], row["k"]
        if key not in episode_map:
            raise InputError(f"steps.csv: step has no episode: seed={key[0]}, k={key[1]}")
        step_groups[key].append(row)
    for key, episode in episode_map.items():
        group = sorted(step_groups[key], key=lambda row: row["step"])
        if [row["step"] for row in group] != list(range(episode["steps"])):
            raise InputError(f"steps.csv: episode {key} needs consecutive steps 0..{episode['steps'] - 1}")
        if any(row["terminal"] for row in group[:-1]):
            raise InputError(f"steps.csv: episode {key} continues after a terminal step")
        if group[-1]["terminal"] != episode["terminal"]:
            raise InputError(f"steps.csv: terminal flag disagrees with episodes.csv for {key}")
        if not math.isclose(sum(row["reward"] for row in group), episode["return"], rel_tol=1e-6, abs_tol=1e-6):
            raise InputError(f"steps.csv: reward sum disagrees with episode return for {key}")
    probe_groups = defaultdict(list)
    seen = set()
    for row in probes:
        key = row["seed"], row["step"], row["k"], row["repeat"]
        if key in seen:
            raise InputError(f"probes.csv: duplicate seed/step/k/repeat {key}")
        seen.add(key)
        if row["seed"] not in seeds or row["k"] not in ks:
            raise InputError(f"probes.csv: probe uses seed or K outside episode sweep: {key}")
        probe_groups[key[:2]].append(row)
    if {seed for seed, _ in probe_groups} != seeds:
        raise InputError("probes.csv: some episode seeds have no fixed-belief probes")
    for belief, group in probe_groups.items():
        repeats_by_k = defaultdict(set)
        for row in group:
            repeats_by_k[row["k"]].add(row["repeat"])
        if set(repeats_by_k) != ks:
            raise InputError(f"probes.csv: fixed belief {belief} is missing one or more K values")
        repeat_sets = list(repeats_by_k.values())
        if any(len(repeats) < 2 for repeats in repeat_sets):
            raise InputError(f"probes.csv: fixed belief {belief} needs at least two repeats per K")
        if any(repeats != repeat_sets[0] for repeats in repeat_sets[1:]):
            raise InputError(f"probes.csv: fixed belief {belief} has incomplete repeats across K")


def quantile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (position - lower) * (sorted_values[upper] - sorted_values[lower])


def mean_ci(values: list[float], bootstrap: int, rng: random.Random) -> tuple[float | None, float | None]:
    """Percentile bootstrap over independent units; n=1 cannot estimate uncertainty."""
    if len(values) < 2:
        return None, None
    draws = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(bootstrap))
    return quantile(draws, 0.025), quantile(draws, 0.975)


def action_statistics(actions: list[int]) -> dict:
    if len(actions) < 2:
        raise InputError("Action stability requires at least two planning repeats")
    counts = Counter(actions)
    n = len(actions)
    mode = min(counts, key=lambda action: (-counts[action], action))
    return {
        "repeats": n,
        "modal_action": mode,
        "pairwise_agreement": sum(count * (count - 1) for count in counts.values()) / (n * (n - 1)),
        "modal_share": counts[mode] / n,
        "entropy_bits": -sum((count / n) * math.log2(count / n) for count in counts.values()),
    }


def stability_rows(probes: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for probe in probes:
        groups[probe["seed"], probe["step"], probe["k"]].append(probe["action"])
    max_k = max(key[2] for key in groups)
    modes = {key[:2]: action_statistics(actions)["modal_action"]
             for key, actions in groups.items() if key[2] == max_k}
    rows = []
    for (seed, step, k), actions in sorted(groups.items()):
        benchmark = modes[seed, step]
        rows.append({
            "seed": seed, "step": step, "k": k,
            **action_statistics(actions),
            "max_k_mode_action": benchmark,
            "agreement_to_max_k_mode": (sum(action == benchmark for action in actions) / len(actions)
                                        if k != max_k else None),
        })
    return rows


def summarize(episodes: list[dict], steps: list[dict], stability: list[dict], bootstrap: int) -> list[dict]:
    if bootstrap < 1:
        raise ValueError("bootstrap must be at least 1")
    rng = random.Random(81723)
    episode_groups = defaultdict(list)
    step_groups = defaultdict(list)
    stability_groups = defaultdict(list)
    for row in episodes:
        episode_groups[row["k"]].append(row)
    for row in steps:
        step_groups[row["seed"], row["k"]].append(row)
    for row in stability:
        stability_groups[row["k"]].append(row)
    max_k = max(episode_groups)
    benchmark_returns = {row["seed"]: row["return"] for row in episode_groups[max_k]}
    result = []
    for k, group in sorted(episode_groups.items()):
        group = sorted(group, key=lambda row: row["seed"])
        record = {"k": k, "episodes": len(group), "mean_episode_steps": statistics.fmean(row["steps"] for row in group)}
        for metric in ("return", "discounted_return"):
            values = [row[metric] for row in group]
            record[f"mean_{metric}"] = statistics.fmean(values)
            record[f"{metric}_ci_low"], record[f"{metric}_ci_high"] = mean_ci(values, bootstrap, rng)
        deltas = [row["return"] - benchmark_returns[row["seed"]] for row in group]
        delta_metric = "paired_return_delta_vs_max_k"
        record[f"mean_{delta_metric}"] = statistics.fmean(deltas)
        record[f"{delta_metric}_ci_low"], record[f"{delta_metric}_ci_high"] = (
            mean_ci(deltas, bootstrap, rng) if k != max_k else (None, None)
        )
        for metric in METRICS:
            factor = 1000 if metric.endswith("_s") else 1
            name = metric[:-2] + "_ms" if metric.endswith("_s") else metric
            values = [statistics.fmean(row[metric] for row in step_groups[episode["seed"], k]) * factor
                      for episode in group]
            record[f"mean_{name}"] = statistics.fmean(values)
        record["probe_beliefs"] = len(stability_groups[k])
        record["probe_seeds"] = len({row["seed"] for row in stability_groups[k]})
        for metric in STABILITY_METRICS:
            clusters = defaultdict(list)
            for row in stability_groups[k]:
                if row[metric] is not None:
                    clusters[row["seed"]].append(row[metric])
            # Each seed is an independent unit. Its beliefs stay together, and
            # seeds receive equal weight even when trajectories differ in length.
            values = [statistics.fmean(values) for _, values in sorted(clusters.items())]
            record[f"mean_{metric}"] = statistics.fmean(values) if values else None
            record[f"{metric}_ci_low"], record[f"{metric}_ci_high"] = mean_ci(values, bootstrap, rng)
        result.append(record)
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(run_directory: Path, summary: list[dict], settings: dict | None = None) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Warning: matplotlib is unavailable; CSV analysis and report are complete, plots skipped.", file=sys.stderr)
        return False
    ks = [row["k"] for row in summary]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), layout="constrained")

    def interval_plot(axis, metric, color):
        means = [row[f"mean_{metric}"] for row in summary]
        axis.plot(ks, means, "o-", color=color, linewidth=2)
        for k, row, value in zip(ks, summary, means):
            low, high = row[f"{metric}_ci_low"], row[f"{metric}_ci_high"]
            if low is not None and high is not None:
                axis.errorbar(k, value, yerr=[[max(0, value - low)], [max(0, high - value)]],
                              fmt="none", color=color, capsize=4)

    interval_plot(axes[0, 0], "return", "#176b93")
    axes[0, 0].set(title="Episode return", ylabel="Undiscounted cumulative reward")
    axes[0, 1].plot(ks, [row["mean_planning_wall_ms"] for row in summary], "o-", color="#ad5729", linewidth=2)
    axes[0, 1].set(title="Full planner latency", ylabel="Wall time per decision (ms)")
    axes[1, 0].plot(ks, [row["mean_tree_nodes"] for row in summary], "o-", label="Explored belief nodes", color="#176b93", linewidth=2)
    axes[1, 0].plot(ks, [row["mean_policy_nodes"] for row in summary], "s-", label="Greedy expanded policy (action nodes)", color="#ad5729", linewidth=2)
    axes[1, 0].set(title="Explored tree and expanded policy size", ylabel="Nodes per decision")
    counts = [row[metric] for row in summary for metric in ("mean_tree_nodes", "mean_policy_nodes")]
    if min(counts) > 0:
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_ylabel("Nodes per decision (log scale)")
    axes[1, 0].legend(frameon=False)
    interval_plot(axes[1, 1], "pairwise_agreement", "#427a40")
    axes[1, 1].set(title="Action stability at fixed beliefs", ylabel="Pairwise action agreement", ylim=(0, 1))
    for axis in axes.flat:
        axis.set_xscale("log", base=2)
        axis.set_xticks(ks, [str(k) for k in ks])
        axis.set_xlabel("Sampled scenarios K (log₂ scale)")
        axis.grid(alpha=0.22)
    settings = settings or {}
    subtitle = (f"{summary[0]['episodes']} episodes per K · "
                f"{summary[0]['probe_seeds']} reference trajectories · 95% seed-bootstrap intervals")
    if "time" in settings and "steps" in settings:
        subtitle += f"\n{settings['steps']}-step cap · {settings['time'] * 1000:g} ms search CPU budget"
    fig.suptitle("Pocman scenario-count experiment\n" + subtitle, fontsize=13)
    fig.savefig(run_directory / "overview.png", dpi=170)
    fig.savefig(run_directory / "overview.svg")
    plt.close(fig)
    return True


def format_value(value: float | None, digits: int = 3) -> str:
    return "unavailable" if value is None else f"{value:.{digits}f}"


def write_report(run_directory: Path, manifest: dict, summary: list[dict], bootstrap: int, plotted: bool) -> None:
    lines = [
        "# Pocman scenario-count experiment", "",
        "This run measures how sampled scenario count K relates to return, planning cost, tree size, "
        "and repeatability of the chosen action. It describes the observed tradeoffs; "
        "it does not establish an optimal K or demonstrate convergence.", "",
        "## Results", "",
        "| K | Episodes | Mean return (95% CI) | Planner wall ms | Explored belief nodes | Greedy policy action nodes | Pairwise agreement (95% CI) |",
        "| ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary:
        reward = f"{row['mean_return']:.2f} [{format_value(row['return_ci_low'], 2)}, {format_value(row['return_ci_high'], 2)}]"
        agreement = f"{row['mean_pairwise_agreement']:.3f} [{format_value(row['pairwise_agreement_ci_low'])}, {format_value(row['pairwise_agreement_ci_high'])}]"
        lines.append(f"| {row['k']} | {row['episodes']} | {reward} | {row['mean_planning_wall_ms']:.3f} | "
                     f"{row['mean_tree_nodes']:.1f} | {row['mean_policy_nodes']:.1f} | {agreement} |")
    lines.extend(["", "## Interpretation", ""])
    if len(summary) > 1:
        first, last = summary[0], summary[-1]
        lines.append(
            f"From K={first['k']} to K={last['k']}, observed mean episode return changed from "
            f"{first['mean_return']:.2f} to {last['mean_return']:.2f}, mean planner wall time from "
            f"{first['mean_planning_wall_ms']:.3f} to {last['mean_planning_wall_ms']:.3f} ms, and "
            f"pairwise action agreement from {first['mean_pairwise_agreement']:.3f} to "
            f"{last['mean_pairwise_agreement']:.3f}. These are descriptive comparisons."
        )
        lines.extend(["", f"Mean search trials per decision fell from {first['mean_trials']:.1f} "
                      f"at K={first['k']} to {last['mean_trials']:.1f} at K={last['k']}. "
                      "This records the search effort available under this run's budget; "
                      "it must be considered when interpreting the K curve."])
    lines.extend([
        "",
        "More repeatable actions need not be better actions. A time budget can also limit how much of a larger "
        "scenario tree gets explored, so inspect the search trials and final bound gaps alongside K. "
        "Equal environment seeds are useful for comparison, but trajectories can diverge after different actions. "
        "Short, capped episodes measure truncated return and may miss later consequences.", "",
        "To decide how many scenarios are enough, expand the number of independent episode seeds and fixed "
        "beliefs, then check whether additional K yields a practically meaningful return or stability gain "
        "relative to its planning cost. The largest tested K is a finite benchmark, not a ground-truth policy.", "",
        "## Metrics and uncertainty", "",
        "- **Return:** total undiscounted reward per episode; `discounted_return` uses the run's discount. "
        "`mean_episode_steps` records actual episode length. Return intervals resample entire episodes.",
        "- **Paired return difference:** `mean_paired_return_delta_vs_max_k` is the mean of return(K) minus "
        "return(max K) at matching episode seeds. Its interval resamples seed pairs together. Negative values "
        "mean lower observed return than the largest K. The largest K's zero self-difference has no interval. "
        "A larger study can compare this difference against a prespecified acceptable reward loss.",
        "- **Planning:** `planning_wall_ms` is full planner wall time per decision in milliseconds. "
        "CPU and internal search times are separate columns in `summary.csv`. "
        "Timing, node counts, trials, gaps, and belief sizes are averaged within each episode first, "
        "then across episodes with equal episode weight, so long episodes do not dominate.",
        "- **Tree size:** `tree_nodes` counts explored belief nodes (VNodes). `policy_nodes` counts action nodes in "
        "the stock planner's greedy expanded policy. It does not count the implicit default policy, and "
        "internal default-policy choices can differ from this counted greedy policy; it is not the full "
        "executed policy size. `expanded_nodes` and `trials` describe search work. `final_gap` is the reported final "
        "upper-minus-lower bound gap in value units.",
        "- **Stability:** repeat planning from the same saved belief with fresh planner seeds. With R repeats "
        "and action counts nₐ, pairwise agreement is Σₐ nₐ(nₐ−1) / [R(R−1)]. It estimates the chance that "
        "two independent repeats select the same action; 1 means perfect agreement. "
        "Modal share is maxₐ nₐ/R; entropy is −Σₐ pₐ log₂ pₐ in bits.",
        "- **Finite benchmark:** `agreement_to_max_k_mode` is the fraction of a K group's repeats matching "
        "the modal action at the largest tested K for the same belief. Ties in the benchmark mode select "
        "the smallest numeric action. The largest K's own benchmark agreement is deliberately blank "
        "to avoid self-reference; this metric is not decision accuracy.",
        "- **Seed clusters:** for each stability metric, average beliefs within each seed, then average seeds "
        "equally. Bootstrap whole seed clusters, keeping all beliefs from a seed together. "
        "Repeated plans and beliefs from the same trajectory are not treated as independent episodes.",
        f"- **Intervals:** 95% percentile bootstrap intervals use {bootstrap:,} draws and deterministic analysis "
        "seed 81723. Fewer than two independent episodes/seeds cannot estimate uncertainty: the CI is blank "
        "in CSV and marked unavailable here. A few seeds produce coarse, unreliable intervals; an observed "
        "zero-width interval can also reflect identical pilot outcomes and is not proof of certainty.",
        "",
        "## Artifacts", "",
        "- [Per-K summary](summary.csv)",
        "- [Per-belief action stability](stability.csv)",
        "- [Raw episodes](episodes.csv), [raw decisions](steps.csv), [raw fixed-belief probes](probes.csv)",
        "- [Run manifest](manifest.json)",
        "- [Analysis provenance](analysis.json)",
    ])
    if plotted:
        lines.extend(["- [Overview PNG](overview.png), [overview SVG](overview.svg)", "", "![Experiment overview](overview.png)"])
    else:
        lines.extend(["", "Plots were skipped because matplotlib is not installed. Install it in an analysis "
                      "environment and rerun `analyze.py` to produce the overview PNG and SVG."])
    recorded = {key: manifest[key] for key in ("model", "lower_bound", "upper_bound", "sampling", "seed_scheme", "notes", "settings")
                if key in manifest}
    lines.extend(["", "## Recorded run settings", "", "Settings are shown below; the full manifest also records commands and source hashes.", "", "```json",
                  json.dumps(recorded, indent=2, sort_keys=True), "```", ""])
    (run_directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(run_directory: Path, bootstrap: int = 2000) -> list[dict]:
    if bootstrap < 1:
        raise InputError("--bootstrap must be at least 1")
    manifest, episodes, steps, probes = load_inputs(run_directory)
    stability = stability_rows(probes)
    summary = summarize(episodes, steps, stability, bootstrap)
    write_csv(run_directory / "summary.csv", summary)
    write_csv(run_directory / "stability.csv", stability)
    plotted = make_plots(run_directory, summary, manifest.get("settings", {}))
    analysis_metadata = {
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {name: hashlib.sha256((run_directory / name).read_bytes()).hexdigest()
                         for name in ("manifest.json", "episodes.csv", "steps.csv", "probes.csv")},
        "bootstrap_draws": bootstrap,
        "bootstrap_seed": 81723,
        "python": sys.version,
        "plots_generated": plotted,
    }
    (run_directory / "analysis.json").write_text(json.dumps(analysis_metadata, indent=2) + "\n")
    write_report(run_directory, manifest, summary, bootstrap, plotted)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples (default: 2000)")
    args = parser.parse_args()
    try:
        summary = analyze(args.run_directory, args.bootstrap)
    except (InputError, OSError) as exc:
        parser.exit(2, f"Analysis failed: {exc}\n")
    print(f"Analyzed {len(summary)} K values; wrote {args.run_directory / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
