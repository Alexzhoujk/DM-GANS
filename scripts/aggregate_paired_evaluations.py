"""Aggregate multiple seeded paired checkpoint evaluation reports."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def mean_and_sample_std(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "sample_std": None, "minimum": None, "maximum": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "minimum": min(values),
        "maximum": max(values),
    }


def required_float(report: dict[str, Any], *keys: str) -> float:
    value: Any = report
    for key in keys:
        value = value[key]
    if value is None:
        raise ValueError(f"Required metric {'/'.join(keys)} was skipped")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Required metric {'/'.join(keys)} must be finite")
    return number


def load_reports(paths: list[Path]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "complete":
            raise ValueError(f"Report is not complete: {path}")
        report["_source_report"] = str(path.resolve())
        reports.append(report)
    return reports


def comparison_protocol(report: dict[str, Any]) -> dict[str, Any]:
    """Return non-random protocol fields that must match across runs."""

    protocol = report.get("protocol", {})
    protocol_fields = (
        "split",
        "resolution",
        "caption_words",
        "generation_batch_size",
        "caption_bank_batch_size",
        "bootstrap_resamples",
        "pairing",
        "baseline_conditioning_mode",
        "candidate_conditioning_mode",
        "baseline_generator_format",
        "candidate_generator_format",
        "candidate_conditioner_is_evaluator",
        "r_precision",
        "fid",
        "is",
    )
    checkpoints = report.get("checkpoints", {})
    fixed_checkpoint_fields = (
        "baseline_conditioning_text_encoder",
        "candidate_conditioning_text_encoder",
        "evaluation_text_encoder",
        "evaluation_image_encoder",
        "fid_real_statistics",
    )
    checkpoint_hashes = {
        name: checkpoints[name].get("sha256")
        for name in fixed_checkpoint_fields
        if isinstance(checkpoints.get(name), dict)
    }
    return {
        "sample_count": int(report["sample_count"]),
        "protocol": {name: protocol.get(name) for name in protocol_fields if name in protocol},
        "fixed_checkpoint_sha256": checkpoint_hashes,
    }


def validate_report_arithmetic(
    report: dict[str, Any],
    *,
    baseline_fid: float,
    candidate_fid: float,
    fid_delta: float,
    baseline_r: float,
    candidate_r: float,
    r_delta: float,
) -> None:
    """Reject stale or hand-edited deltas before they reach an aggregate claim."""

    expected = {
        "FID": candidate_fid - baseline_fid,
        "R-precision": candidate_r - baseline_r,
    }
    reported = {"FID": fid_delta, "R-precision": r_delta}
    for name, expected_value in expected.items():
        if not math.isclose(reported[name], expected_value, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(f"Reported {name} delta is inconsistent with method values")


def validated_supported_verdict(report: dict[str, Any], fid_delta: float) -> bool:
    """Recompute the fixed per-run rule instead of trusting a verdict string."""

    ci = report["paired_r_precision"][
        "cluster_bootstrap_delta_percentage_points_95_ci"
    ]
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError("Paired R-precision confidence interval must contain two endpoints")
    ci_lower = float(ci[0])
    margin = float(
        report["decision"].get("r_precision_noninferiority_margin_percentage_points", -1.0)
    )
    if not math.isfinite(ci_lower) or not math.isfinite(margin):
        raise ValueError("R-precision confidence bound and margin must be finite")
    expected = int(report["sample_count"]) >= 30000 and fid_delta < 0.0 and ci_lower > margin
    declared = report["decision"]["verdict"] == "supported"
    if declared != expected:
        raise ValueError("Individual verdict is inconsistent with the fixed success rule")
    return expected


def aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("At least one report is required")
    baseline_name = reports[0].get("baseline_name", "DM-GAN")
    candidate_name = reports[0]["candidate_name"]
    comparison_names = {
        (report.get("baseline_name", "DM-GAN"), report["candidate_name"]) for report in reports
    }
    if len(comparison_names) != 1:
        raise ValueError("All reports must compare the same named methods")
    protocols = [comparison_protocol(report) for report in reports]
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("All reports must use the same non-random evaluation protocol")
    seeds = [int(report["seed"]) for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Each report must have a unique evaluation seed")
    distinct_checkpoints_verified = True
    for field in ("baseline_generator", "candidate_generator"):
        hashes = [
            report.get("checkpoints", {}).get(field, {}).get("sha256") for report in reports
        ]
        present_hashes = [value for value in hashes if isinstance(value, str)]
        if len(present_hashes) != len(reports):
            distinct_checkpoints_verified = False
        elif len(set(present_hashes)) != len(present_hashes):
            raise ValueError(f"Each run must use a distinct {field} checkpoint")

    runs: list[dict[str, Any]] = []
    metric_values: dict[str, list[float]] = {
        "baseline_fid": [],
        "candidate_fid": [],
        "fid_delta": [],
        "baseline_r_percent": [],
        "candidate_r_percent": [],
        "r_delta_percentage_points": [],
    }
    for report in sorted(reports, key=lambda item: int(item["seed"])):
        baseline_fid = required_float(report, "results", "baseline", "fid_pytorch")
        candidate_fid = required_float(report, "results", "candidate", "fid_pytorch")
        fid_delta = required_float(
            report, "results", "delta_candidate_minus_baseline", "fid_pytorch"
        )
        baseline_r = required_float(
            report, "results", "baseline", "r_precision", "overall_percent"
        )
        candidate_r = required_float(
            report, "results", "candidate", "r_precision", "overall_percent"
        )
        r_delta = required_float(
            report,
            "results",
            "delta_candidate_minus_baseline",
            "r_precision_percentage_points",
        )
        validate_report_arithmetic(
            report,
            baseline_fid=baseline_fid,
            candidate_fid=candidate_fid,
            fid_delta=fid_delta,
            baseline_r=baseline_r,
            candidate_r=candidate_r,
            r_delta=r_delta,
        )
        supported = validated_supported_verdict(report, fid_delta)
        metric_values["baseline_fid"].append(baseline_fid)
        metric_values["candidate_fid"].append(candidate_fid)
        metric_values["fid_delta"].append(fid_delta)
        metric_values["baseline_r_percent"].append(baseline_r)
        metric_values["candidate_r_percent"].append(candidate_r)
        metric_values["r_delta_percentage_points"].append(r_delta)
        runs.append(
            {
                "seed": int(report["seed"]),
                "sample_count": int(report["sample_count"]),
                "baseline_generator_format": report["protocol"].get(
                    "baseline_generator_format", "unreported"
                ),
                "candidate_generator_format": report["protocol"].get(
                    "candidate_generator_format", "unreported"
                ),
                "baseline_fid": baseline_fid,
                "candidate_fid": candidate_fid,
                "fid_delta": fid_delta,
                "baseline_r_percent": baseline_r,
                "candidate_r_percent": candidate_r,
                "r_delta_percentage_points": r_delta,
                "individual_verdict": report["decision"]["verdict"],
                "supported": supported,
                "source_report": report["_source_report"],
            }
        )

    all_formal = all(run["sample_count"] >= 30000 for run in runs)
    supported_count = sum(run["supported"] for run in runs)
    if not all_formal:
        verdict = "smoke only"
        summary = "At least one run has fewer than 30,000 samples; do not make a quality claim."
    elif supported_count == len(runs):
        verdict = "supported across seeds"
        summary = f"All {len(runs)} supplied seeded evaluations satisfy the fixed success rule."
    elif supported_count:
        verdict = "mixed across seeds"
        summary = f"Only {supported_count} of {len(runs)} supplied seeded evaluations is supported."
    else:
        verdict = "not supported across seeds"
        summary = "No supplied seeded evaluation satisfies the fixed success rule."

    return {
        "status": "complete",
        "baseline_name": baseline_name,
        "candidate_name": candidate_name,
        "run_count": len(runs),
        "seeds": sorted(seeds),
        "runs": runs,
        "aggregate": {
            name: mean_and_sample_std(values) for name, values in metric_values.items()
        },
        "decision": {
            "verdict": verdict,
            "summary": summary,
            "rule": (
                "Every seed must use at least 30,000 samples, lower candidate FID, and have a "
                "95% image-cluster bootstrap lower bound above -1 percentage point for the "
                "candidate-minus-baseline R-precision delta."
            ),
            "supported_run_count": supported_count,
            "distinct_generator_checkpoint_pairs_verified": distinct_checkpoints_verified,
        },
        "limitations": [
            "The report seed controls evaluation randomness. Unless training seed metadata is separately recorded and evaluation streams are held fixed, across-run SD conflates checkpoint/training variation with caption truncation, negatives, z, and CA sampling.",
            "With three supplied runs, across-run uncertainty remains imprecise; individual paired R-precision confidence intervals remain primary evidence.",
            "FID deltas are distribution-level differences and do not have a per-image paired confidence interval.",
        ],
    }


def format_mean_std(summary: dict[str, float | int | None], digits: int = 4) -> str:
    mean = summary["mean"]
    std = summary["sample_std"]
    if mean is None:
        return "missing"
    if std is None:
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f} ± {float(std):.{digits}f}"


def markdown_report(aggregate: dict[str, Any]) -> str:
    rows = "\n".join(
        "| {seed} | {sample_count:,} | {baseline_fid:.4f} | {candidate_fid:.4f} | "
        "{fid_delta:+.4f} | {baseline_r_percent:.2f}% | {candidate_r_percent:.2f}% | "
        "{r_delta_percentage_points:+.2f} pp | {individual_verdict} |".format(**run)
        for run in aggregate["runs"]
    )
    metrics = aggregate["aggregate"]
    limitations = "\n".join(f"- {item}" for item in aggregate["limitations"])
    return f"""# Multiple seeded paired checkpoint evaluations

## Conclusion

**{aggregate["decision"]["verdict"].upper()}** — {aggregate["decision"]["summary"]}

Comparison: **{aggregate["baseline_name"]}** vs **{aggregate["candidate_name"]}**.

## Per-seed results

| Seed | N | Baseline FID | Candidate FID | ΔFID | Baseline R | Candidate R | ΔR | Verdict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{rows}

## Across-seed summary (mean ± sample SD)

- Baseline FID: {format_mean_std(metrics["baseline_fid"])}
- Candidate FID: {format_mean_std(metrics["candidate_fid"])}
- Candidate-minus-baseline FID: {format_mean_std(metrics["fid_delta"])}
- Baseline R-precision: {format_mean_std(metrics["baseline_r_percent"], 2)}%
- Candidate R-precision: {format_mean_std(metrics["candidate_r_percent"], 2)}%
- Candidate-minus-baseline R-precision: {format_mean_std(metrics["r_delta_percentage_points"], 2)} pp

## Decision rule

{aggregate["decision"]["rule"]}

## Limitations

{limitations}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path, help="Per-seed report.json files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/final/contrastive_finetune_evaluation"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate = aggregate_reports(load_reports(args.reports))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "aggregate_report.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "AGGREGATE_EVALUATION.md").write_text(
        markdown_report(aggregate), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
