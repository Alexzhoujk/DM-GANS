from typing import Any

import pytest

from scripts.aggregate_paired_evaluations import aggregate_reports


def evaluation_report(seed: int, fid_delta: float, r_delta: float) -> dict[str, Any]:
    return {
        "status": "complete",
        "baseline_name": "control",
        "candidate_name": "CL2",
        "sample_count": 30000,
        "seed": seed,
        "protocol": {
            "baseline_generator_format": "modern-ema",
            "candidate_generator_format": "modern-ema",
        },
        "results": {
            "baseline": {
                "fid_pytorch": 16.0,
                "r_precision": {"overall_percent": 75.0},
            },
            "candidate": {
                "fid_pytorch": 16.0 + fid_delta,
                "r_precision": {"overall_percent": 75.0 + r_delta},
            },
            "delta_candidate_minus_baseline": {
                "fid_pytorch": fid_delta,
                "r_precision_percentage_points": r_delta,
            },
        },
        "paired_r_precision": {
            "cluster_bootstrap_delta_percentage_points_95_ci": [-0.2, 0.4]
        },
        "decision": {
            "verdict": "supported",
            "r_precision_noninferiority_margin_percentage_points": -1.0,
        },
        "_source_report": f"/tmp/seed-{seed}/report.json",
    }


def test_aggregate_reports_summarizes_independent_seeds() -> None:
    aggregate = aggregate_reports(
        [
            evaluation_report(1, -0.2, 0.1),
            evaluation_report(2, -0.4, 0.2),
            evaluation_report(3, -0.6, 0.3),
        ]
    )

    assert aggregate["decision"]["verdict"] == "supported across seeds"
    assert aggregate["aggregate"]["fid_delta"]["mean"] == pytest.approx(-0.4)
    assert aggregate["aggregate"]["r_delta_percentage_points"]["sample_std"] == pytest.approx(
        0.1
    )


def test_aggregate_reports_rejects_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="unique"):
        aggregate_reports(
            [evaluation_report(1, -0.2, 0.1), evaluation_report(1, -0.3, 0.2)]
        )


def test_aggregate_reports_rejects_mixed_protocols() -> None:
    first = evaluation_report(1, -0.2, 0.1)
    second = evaluation_report(2, -0.3, 0.2)
    second["protocol"]["candidate_generator_format"] = "modern-raw"

    with pytest.raises(ValueError, match="same non-random evaluation protocol"):
        aggregate_reports([first, second])


def test_aggregate_reports_rejects_inconsistent_deltas() -> None:
    report = evaluation_report(1, -0.2, 0.1)
    report["results"]["delta_candidate_minus_baseline"]["fid_pytorch"] = -9.0

    with pytest.raises(ValueError, match="FID delta is inconsistent"):
        aggregate_reports([report])


def test_aggregate_reports_rejects_stale_supported_verdict() -> None:
    report = evaluation_report(1, 0.2, 0.1)

    with pytest.raises(ValueError, match="verdict is inconsistent"):
        aggregate_reports([report])


def test_aggregate_reports_rejects_reused_generator_checkpoint() -> None:
    first = evaluation_report(1, -0.2, 0.1)
    second = evaluation_report(2, -0.3, 0.2)
    for report in (first, second):
        report["checkpoints"] = {
            "baseline_generator": {"sha256": "same-baseline"},
            "candidate_generator": {"sha256": "same-candidate"},
        }

    with pytest.raises(ValueError, match="distinct baseline_generator"):
        aggregate_reports([first, second])


def test_aggregate_reports_rejects_nonfinite_metrics() -> None:
    report = evaluation_report(1, -0.2, 0.1)
    report["results"]["candidate"]["fid_pytorch"] = float("nan")

    with pytest.raises(ValueError, match="must be finite"):
        aggregate_reports([report])
