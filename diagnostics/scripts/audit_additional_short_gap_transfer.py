#!/usr/bin/env python3
"""Stress-test the temporal-shape rescue on an additional short-gap date.

This wrapper reuses the already validated original-V75 downstream reproduction
and object audit. It changes only the target, previous, and retrospective
follow-up dates. No DEM volume is computed and no production output is changed.
Run one date pair at a time so every result can be reviewed before continuing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "diagnostics/scripts"))
import audit_temporal_shape_cross_region_bridge as bridge  # noqa: E402


PAIRS = {
    "early2024": {
        "target": "2024-05-21",
        "previous": "2024-05-01",
        "follow_up": "2024-06-01",
    },
    "mid2025": {
        "target": "2025-06-01",
        "previous": "2025-05-21",
        "follow_up": "2025-08-01",
    },
}
REGIONS = ("pihe", "shihe", "hangbuhe")
OUTPUT_ROOT = ROOT / "diagnostics/results/additional_short_gap_transfer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pair", choices=tuple(PAIRS), help="date pair to test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dates = PAIRS[args.pair]

    # The imported audit deliberately exposes dates as module-level settings.
    # Replacing them here keeps the tested downstream logic identical.
    bridge.TARGET = dates["target"]
    bridge.PREVIOUS = dates["previous"]
    bridge.FUTURE = dates["follow_up"]

    output = OUTPUT_ROOT / args.pair / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "experiment": "Additional short-gap temporal-shape transfer stress test",
        "pair": args.pair,
        "target_date": dates["target"],
        "previous_date": dates["previous"],
        "follow_up_proxy_date": dates["follow_up"],
        "selection": (
            "previous support >=0.25, area 0.2-5 ha, "
            "bbox fill >=0.45, elongation <=3"
        ),
        "regions": {},
    }

    for region in REGIONS:
        print(f"Additional short-gap transfer: {args.pair} {region}", flush=True)
        report["regions"][region] = bridge.process_region(region, output)
        (output / "summary.partial.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    report["status"] = "complete"
    report["interpretation"] = [
        "A successful baseline reproduction confirms that the same saved model chain is being audited on the new target date.",
        "Candidate counts and areas test whether the rescue mechanism remains non-zero and bounded across dates and regions.",
        "Follow-up support is retrospective risk evidence only and never selects an object.",
        "This test evaluates mechanism transfer; without matched manual totals it does not claim trend or volume accuracy.",
    ]
    report["limits"] = [
        "The same loose morphology range is used as a stress-test reference, not as a frozen production parameter.",
        "Follow-up absence is not a false-water label because later observations may miss seasonal or cloud-affected water.",
        "No DEM volume, production mask, production config, or model parameter is changed.",
    ]
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"RESULT_FOLDER: {output}", flush=True)
    print("STATUS: complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
