"""Render a small pilot corpus you can actually listen to.

The point is to find out whether synthesised guitar is realistic enough before
spending days of GPU time on it. A checkpoint trained on material that does not
resemble a metal recording will not transfer to one, and no metric in the
admission contract measures whether something sounds like a guitar.

Every example is written with its measured gates beside it, so what you hear and
what the contract measures are read together rather than one standing in for the
other.

    python -m Tools.render_pilot [--seconds 20] [--out data/pilot]
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from RoleSpecialist.corpus.mixture import LabelledMixture, render
from RoleSpecialist.vendor.role_metrics import (
    cross_role_energy_ratio_db,
    peak_dbfs,
    signal_to_residual_db,
)

# The corpus is training truth, not a published result. Writing it at 16 bits
# would bake in a quantisation floor that Stemslayer only has because Demucs
# wrote its stems that way.
SUBTYPE = "FLOAT"


@dataclass(frozen=True)
class Case:
    name: str
    seed: int
    why: str
    with_lead: bool = True
    centred_rhythm: bool = False


CASES = (
    Case(
        "01-lead-over-doubled-rhythm",
        seed=101,
        why="The ordinary arrangement: a centred lead over two panned rhythm takes.",
    ),
    Case(
        "02-rhythm-only",
        seed=202,
        why="No lead at all. The contract calls a silent lead lane a valid result, "
        "so the corpus has to contain tracks that produce one.",
        with_lead=False,
    ),
    Case(
        "03-centred-rhythm",
        seed=303,
        why="Rhythm placed in the centre on purpose, breaking the stereo convention "
        "the deterministic profile relies on. A specialist that learned position "
        "instead of role fails here.",
        centred_rhythm=True,
    ),
)


def measure(example: LabelledMixture) -> dict:
    lead, rhythm = example.lead, example.rhythm
    return {
        "lead_peak_dbfs": _finite(peak_dbfs(lead)),
        "rhythm_peak_dbfs": _finite(peak_dbfs(rhythm)),
        "lead_leakage_db": _finite(cross_role_energy_ratio_db(lead, rhythm)),
        "rhythm_leakage_db": _finite(cross_role_energy_ratio_db(rhythm, lead)),
        "reconstruction_db": _finite(
            signal_to_residual_db(example.guitar_family, lead + rhythm)
        ),
    }


def _finite(value: float) -> float | str:
    return value if np.isfinite(value) else ("-inf" if value < 0 else "inf")


def render_case(case: Case, seconds: float, root: Path) -> dict:
    example = render(
        seconds, seed=case.seed, with_lead=case.with_lead, centred_rhythm=case.centred_rhythm
    )
    folder = root / case.name
    folder.mkdir(parents=True, exist_ok=True)
    for name, samples in (
        ("guitar_family.wav", example.guitar_family),
        ("lead_guitar.wav", example.lead),
        ("rhythm_guitar.wav", example.rhythm),
    ):
        sf.write(folder / name, samples, example.sample_rate, subtype=SUBTYPE)

    record = {
        "name": case.name,
        "why": case.why,
        "seed": case.seed,
        "seconds": seconds,
        "with_lead": case.with_lead,
        "centred_rhythm": case.centred_rhythm,
        "sample_rate": example.sample_rate,
        "frames": example.frame_count,
        "measurements": measure(example),
    }
    (folder / "example.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="render-pilot", description=__doc__)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--out", type=Path, default=Path("data/pilot"))
    arguments = parser.parse_args(argv)

    records = [render_case(case, arguments.seconds, arguments.out) for case in CASES]
    (arguments.out / "pilot.json").write_text(
        json.dumps({"examples": records}, indent=2) + "\n", encoding="utf-8"
    )

    for record in records:
        measurements = record["measurements"]
        print(f"\n{record['name']}  ({record['seconds']:.0f}s, seed {record['seed']})")
        print(f"  {record['why']}")
        print(
            "  lead {lead_peak_dbfs} dBFS | rhythm {rhythm_peak_dbfs} dBFS | "
            "leakage lead {lead_leakage_db} dB, rhythm {rhythm_leakage_db} dB".format(
                **{
                    key: (value if isinstance(value, str) else f"{value:.1f}")
                    for key, value in measurements.items()
                }
            )
        )
    print(f"\nwrote {len(records)} examples to {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
