"""Render a split to disk once, so training does not re-synthesise every epoch.

Synthesis costs about a fifth of real time, which is fine for a pilot and
ruinous inside a training loop: a batch would take longer to render than to
learn from. Examples are therefore rendered once and read back.

Only the manifest is worth keeping. Every example is reproducible from its
split and index, so a corpus is regenerated rather than archived, and a
checkpoint's training data can be rebuilt from the manifest it was trained on.

    python -m Tools.build_corpus --split train --count 256 --seconds 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf

from RoleSpecialist.corpus.arrangement import TRAIN, VALIDATION, sample
from RoleSpecialist.corpus.mixture import render_arrangement

SUBTYPE = "FLOAT"
MANIFEST_NAME = "corpus.json"


def build(split: str, count: int, seconds: float, root: Path) -> dict:
    folder = root / split
    folder.mkdir(parents=True, exist_ok=True)
    examples = []
    for index in range(count):
        arrangement = sample(split, index)
        example = render_arrangement(arrangement, seconds)
        stem = f"{index:05d}"
        for name, samples in (
            ("mixture", example.guitar_family),
            ("lead_guitar", example.lead),
            ("rhythm_guitar", example.rhythm),
        ):
            sf.write(folder / f"{stem}-{name}.wav", samples, example.sample_rate, subtype=SUBTYPE)
        examples.append(
            {
                "index": index,
                "stem": stem,
                "seed": arrangement.seed,
                "frames": example.frame_count,
                "with_lead": arrangement.with_lead,
                "centred_rhythm": arrangement.centred_rhythm,
                "riff": list(arrangement.riff),
                "scale": list(arrangement.scale),
                "tempo_bpm": arrangement.tempo_bpm,
                "root_hz": arrangement.root_hz,
            }
        )
        if (index + 1) % 25 == 0:
            print(f"  {split}: {index + 1}/{count}", flush=True)

    manifest = {
        "split": split,
        "count": count,
        "seconds": seconds,
        "sample_rate": 44_100,
        "subtype": SUBTYPE,
        "examples": examples,
    }
    (folder / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build-corpus", description=__doc__)
    parser.add_argument("--split", choices=(TRAIN, VALIDATION), default=TRAIN)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--out", type=Path, default=Path("data/corpus"))
    arguments = parser.parse_args(argv)

    manifest = build(arguments.split, arguments.count, arguments.seconds, arguments.out)
    without_lead = sum(1 for e in manifest["examples"] if not e["with_lead"])
    centred = sum(1 for e in manifest["examples"] if e["centred_rhythm"])
    print(
        f"{manifest['split']}: {manifest['count']} examples of {manifest['seconds']:.0f}s "
        f"({without_lead} with no lead, {centred} with centred rhythm) "
        f"-> {arguments.out / arguments.split}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
