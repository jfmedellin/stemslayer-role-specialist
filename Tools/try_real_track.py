"""Run a checkpoint on a real recording, which is the test that matters.

Every number this project has produced so far was measured on audio its own
generator made. A model can score perfectly on that by learning the generator
rather than the role, and no gate in the contract can tell the difference.

This takes the guitar family htdemucs_6s isolated from an actual track and
reports the same gates. A large drop against the synthetic figures is the
expected result, and it says the corpus needs realism rather than the
architecture needing changes.

    python -m Tools.try_real_track data/real/htdemucs_6s/<track>/guitar.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from RoleSpecialist.publish import resolve_absence
from RoleSpecialist.training.model import SOURCES, build_model
from RoleSpecialist.training.train import WINDOW_SECONDS
from RoleSpecialist.vendor.role_metrics import (
    RoleThresholds,
    cross_role_energy_ratio_db,
    peak_dbfs,
    signal_to_residual_db,
)

VENDOR = Path(__file__).resolve().parent.parent / "RoleSpecialist" / "vendor"


def separate(model, guitar: np.ndarray, sample_rate: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Run the model window by window and stitch the result.

    Windows are the length the model trained on and are laid end to end with a
    crossfade, because a boundary discontinuity would show up as lost energy
    and be indistinguishable from a separation failure.
    """
    frames = int(WINDOW_SECONDS * sample_rate)
    fade = frames // 8
    lead = np.zeros_like(guitar)
    rhythm = np.zeros_like(guitar)
    weight = np.zeros((guitar.shape[0], 1), dtype=np.float64)
    ramp = np.concatenate(
        [np.linspace(0, 1, fade), np.ones(frames - 2 * fade), np.linspace(1, 0, fade)]
    ).reshape(-1, 1)

    model.eval()
    with torch.no_grad():
        for start in range(0, max(1, guitar.shape[0] - fade), frames - fade):
            block = guitar[start : start + frames]
            if block.shape[0] < frames:
                block = np.pad(block, ((0, frames - block.shape[0]), (0, 0)))
            estimate = model(torch.from_numpy(block.T[None]).to(device))[0].cpu().numpy()
            span = min(frames, guitar.shape[0] - start)
            lead[start : start + span] += estimate[SOURCES.index("lead_guitar")].T[:span] * ramp[:span]
            rhythm[start : start + span] += estimate[SOURCES.index("rhythm_guitar")].T[:span] * ramp[:span]
            weight[start : start + span] += ramp[:span]

    weight[weight == 0] = 1.0
    return (lead / weight).astype(np.float32), (rhythm / weight).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="try-real-track", description=__doc__)
    parser.add_argument("guitar", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/peak-aimed-400.pt"))
    parser.add_argument("--out", type=Path, default=Path("data/real/roles"))
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args(argv)

    guitar, sample_rate = sf.read(arguments.guitar, dtype="float32", always_2d=True)
    thresholds = RoleThresholds.load(VENDOR / "thresholds.json")
    saved = torch.load(arguments.checkpoint, weights_only=False)
    model = build_model(sample_rate=sample_rate).to(arguments.device)
    model.load_state_dict(saved["state_dict"])

    lead, rhythm = separate(model, guitar, sample_rate, arguments.device)
    published = resolve_absence(guitar, lead, rhythm, thresholds)

    arguments.out.mkdir(parents=True, exist_ok=True)
    sf.write(arguments.out / "lead_guitar.wav", published.lead, sample_rate, subtype="FLOAT")
    sf.write(arguments.out / "rhythm_guitar.wav", published.rhythm, sample_rate, subtype="FLOAT")

    reconstruction = signal_to_residual_db(guitar, published.lead + published.rhythm)
    worst_leakage = max(
        cross_role_energy_ratio_db(published.lead, published.rhythm),
        cross_role_energy_ratio_db(published.rhythm, published.lead),
    )
    print(f"checkpoint epoch {saved['epoch']} on {arguments.guitar.name}")
    print(f"  duration        {guitar.shape[0] / sample_rate / 60:.1f} min")
    print(f"  reconstruction  {reconstruction:6.1f} dB   (gate {thresholds.reconstruction_minimum_db:.0f})")
    print(f"  worst leakage   {worst_leakage:6.1f} dB   (gate {thresholds.leakage_maximum_db:.0f})")
    print(f"  lead peak       {peak_dbfs(published.lead):6.1f} dBFS")
    print(f"  rhythm peak     {peak_dbfs(published.rhythm):6.1f} dBFS")
    print(f"  lead published as absent: {published.lead_absent}")
    print(f"  -> {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
