"""Run a training pass and report it in the terms admission uses.

    python -m Tools.train --epochs 20 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from RoleSpecialist.corpus.arrangement import TRAIN, VALIDATION
from RoleSpecialist.training.dataset import RenderedCorpus
from RoleSpecialist.training.train import fit, selection_key
from RoleSpecialist.vendor.role_metrics import RoleThresholds

VENDOR = Path(__file__).resolve().parent.parent / "RoleSpecialist" / "vendor"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="train", description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps-per-epoch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--silence-weight-per-db", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pt"))
    arguments = parser.parse_args(argv)

    train_corpus = RenderedCorpus(arguments.corpus / TRAIN)
    validation_corpus = RenderedCorpus(arguments.corpus / VALIDATION)
    thresholds = RoleThresholds.load(VENDOR / "thresholds.json")
    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"train {len(train_corpus)} examples | validation {len(validation_corpus)} | "
        f"device {arguments.device}"
    )
    if not thresholds.calibrated:
        # Saying this out loud every run keeps a good-looking number from being
        # mistaken for an admissible one.
        print(f"thresholds are UNCALIBRATED targets: {thresholds.calibration_blocker}")

    def report_epoch(epoch):
        summary = epoch.report
        print(
            f"epoch {epoch.index:3d}  loss {epoch.train_loss:.5f}"
            f" (silence {epoch.silence_loss:.5f})  "
            f"publishable {summary['publishable']:5.1%}  "
            f"recon {summary['median_reconstruction_db']:6.1f} dB  "
            f"leak {summary['worst_leakage_db']:6.1f} dB  "
            f"silent lane {summary['silent_lane_dbfs']:7.1f} dBFS  "
            f"absence {summary['absence_recalled']:.2f}  "
            f"({epoch.seconds:.0f}s)",
            flush=True,
        )

    history = fit(
        train_corpus,
        validation_corpus,
        thresholds,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        steps_per_epoch=arguments.steps_per_epoch,
        learning_rate=arguments.learning_rate,
        **(
            {}
            if arguments.silence_weight_per_db is None
            else {"silence_weight_per_db": arguments.silence_weight_per_db}
        ),
        device=arguments.device,
        checkpoint=arguments.checkpoint,
        on_epoch=report_epoch,
    )

    best = max(history, key=selection_key)
    print(
        f"\nbest epoch {best.index} at {best.report['publishable']:.1%} publishable "
        f"-> {arguments.checkpoint}"
    )
    print(json.dumps(best.report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
