"""The training loop, and the sanity check that has to pass before it matters.

A harness that cannot drive the loss to near zero on a handful of examples is
broken, and no amount of corpus or compute will rescue it. `overfit` runs that
check in seconds: it is the cheapest evidence that data, model, loss and
optimiser are wired to each other correctly.

Validation reports what the admission contract measures, not loss. A run whose
loss falls while its results stay unpublishable has learned something that
cannot ship.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from RoleSpecialist.training.dataset import RenderedCorpus
from RoleSpecialist.training.evaluate import score_example, summarise
from RoleSpecialist.training.model import SOURCES, build_model
from RoleSpecialist.vendor.role_metrics import RoleThresholds

# HTDemucs works on windows, not whole songs. Four seconds covers several bars
# at the tempos the corpus renders, which is enough context for a riff to be a
# riff rather than a chord.
WINDOW_SECONDS = 4.0


@dataclass(frozen=True)
class Batch:
    mixture: np.ndarray  # (batch, channels, frames)
    targets: np.ndarray  # (batch, source, channels, frames)
    lead_absent: tuple[bool, ...]


def draw_batch(corpus: RenderedCorpus, indices, frames: int, rng) -> Batch:
    """Return one aligned batch, cropped identically across mixture and labels."""
    mixtures, targets, absent = [], [], []
    for index in indices:
        mixture, lead, rhythm = corpus.crop(index, frames, rng=rng)
        # Audio arrives frames-by-channels and the model wants channels-first.
        mixtures.append(mixture.T)
        targets.append(np.stack([lead.T, rhythm.T]))
        absent.append(not corpus.examples[index].with_lead)
    return Batch(
        mixture=np.stack(mixtures),
        targets=np.stack(targets),
        lead_absent=tuple(absent),
    )


def overfit(
    corpus: RenderedCorpus,
    *,
    steps: int = 400,
    size: int = 2,
    learning_rate: float = 3e-3,
    device: str = "cpu",
) -> list[float]:
    """Drive one fixed batch toward zero loss, and report whether it moved.

    This proves the harness works, never that the model is good. A run that
    cannot overfit two examples is misconfigured, and finding that out here
    costs a minute rather than a night of GPU time.

    The defaults are chosen so the check is conclusive. Measured on the corpus
    this repository renders, 400 steps at 3e-3 take an L1 loss from about 0.15
    to about 0.0017. Sixty steps only reached 0.18, which looks like a broken
    harness and is only an impatient one; a sanity check that cannot tell those
    apart is worse than none.
    """
    import torch

    frames = int(WINDOW_SECONDS * corpus.sample_rate)
    rng = np.random.default_rng(0)
    batch = draw_batch(corpus, range(min(size, len(corpus))), frames, rng)

    model = build_model(sample_rate=corpus.sample_rate).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mixture = torch.from_numpy(batch.mixture).to(device)
    targets = torch.from_numpy(batch.targets).to(device)

    history = []
    model.train()
    for _step in range(steps):
        optimiser.zero_grad(set_to_none=True)
        predicted = model(mixture)
        loss = torch.nn.functional.l1_loss(predicted, targets)
        loss.backward()
        optimiser.step()
        history.append(float(loss.item()))
    return history


def evaluate(model, corpus: RenderedCorpus, thresholds: RoleThresholds, *, limit: int = 32, device: str = "cpu"):
    """Score a model against the gates a checkpoint has to clear."""
    import torch

    frames = int(WINDOW_SECONDS * corpus.sample_rate)
    rng = np.random.default_rng(1)
    scores = []
    model.eval()
    with torch.no_grad():
        for index in range(min(limit, len(corpus))):
            batch = draw_batch(corpus, [index], frames, rng)
            predicted = model(torch.from_numpy(batch.mixture).to(device))
            estimate = predicted[0].cpu().numpy()
            scores.append(
                score_example(
                    batch.mixture[0].T,
                    estimate[SOURCES.index("lead_guitar")].T,
                    estimate[SOURCES.index("rhythm_guitar")].T,
                    thresholds,
                    lead_absent_expected=batch.lead_absent[0],
                )
            )
    return summarise(scores, thresholds)
