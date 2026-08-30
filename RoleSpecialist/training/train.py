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

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from RoleSpecialist.training.dataset import RenderedCorpus
from RoleSpecialist.training.evaluate import score_example, summarise
from RoleSpecialist.training.loss import DEFAULT_WEIGHT_PER_DB, separation_loss
from RoleSpecialist.training.model import SOURCES, build_model
from RoleSpecialist.publish import resolve_absence
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
    absence_candidate_dbfs: float = -40.0,
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
        loss, _reconstruction, _silence = separation_loss(
            predicted, targets, candidate_dbfs=absence_candidate_dbfs
        )
        loss.backward()
        optimiser.step()
        history.append(float(loss.item()))
    return history


@dataclass(frozen=True)
class Epoch:
    """What one pass produced, in the terms a decision is made in."""

    index: int
    train_loss: float
    seconds: float
    report: dict
    silence_loss: float = 0.0


def selection_key(epoch: "Epoch") -> tuple[float, float]:
    """Rank epochs by publishability, breaking ties on reconstruction.

    Publishability alone cannot rank an early project, where every epoch scores
    zero and no later one ever beats the first. Selecting on it unqualified
    keeps the least trained model of the run and reports it as the best, which
    is how a night of training quietly produces nothing.

    Reconstruction is the tiebreaker because it is the gate the other two
    depend on: a decomposition that has lost energy cannot be rescued by
    lowering leakage.
    """
    reconstruction = epoch.report.get("median_reconstruction_db", float("-inf"))
    if not np.isfinite(reconstruction):
        # Infinite reconstruction is a degenerate estimate, not a perfect one.
        reconstruction = float("-inf") if reconstruction < 0 else 0.0
    return (epoch.report.get("publishable", 0.0), reconstruction)


def fit(
    train_corpus: RenderedCorpus,
    validation_corpus: RenderedCorpus,
    thresholds: RoleThresholds,
    *,
    epochs: int = 20,
    batch_size: int = 4,
    steps_per_epoch: int = 64,
    learning_rate: float = 3e-4,
    silence_weight_per_db: float = DEFAULT_WEIGHT_PER_DB,
    device: str = "cpu",
    checkpoint: Path | None = None,
    on_epoch=None,
) -> list[Epoch]:
    """Train, and report each epoch against the gates rather than the loss.

    The best epoch is chosen by how many validation results would be
    publishable, not by the lowest loss. Those are different questions, and
    only one of them is what a checkpoint has to answer.
    """
    import torch

    frames = int(WINDOW_SECONDS * train_corpus.sample_rate)
    rng = np.random.default_rng(0)
    model = build_model(sample_rate=train_corpus.sample_rate).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[Epoch] = []
    best: tuple[float, float] | None = None
    for index in range(epochs):
        started = time.monotonic()
        model.train()
        losses, silences = [], []
        for _step in range(steps_per_epoch):
            picks = rng.integers(0, len(train_corpus), size=batch_size)
            batch = draw_batch(train_corpus, picks, frames, rng)
            optimiser.zero_grad(set_to_none=True)
            loss, reconstruction, silence = separation_loss(
                model(torch.from_numpy(batch.mixture).to(device)),
                torch.from_numpy(batch.targets).to(device),
                candidate_dbfs=thresholds.audibility_minimum_dbfs,
                weight_per_db=silence_weight_per_db,
            )
            loss.backward()
            optimiser.step()
            losses.append(float(loss.item()))
            silences.append(silence)

        report = evaluate(model, validation_corpus, thresholds, limit=32, device=device)
        epoch = Epoch(
            index=index,
            train_loss=float(np.mean(losses)),
            seconds=time.monotonic() - started,
            report=report,
            silence_loss=float(np.mean(silences)),
        )
        history.append(epoch)
        if on_epoch is not None:
            on_epoch(epoch)
        # Keep the epoch that produced the most publishable results, because a
        # lower loss that publishes nothing is not progress.
        ranked = selection_key(epoch)
        if checkpoint is not None and (best is None or ranked > best):
            best = ranked
            torch.save(
                {
                    "sources": list(SOURCES),
                    "sample_rate": train_corpus.sample_rate,
                    "epoch": index,
                    "report": report,
                    "state_dict": model.state_dict(),
                },
                checkpoint,
            )
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
            # Score what would be published, not the raw tensor. Absence is a
            # publication decision, so a report that skips it measures a
            # pipeline nobody runs.
            published = resolve_absence(
                batch.mixture[0].T,
                estimate[SOURCES.index("lead_guitar")].T,
                estimate[SOURCES.index("rhythm_guitar")].T,
                thresholds,
            )
            scores.append(
                score_example(
                    batch.mixture[0].T,
                    published.lead,
                    published.rhythm,
                    thresholds,
                    lead_absent_expected=batch.lead_absent[0],
                )
            )
    return summarise(scores, thresholds)
