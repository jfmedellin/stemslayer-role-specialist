"""Reading a rendered split, and cropping it into training windows.

A crop is taken at the same offset from the mixture and from both labels, so a
training example is always three views of one moment. Cropping them
independently would teach the model to invent material rather than separate it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

MANIFEST_NAME = "corpus.json"
SOURCES = ("lead_guitar", "rhythm_guitar")


@dataclass(frozen=True)
class Example:
    stem: str
    frames: int
    with_lead: bool
    centred_rhythm: bool


class RenderedCorpus:
    """A split rendered to disk, addressed by index."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        manifest_path = self.folder / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No corpus manifest at {manifest_path}. Build the split with "
                "python -m Tools.build_corpus first."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.split = manifest["split"]
        self.sample_rate = int(manifest["sample_rate"])
        self.examples = tuple(
            Example(
                stem=entry["stem"],
                frames=int(entry["frames"]),
                with_lead=bool(entry["with_lead"]),
                centred_rhythm=bool(entry["centred_rhythm"]),
            )
            for entry in manifest["examples"]
        )

    def __len__(self) -> int:
        return len(self.examples)

    def _read(self, stem: str, name: str, start: int, frames: int) -> np.ndarray:
        path = self.folder / f"{stem}-{name}.wav"
        block, _rate = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
        return np.asarray(block, dtype=np.float32)

    def crop(self, index: int, frames: int, *, offset: int | None = None, rng=None):
        """Return one aligned (mixture, lead, rhythm) window.

        The offset is shared: the labels must describe the same moment as the
        mixture, or the model learns to hallucinate rather than separate.
        """
        example = self.examples[index]
        if frames > example.frames:
            raise ValueError(
                f"Window of {frames} frames exceeds example {example.stem} of {example.frames}."
            )
        if offset is None:
            generator = rng if rng is not None else np.random.default_rng()
            offset = int(generator.integers(0, example.frames - frames + 1))
        views = tuple(self._read(example.stem, name, offset, frames) for name in ("mixture", *SOURCES))
        return views
