# Stemslayer Role Specialist

Training ground for the Lead/Rhythm guitar specialist that
[Stemslayer](https://github.com/jfmedellin/separador-pistas) cannot ship.

Stemslayer registers a **Metal** profile with separate lead guitar and rhythm
guitar lanes, and that profile is **disabled**, because no redistributable model
decomposes guitar into semantic roles offline. Its
[admission contract](https://github.com/jfmedellin/separador-pistas/blob/master/Compliance/evidence/metal-guitar/CONTRACT.md)
records why every public candidate was rejected and what a future one must
satisfy. That contract ends with the sentence this repository exists to honour:

> Training happens in a separate repository with its own runtime. Only the
> resulting checkpoint, its evidence, and its registration enter this project.

Stemslayer ships a **Metal Stereo** profile instead, which splits the isolated
guitar by stereo position. It is useful and it is honest about being position
rather than role. This repository is the attempt at the real thing.

## The problem is the corpus, not the architecture

The architecture is settled: a role-aware two-output HTDemucs-family checkpoint
applied to the guitar family that `htdemucs_6s` already isolates.

What does not exist is training data. Public metal multitracks with lead and
rhythm labels and usable rights are not available, and a model trained on
material whose rights do not permit redistributing the resulting weights is
rejected by the contract regardless of how well it scores.

So the corpus is synthetic and rights-owned. Guitar signals are rendered, driven
through amplifier simulation, and mixed. Role labels are then **exact by
construction**: nothing is annotated, because the lead and rhythm stems are the
sources that were rendered, and no source ever carries both roles.

### Why synthesis rather than recorded DI

A recorded DI library or a sampled virtual instrument would sound better. Neither
states whether its licence permits training weights that are then redistributed,
and unresolved rights are exactly what the contract rejects. Every sample here
comes from an algorithm, so the rights question does not arise.

This is a deliberate trade of realism for certainty, and it is staged. If the
pilot shows the architecture works and realism is the remaining gap, real DI can
be recorded or licensed with rights negotiated for training, and the rendering
pipeline does not change: only the source of the dry signal does.

## The shared measurements are copied, never rewritten

`RoleSpecialist/vendor/` holds `role_metrics.py` and `thresholds.json`, copied
verbatim from a tagged Stemslayer release. They define what *absent*, *audible*
and *reconstructed* mean.

A second definition would be worse than no definition: a checkpoint could
evaluate clean here and be rejected by admission there, or be admitted on
measurements the app never agreed to. `Tests/test_vendored_metrics.py` fails if
a vendored file stops matching the digest recorded in `PROVENANCE.json`, and
`Tools/refresh_vendored_metrics.py` re-copies them from a newer tag.

## What is here so far

Corpus synthesis and a training harness that has run. **No checkpoint is
admissible, and none is close.**

```text
RoleSpecialist/
|-- corpus/
|   |-- strings.py       # Karplus-Strong plucked string, seeded and reproducible
|   |-- amp.py           # Deterministic saturation and cabinet voicing
|   |-- performance.py   # Rhythm riffs and lead lines, sharing no note event
|   |-- arrangement.py   # What each example varies, and the disjoint splits
|   `-- mixture.py       # One example whose role labels are exact by construction
|-- training/
|   |-- model.py         # Two-output HTDemucs from the reference implementation
|   |-- dataset.py       # A rendered split, cropped into aligned windows
|   |-- evaluate.py      # Scoring against the contract gates, not against loss
|   `-- train.py         # The loop, and the overfit check that precedes trusting it
`-- vendor/              # role_metrics.py and thresholds.json, copied from a tag
```

### Running it

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe -m Tools.build_corpus --split train --count 256 --seconds 8
.\.venv\Scripts\python.exe -m Tools.build_corpus --split validation --count 64 --seconds 8
.\.venv\Scripts\python.exe -m Tools.train --epochs 200 --device cuda

.\.venv\Scripts\python.exe -m unittest discover -s Tests -t . -v
```

A corpus is regenerated from its manifest rather than archived: every example is
reproducible from its split and its index, so nothing large needs keeping.

### What a training run reports

Loss says how close two waveforms are. It says nothing about whether a result is
publishable, so every epoch is scored with the vendored measurements instead:

```text
epoch  19  loss 0.07173  publishable 0.0%  recon 20.7 dB  worst leakage -10.2 dB  absence 0.00
```

The checkpoint kept is the one that publishes most, with reconstruction breaking
ties. A run whose loss falls while its results stay unpublishable has learned
something that cannot ship, and selecting on loss would hide that.

## Findings so far

Recorded because they are evidence toward calibrating thresholds the contract
still marks `"calibrated": false`, and because each was invisible until measured.

### The leakage gate is permissive

| Corpus | Lead-versus-rhythm leakage | Gate at -12 dB |
| --- | --- | --- |
| Rendered honestly | -48.7 dB | passes with 36 dB of margin |
| Lead is the riff again | -0.0 dB | rejected |
| 30% of the rhythm bled into the lead | -12.4 dB | **passes** |

A corpus can bleed 30% of one role into the other and clear the gate. The tests
therefore assert real margin rather than a bare pass, so synthesis that quietly
muddied the labels shows up as a collapsing margin instead of a still-passing
gate.

### Selecting a checkpoint on publishability alone keeps the worst model

The first real run took reconstruction from 4.6 dB to 20.7 dB across twenty
epochs and saved epoch zero. Nothing publishes yet, so every epoch tied at zero
and none beat the first. That is the entire early life of this project, and it
would have returned the model from before it learned anything after every run,
including an overnight one. Reconstruction now breaks the tie.

### Absent leads are 70 dB away from counting as absent

On validation tracks whose true lead is digital silence, the trained model
predicts a lead lane at about -10 dBFS. The contract counts a lane as absent at
or below -80 dBFS.

Two explanations fit: too little training, or an L1 waveform loss that pushes
toward an average and never toward digital silence. If it is the second, no
track without a solo can ever score as correct, and the loss needs a term that
rewards silence when the target is silent. Distinguishing them is the next
measurement, not the next assumption.

## Status

Nothing here is admissible. A corpus renders and satisfies the gates its own
ground truth must satisfy, a harness learns from it, and a checkpoint reaches
about 20 dB of reconstruction against a 30 dB gate with nothing publishable.
Evaluation on real metal recordings and the perceptual evidence the contract
requires are both still ahead.

