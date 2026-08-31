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
|-- publish.py           # Absence decided by reconstruction, at publication time
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

### Absence is a publication decision, not something a network learns

Four runs went into this, and three of the diagnoses along the way were wrong.
Each was corrected by a measurement rather than by another run.

The symptom was that absence recall stayed at exactly zero. Two hundred epochs
took reconstruction from 4.4 dB to 32.2 dB and reached 75% publishable while
every track with no lead failed and every track with one passed: the ceiling
was not the model's quality, it was the fraction of the corpus that has a solo.

**It was not undertraining.** Reconstruction crossed its gate and absence recall
never moved.

**It was not a missing loss term, exactly.** A term aimed at the contract's
-80 dBFS absence floor moved a silent lane from -23 dBFS to -51 dBFS and cost
nine decibels of reconstruction, taking the run to nothing publishable. A
shared network cannot push one lane thirty decibels further down without
biasing that lane everywhere else.

**The floor was the wrong target.** It is a classification threshold, not a
level a network has to reach. A lane only needs to fall below the audibility
minimum to become a *candidate*, and reconstruction decides the rest. That is
forty decibels less work.

**And the term was measuring the wrong thing.** Aimed at the audibility
minimum it reported itself satisfied from epoch 49 while the level that decides
publication drifted from -26 dBFS to -13 dBFS. It optimised RMS and was judged
on peak, and residue in a lane that should be silent is impulsive:

| Estimator | Level | Distance from the peak |
| --- | --- | --- |
| RMS | -55.3 dB | 30.4 dB |
| Mean of loudest 1% | -41.6 dB | 16.7 dB |
| Mean of loudest 0.1% | -32.3 dB | 7.4 dB |
| Mean of loudest 0.01% | -27.8 dB | 2.9 dB |
| Peak | -24.9 dBFS | what the gate reads |

The earlier floor-aimed run had only worked because aiming forty decibels too
low dragged the peak down as a side effect. The margin was absorbing the error,
not the reasoning being right.

With the term aimed at the audibility minimum and measuring the loudest
fraction, absence recall reached 1.00: six of six absences found, no false
calls in either direction, and the published lane is exact digital silence
because `publish.py` zeroes it once removing it is shown to cost no energy.

Removing a genuinely absent lead leaves reconstruction at 33.3 dB. Removing a
real one collapses it from 31.0 dB to 2.9 dB. That margin is what makes the
decision safe to act on, and it lives in publication rather than in training
because the contract puts it there.

### The model learns the role, and cannot reconstruct real audio

The checkpoint trained entirely on Karplus-Strong plucks through a `tanh` found
the solo in a real Megadeth track. Lead energy rose at 2:16, held between 18%
and 45% through the solo, and fell to 0.1% at 2:48, against a solo a listener
independently placed at 2:20 to 2:47. Inside the solo it carried 30% of the
energy and outside it 13%.

That answers the question this corpus existed to ask. A model trained on
synthesis could have learned the generator; this one learned something about
what separates a foreground voice from an accompaniment, and it transfers.

Reconstruction does not. 36.8 dB on synthetic validation becomes 25.5 dB on the
real stem, below the 30 dB gate, and the audible symptom is energy landing
cleanly in neither lane.

### Articulation was not the missing realism

The obvious next move was to make the synthesis more like a real riff: palm
muting, held lead notes, per-example amplifier voicing, and four times the
corpus. Measured on the same track:

| Corpus | Synthetic publishable | Real reconstruction | Solo-to-rest ratio |
| --- | --- | --- | --- |
| 256 examples | 100% | 25.5 dB | 2.3x |
| 1024, enriched | 96.9% | 25.0 dB | 2.2x |

**Nothing moved.** Not the reconstruction, not the solo detection. The
enrichment was worth doing precisely because it rules out the cheap
explanation: if the gap were chugging or amplifier variety, this would have
closed some of it.

What remains is what a sixty-line physical model cannot produce. String and
fret interaction, pick noise, the harmonics of fretted notes. A real
amplification chain, where a `tanh` is not a valve and a one-pole filter is not
a cabinet. A room, a microphone, a mix. And one thing synthesis cannot address
at all: the guitar handed to the specialist has already been through
`htdemucs_6s`, so it carries that separation's artefacts, and a clean corpus
contains none.

Closing this gap means recorded DI with training rights negotiated, which is the
staged fallback this repository named on its first day. It now has evidence
behind it rather than an intuition.

## Where the runs stand

| Run | Reconstruction | Publishable | Absence recall |
| --- | --- | --- | --- |
| L1 only | 32.2 dB | 75% | 0 of 6 |
| Silence term aimed at the -80 dBFS floor | 23.3 dB | 0% | 6 of 6 |
| Aimed at audibility, measuring RMS | 30.3 dB | 41% | 0 of 6 |
| Aimed at audibility, measuring the peak | 27.9 dB | 19% | 6 of 6 |
| The same, run to 400 epochs | 38.1 dB | 100% | 6 of 6 |
| Enriched corpus of 1024, 400 epochs | 32.7 dB | 96.9% | all |

The fourth row was not a ceiling, only a halfway point: the same configuration
run twice as long cleared every gate on synthetic validation. The fifth scores
slightly lower on a corpus that is harder, and neither improves the real
recording.

## Status

Nothing here is admissible, and what blocks it is now known rather than
suspected.

A corpus renders and satisfies the gates its own ground truth must satisfy. A
harness learns from it. Absence is decided the way the contract specifies. A
checkpoint clears every gate on synthetic validation and finds the solo in a
real recording with the right edges.

It does not reconstruct that recording, and enriching the synthesis did not
help. The next step is not code: it is recorded guitar DI with rights that
permit training redistributable weights. The perceptual evidence the contract
requires has not been gathered either, and the thresholds remain
`"calibrated": false`.

