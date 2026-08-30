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

Corpus synthesis and the tests that hold it to the admission gates. **No
training has run and no checkpoint exists.**

```text
RoleSpecialist/
|-- corpus/
|   |-- strings.py       # Karplus-Strong plucked string, seeded and reproducible
|   |-- amp.py           # Deterministic saturation and cabinet voicing
|   |-- performance.py   # Rhythm riffs and lead lines, sharing no note event
|   `-- mixture.py       # One example whose role labels are exact by construction
`-- vendor/              # role_metrics.py and thresholds.json, copied from a tag

Tests/
|-- test_vendored_metrics.py   # The copies still match, and are still uncalibrated
`-- test_corpus_contract.py    # The rendered ground truth satisfies the gates
```

Run it:

```powershell
python -m unittest discover -s Tests -t . -v
```

## First finding: the leakage gate is permissive

Measured against this corpus with the contract's uncalibrated target of
-12 dB:

| Corpus | Lead-versus-rhythm leakage | Gate |
| --- | --- | --- |
| Rendered honestly | -48.7 dB | passes with 36 dB of margin |
| Lead is the riff again | -0.0 dB | rejected |
| 30% of the rhythm bled into the lead | -12.4 dB | **passes** |

A corpus can bleed 30% of one role into the other and still clear the gate. The
contract already marks these thresholds `"calibrated": false`, and this is the
first evidence toward what they should become. Nothing is changed on the
strength of one probe; the observation is pinned by a test that asserts real
margin rather than a bare pass, so synthesis that quietly muddied the labels
would show up as a collapsing margin instead of a still-passing gate.

## Status

Nothing here is admissible yet, and the honest statement of where this stands is
that a corpus renders and satisfies the gates its own ground truth must satisfy.
Training, evaluation on real metal recordings, and the perceptual evidence the
contract requires are all still ahead.
