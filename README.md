# AUDIO EFFECTS DATASET

## Dry Inputs

[https://zenodo.org/uploads/10455730](https://zenodo.org/uploads/10455730)

Dry inputs are a selection of clean guitar and bass recordings from different sources:

- [IDMT-SMT-GUITAR](https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html) - dataset 2 (7:23 min)
- [IDMT-SMT-GUITAR](https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html) - dataset 4 - Career SG (6:08 min)
- [IDMT-SMT-GUITAR](https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html) - dataset 4 - Ibanez 2820 (5:14 min)
- [IDMT-SMT-Bass-Single-Track](https://www.idmt.fraunhofer.de/en/publications/datasets/bass_lines.html) - (5:58 min)
- [NAM: Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler?tab=readme-ov-file#download-audio-files) - (3:11 min)
- Private Guitar Data - (5:19 min)
- YouTube Bass Recordings - (10:09 min)

Pre-processing:

- All:
  - synchronization markers (2 impulses) added at start and end of every file
- IDMT-SMT-GUITAR - dataset 2:
  - peak normalized to -6dBFS
- NAM:
  - no pre-processing
- Others:
  - peak normalized to -0.1dBFS
  - signal multiplied by random number every 5 seconds (uniform distribution [0.1, 1.0] = [-20dB, 0dB])

---

## Analog Effects

### Compressor

- [Ampeg Optocomp](https://zenodo.org/uploads/10465454)
- [Compressor 2](Compressor 2)

### Overdrive

- [Harley Benton Green Tint](Overdrive 1) - clone of Ibanez Tube Screamer
- [Overdrive 1](Overdrive 2)

### Distortion

- [Harley Benton Big Fur](Distortion 1) - clone of Electro Harmonix Big Muff
- [Distortion 1](Distortion 2)

### Fuzz

- [Fuzz 1](Fuzz 1)
- [Fuzz 1](Fuzz 2)

### Tremolo

- [Tremolo 1](Tremolo 1)
- [Tremolo 1](Tremolo 2)

---

## Digital Effects

---

## References

If you make use of AUDIO EFFECTS DATASET, please cite the following publication:

```
@article{TBD}
```