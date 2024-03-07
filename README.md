# AUDIO EFFECTS DATASET

---
## Dry Inputs

Available here: [https://zenodo.org/uploads/10455730](https://zenodo.org/uploads/10455730)

Dry inputs are a selection of clean guitar and bass recordings from several sources:

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
## External Data

This repo contains also links to external sources (i.e., data recorded by others for scientific publication purposes or personal projects). In these cases the dry inputs will be different from the ones described above and will have a separate link for download.

Attributions to the original authors are included in this repo and references to publication, code, webpage etc. are included in a README file.

---
## Analog Effects

### Amplifier
- [Blackstar HT1 - Channel: Overdrive](https://zenodo.org/uploads/10794425)
  - External Source: [https://github.com/Alec-Wright/Automated-GuitarAmpModelling](https://github.com/Alec-Wright/Automated-GuitarAmpModelling)

### Compressor

- [Ampeg Optocomp](https://zenodo.org/uploads/10465454)
- [Flamma Opto Comp](https://zenodo.org/uploads/10794703)

### Overdrive

- [Fulltone Full Drive 2](https://zenodo.org/uploads/10794615)
- [Overdrive 1](Overdrive 2)

### Distortion

- [Electro Harmonix Metal Muff](https://zenodo.org/uploads/10794659)
- [Harley Benton Big Fur](https://zenodo.org/uploads/10794737) (similar to Electro Harmonix Big Muff)
- [Harley Benton Drop Kick](https://zenodo.org/uploads/10794776) (similar to Suhr Riot)

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
