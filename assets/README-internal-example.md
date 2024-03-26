# AUDIO EFFECTS DATASET

## Ampeg Optocomp

### Settings

| Compression | Release | Output Level |
|---|---|---|
| 3 | 5 | 6 |
| 5 | 0 | 6 |
| 5 | 5 | 6 |
| 5 | 10 | 6 |
| 10 | 10 | 6 |

### Dry with markers

Dry inputs are a selection of clean guitar and bass recordings from different sources:

- [IDMT-SMT-GUITAR](https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html) - dataset 2 (7:23 min)
- [IDMT-SMT-GUITAR](https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html) - dataset 4 - Career SG (6:08 min)
- [IDMT-SMT-GUITAR](https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html) - dataset 4 - Ibanez 2820 (5:14 min)
- [IDMT-SMT-Bass-Single-Track](https://www.idmt.fraunhofer.de/en/publications/datasets/bass_lines.html) - (5:58 min)
- [NAM: Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler?tab=readme-ov-file#download-audio-files) - (3:11 min)
- Private Guitar Data - (5:19 min)
- YouTube Bass Recordings - (10:09 min)

Pre-processing:

- All: synchronization markers (2 impulses) added at start and end of every file
- IDMT-SMT-GUITAR - dataset 2:
  - peak normalized to -6dBFS
- NAM:
  - no pre-processing
- Others:
  - peak normalized to -0.1dBFS
  - signal multiplied by random number every 5 seconds (uniform distribution [0.1, 1.0] = [-20dB, 0dB])

### Authors

[Marco Comunità](https://mcomunita.github.io/) - [Centre for Digital Music](https://c4dm.eecs.qmul.ac.uk/), Queen Mary University of London

### Github

[https://github.com/mcomunita/audio-effects-dataset](https://github.com/mcomunita/audio-effects-dataset)

### Reference

If you make use of AUDIO EFFECTS DATASET, please cite the following publication:

```
@article{TBD}
```