# Cross-Lingual Prosody in Mandarin and English

This repository contains the analysis code and reproducibility materials associated with the study:

**Cross-linguistic prosodic variation in read speech from open corpora: A multidimensional acoustic analysis of Mandarin and English**

## Overview

This study investigates multidimensional acoustic prosodic variation between Mandarin Chinese and English using balanced read-speech samples drawn from two publicly available speech corpora.

* **Mandarin Chinese:** AISHELL-1
* **English:** LibriSpeech
* **Mandarin utterances:** 500
* **English utterances:** 500
* **Total utterances:** 1,000
* **Sampling rate:** 16 kHz

A unified computational pipeline is applied to both corpus samples to improve methodological consistency in cross-corpus acoustic comparison.

## Acoustic Features

Seven acoustic measures are analyzed across complementary prosodic dimensions:

### Pitch-related features

* F0 mean
* F0 standard deviation
* F0 range

### Energy-related features

* RMS energy

### Temporal and voicing features

* Duration
* Pause ratio
* Voicing ratio

## Analysis Pipeline

### Preparing a local manifest

The repository does not redistribute the original recordings. After obtaining
AISHELL-1 and LibriSpeech, create the balanced manifest and speaker pitch
configuration with:

```bash
python scripts/prepare_manifest.py \
  --mandarin-root /path/to/aishell/wav/test \
  --english-root /path/to/LibriSpeech/dev-clean \
  --manifest data/sample_manifest.csv \
  --pitch-config data/speaker_pitch_config.csv \
  --n-per-language 500
```

The selection is deterministic because files are ordered by a stable hash
rather than by the operating system's directory order. The generated CSV
files contain absolute audio paths for the local analysis environment and
can be regenerated without copying or modifying the source recordings.

The analysis consists of the following stages:

1. Audio preprocessing
2. Manifest-level audio validation
3. Acoustic feature extraction
4. Feature normalization
5. Distributional analysis
6. Mann–Whitney U testing
7. Multiple-comparison control
8. Effect-size estimation
9. Principal component analysis (PCA)
10. Linear mixed-effects modeling
11. Figure generation

The same general analytical procedure is applied to the Mandarin and English samples.

## Statistical Analysis

Between-corpus differences are evaluated using nonparametric statistical testing together with effect-size estimation.

Principal component analysis is used to characterize the multidimensional structure of the acoustic feature space.

Speaker-level variability is further considered using a linear mixed-effects model of the general form:

`F0 mean ~ Language + (1 | Speaker)`

Positive effect-size values indicate higher values in the Mandarin sample, whereas negative values indicate higher values in the English sample.

## Main Findings

The analysis identifies systematic acoustic differences between the Mandarin and English corpus samples across multiple prosodic dimensions.

Temporal measures show particularly large between-corpus effects. Duration exhibits the largest absolute effect size (*d* = −2.68), followed by pause ratio (*d* = 1.38). RMS energy also shows a substantial difference (*d* = −1.02), whereas several F0-related measures exhibit comparatively smaller effects.

The first two principal components explain 59.80% of the total variance and reveal partial separation between the two corpus samples within a shared multidimensional acoustic feature space.

Linear mixed-effects analysis further identifies a significant language-associated difference in F0 mean after accounting for speaker-level variability.

## Repository Structure

```text
CrossLingual-Prosody-Mandarin-English/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── data/
│   ├── README.md
│   └── sample_manifest.csv
│
├── src/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── extract_features.py
│   ├── statistical_analysis.py
│   ├── pca_analysis.py
│   └── mixed_effects_model.py
│
├── results/
│   └── tables/
│
├── figures/
│   ├── Figure5_distribution.png
│   ├── Figure6_standardized.png
│   ├── Figure7_PCA.png
│   ├── Figure8_LMM.png
│   └── Figure9_effect_sizes.png
│
└── scripts/
    └── reproduce_all.py
```

## Data Availability

The original speech recordings are not redistributed in this repository.

Mandarin speech data were obtained from the publicly available **AISHELL-1** corpus, and English speech data were obtained from **LibriSpeech**. Users should obtain the original speech data from the respective corpus providers and comply with their applicable licenses and terms of use.

The `data/sample_manifest.csv` file is intended to provide the identifiers and metadata required to reconstruct the subset of utterances analyzed in the study.

## Reproducibility

This repository is designed to provide the preprocessing, acoustic feature extraction, statistical analysis, dimensionality reduction, mixed-effects modeling, and figure-generation procedures associated with the study.

Detailed software dependencies are provided in `requirements.txt`.

## Interpretation

Because AISHELL-1 and LibriSpeech were independently constructed corpora, language-related differences cannot be completely separated from corpus-specific differences in speaker composition, recording conditions, and utterance characteristics.

Accordingly, results from this repository should primarily be interpreted as **corpus-level cross-linguistic acoustic differences**, rather than as direct causal effects of language typology.

## Citation

If you use the code or analytical workflow provided in this repository, please cite the associated article.

Citation information will be updated following publication.

## License

The analysis code in this repository is released under the MIT License. The original AISHELL-1 and LibriSpeech speech data remain subject to their respective licenses and terms of use.
