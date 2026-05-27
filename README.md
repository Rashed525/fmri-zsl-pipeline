# Zero-Shot fMRI Brain Decoding

Reproducible implementation of the **Semantic Output Code (SOC)** framework
for zero-shot decoding of conceptual knowledge from human fMRI data.

Reproduces all results from:

> [Md Rashedur Rahman] (2026). *Reproducible Zero-Shot Decoding of Conceptual
> Knowledge from Human fMRI: A Systematic Evaluation of the Semantic Output
> Code Framework.* PLOS ONE (under review).

---

## Results

| Configuration | Mean pairwise accuracy |
|---|---|
| Hand-coded KB, variance selector, 500 vox | 71.4% |
| Official 25-verb KB, variance selector, per-subject optimal | 74.5% |
| Official 25-verb KB, **stability selector**, per-subject optimal | **76.5%** |
| Mitchell et al. (2008) published benchmark | 77.0% |

---

## Dataset

The fMRI data is publicly available from the Mitchell Neuroimaging Laboratory
at Carnegie Mellon University:

**Download:** http://www.cs.cmu.edu/~fmri/science2008/

Download `data-science-P1.mat` through `data-science-P9.mat` and place them
in the same directory as this repository before running.

The semantic feature vectors are published at:
http://www.cs.cmu.edu/~tom/science2008/semanticFeatureVectors.html

**Note:** The word *skyscraper* is excluded from the evaluation vocabulary
because its official feature vector is a zero vector, causing a systematic
evaluation artefact. See paper Section 4.2 for full documentation.

---

## Repository structure

```
fmri-zsl-pipeline/
├── zsl_fmri_pipeline.py   # Main pipeline: KB, data loading, S map, L map, evaluation
├── test_zsl_pipeline.py   # 30 unit tests (run with: pytest test_zsl_pipeline.py -v)
├── analysis.ipynb         # Complete analysis notebook — reproduces all paper results
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yourusername/fmri-zsl-pipeline.git
cd fmri-zsl-pipeline
pip install -r requirements.txt
```

---

## Quick start

```python
from zsl_fmri_pipeline import (
    build_knowledge_base,
    load_subject_data,
    stability_select,
    evaluate_pairwise,
    evaluate_leave_one_word_out,
)
import numpy as np

# Build knowledge base
K, words, feat_names = build_knowledge_base()
print(f"KB: {K.shape[0]} words x {K.shape[1]} features")

# Load subject data
X_raw, words_per_sample, _, _ = load_subject_data("data-science-P1.mat")

# Filter to KB vocabulary
keep = [i for i, w in enumerate(words_per_sample) if w in set(words)]
X_raw = X_raw[keep]
words_per_sample = [words_per_sample[i] for i in keep]

# Select stable voxels (correlation-stability criterion)
sel = stability_select("data-science-P1.mat", n_voxels=1000)
X   = X_raw[:, sel]

# Evaluate — pairwise 2-way forced choice
pacc = evaluate_pairwise(X, words_per_sample, K, words, alpha=0.01)
print(f"Pairwise accuracy: {pacc:.3f}")  # expect ~0.84 for P1
```

---

## Running the full analysis

Open `analysis.ipynb` in Jupyter and run all cells from top to bottom.
The notebook reproduces every table and figure in the paper in order:

1. Knowledge base construction
2. Baseline (hand-coded KB, variance selector)
3. Sensitivity analysis: normalisation, alpha sweep, voxel sweep
4. Full replication: stability selector, all 9 subjects
5. Statistical significance test
6. All four paper figures

---

## Running the unit tests

```bash
pytest test_zsl_pipeline.py -v
```

All 30 tests should pass. The tests cover:
- Knowledge base construction and validation
- S map (Ridge regression)
- L map (1-nearest neighbour)
- ZeroShotClassifier end-to-end
- Voxel selection
- Both evaluation protocols
- Integration tests

---

## Key findings

- **76.5% mean pairwise accuracy** across all 9 subjects with the official
  knowledge base and correlation-stability voxel selector — within 0.5
  percentage points of the Mitchell (2008) published benchmark of 77.0%
- **Skyscraper artefact:** a zero-vector placeholder for *skyscraper* in the
  official feature table suppresses accuracy by ~8 percentage points.
  Documented and resolved in this work.
- **Voxel selector matters:** the correlation-stability selector outperforms
  the variance-based selector by ~2 percentage points
- **Alpha robustness:** performance is stable across α ∈ [0.001, 0.1]
- **Inter-subject variability:** pairwise accuracy ranges from 70.0% (P9) to
  84.1% (P1), a 14-point spread

---

## Citation

If you use this code please cite:

```bibtex
@article{author2024fmri,
  title   = {Reproducible Zero-Shot Decoding of Conceptual Knowledge from
             Human fMRI: A Systematic Evaluation of the Semantic Output
             Code Framework},
  author  = {[Author Name]},
  journal = {PLOS ONE},
  year    = {2024},
  note    = {Under review}
}
```

And the original dataset and framework:

```bibtex
@article{mitchell2008predicting,
  title   = {Predicting human brain activity associated with the meanings of nouns},
  author  = {Mitchell, Tom M and others},
  journal = {Science},
  volume  = {320},
  number  = {5880},
  pages   = {1191--1195},
  year    = {2008}
}

@inproceedings{palatucci2009zero,
  title     = {Zero-shot learning with semantic output codes},
  author    = {Palatucci, Mark and others},
  booktitle = {NeurIPS},
  pages     = {1410--1418},
  year      = {2009}
}
```

---

## License

MIT License — see LICENSE file for details.
