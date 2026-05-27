# fmri-zsl-pipeline
# Zero-Shot fMRI Brain Decoding

Reproducible implementation of the Semantic Output Code (SOC) 
framework for zero-shot decoding of conceptual knowledge from 
human fMRI data.

## Paper

[Md Rashedur Rahman] (2024). Reproducible Zero-Shot Decoding of Conceptual 
Knowledge from Human fMRI: A Systematic Evaluation of the Semantic 
Output Code Framework. PLOS ONE (under review).

Preprint: [add bioRxiv link once posted]

## Dataset

The fMRI data used in this work is publicly available from the 
Mitchell Neuroimaging Laboratory at Carnegie Mellon University:
http://www.cs.cmu.edu/~fmri/science2008/

Download the subject files (data-science-P1.mat through 
data-science-P9.mat) and place them in the same directory 
as the pipeline file before running.

## Files

- `zsl_fmri_pipeline.py` — Main pipeline: data loading, 
  voxel selection, S map, L map, evaluation functions
- `test_zsl_pipeline.py` — 30 unit tests covering all 
  pipeline components (run with: pytest test_zsl_pipeline.py -v)
- `analysis.ipynb` — Complete analysis notebook: all 
  experiments, sensitivity analyses, and figure generation
- `requirements.txt` — Required Python packages

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```python
from zsl_fmri_pipeline import (
    build_knowledge_base,
    load_subject_data,
    fast_select_stable_voxels,
    evaluate_pairwise,
    evaluate_leave_one_word_out,
)

# Build knowledge base
K, words, feat_names = build_knowledge_base()

# Load subject data
X_raw, words_per_sample, _, _ = load_subject_data(
    "data-science-P1.mat")

# Select stable voxels
from zsl_fmri_pipeline import fast_select_stable_voxels
sel = fast_select_stable_voxels(X_raw, words_per_sample, 500)
X = X_raw[:, sel]

# Evaluate
pacc = evaluate_pairwise(X, words_per_sample, K, words)
print(f"Pairwise accuracy: {pacc:.3f}")
```

## Results

| Configuration | Mean pairwise accuracy |
|---|---|
| Hand-coded KB, variance selector | 71.4% |
| Official 25-verb KB, stability selector | 76.5% |
| Mitchell et al. (2008) published | 77.0% |

## Citation

If you use this code please cite:
[add full citation once published]

## License

MIT License — see LICENSE file
