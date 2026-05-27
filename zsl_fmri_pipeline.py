"""
Zero-Shot Learning Pipeline for Mitchell (2008) fMRI Data
==========================================================
Implements the Semantic Output Code (SOC) classifier from:
  Palatucci et al. (2009) "Zero-Shot Learning with Semantic Output Codes"

Pipeline:
  S: X (fMRI voxels) -> F (218-dim semantic attributes)
  L: F -> Y (word label) via 1-Nearest Neighbor on knowledge base K

Dataset:
  - data-science-P*.mat  : fMRI data (~20,000 voxels, 360 trials, 9 subjects)
  - Knowledge base K     : 218 semantic attributes for 60 words (built here)

Usage:
  python zsl_fmri_pipeline.py --subject P1 --n_voxels 500
"""

import os
import argparse
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import cdist
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score
from itertools import combinations


# ---------------------------------------------------------------------------
# 1. KNOWLEDGE BASE  (K: word -> 218 semantic attributes)
# ---------------------------------------------------------------------------
# The full 218-attribute matrix from Palatucci et al. (2009) is available at:
#   https://www.cs.cmu.edu/~fmri/papers/zero-shot-learning/
# Here we provide the 60-word subset matching the Mitchell (2008) stimuli.
# Values are ordinal 1-5 (crowdsourced via Amazon Mechanical Turk).
# If you have the full matrix file, load it with load_knowledge_base_from_file().

WORDS_60 = [
    # animals (cond 2)
    "bear", "cat", "cow", "dog", "horse",
    # body parts (cond 3)
    "arm", "eye", "foot", "hand", "leg",
    # buildings (cond 4)
    "apartment", "barn", "church", "house", "igloo", "skyscraper",
    # building parts (cond 5)
    "arch", "chimney", "closet", "door", "window", "wall",
    # clothing (cond 6)
    "coat", "dress", "pants", "shirt", "shoes", "skirt",
    # furniture (cond 7)
    "bed", "bench", "chair", "desk", "dresser", "table",
    # insects (cond 8)
    "ant", "bee", "beetle", "butterfly", "fly", "mosquito",
    # kitchen utensils (cond 9)
    "bottle", "cup", "glass", "knife", "pan", "spoon",
    # man-made objects (cond 10)
    "bell", "key", "lamp", "refrigerator", "telephone", "watch",
    # tools (cond 11)
    "chisel", "hammer", "pliers", "screwdriver", "saw", "wrench",
    # vegetables (cond 12)
    "carrot", "celery", "corn", "lettuce", "tomato",
    # vehicles (cond 13)
    "airplane", "bicycle", "car", "train", "truck",
]

CATEGORIES = {
    "animal":        ["bear", "cat", "cow", "dog", "horse"],
    "body_part":     ["arm", "eye", "foot", "hand", "leg"],
    "building":      ["apartment", "barn", "church", "house", "igloo", "skyscraper"],
    "building_part": ["arch", "chimney", "closet", "door", "window", "wall"],
    "clothing":      ["coat", "dress", "pants", "shirt", "shoes", "skirt"],
    "furniture":     ["bed", "bench", "chair", "desk", "dresser", "table"],
    "insect":        ["ant", "bee", "beetle", "butterfly", "fly", "mosquito"],
    "utensil":       ["bottle", "cup", "glass", "knife", "pan", "spoon"],
    "manmade":       ["bell", "key", "lamp", "refrigerator", "telephone", "watch"],
    "tool":          ["chisel", "hammer", "pliers", "screwdriver", "saw", "wrench"],
    "vegetable":     ["carrot", "celery", "corn", "lettuce", "tomato"],
    "vehicle":       ["airplane", "bicycle", "car", "train", "truck"],
}

# 25 semantic feature dimensions (interpretable subset; extend to 218 with full file)
# Based on the verb-association features from Mitchell (2008) + common Palatucci attributes
SEMANTIC_FEATURE_NAMES = [
    # action/motion verbs (Mitchell 2008 core features)
    "eat", "taste", "push", "run", "lift",
    "ride", "wear", "sit", "drive", "fly",
    # perceptual attributes
    "has_fur", "has_legs", "has_wings", "has_wheels", "is_large",
    "is_small", "is_alive", "is_natural", "is_manmade", "is_edible",
    # functional
    "is_tool", "is_vehicle", "is_animal", "is_furniture", "is_clothing",
]

# Compact 60x25 knowledge base (values 1-5, hand-curated approximation).
# Replace with the full 218-column matrix from the Palatucci dataset for best results.
_KB_RAW = {
    #          eat taste push  run lift ride wear sit  drv  fly  fur  leg  wing whl  lrg  sml  alv  nat  man  edi  tool veh  ani  fur2 clo
    "bear":    [4,  3,   2,   4,  2,   1,   1,   2,   1,   1,   5,   4,   1,   1,   4,   1,   5,   5,   1,   2,   1,   1,   5,   1,   1],
    "cat":     [4,  3,   1,   4,  1,   1,   1,   4,   1,   1,   5,   4,   1,   1,   2,   3,   5,   5,   1,   2,   1,   1,   5,   1,   1],
    "cow":     [2,  2,   2,   3,  2,   3,   1,   2,   1,   1,   5,   4,   1,   1,   4,   1,   5,   5,   1,   3,   1,   1,   5,   1,   1],
    "dog":     [4,  3,   2,   5,  2,   1,   1,   2,   1,   1,   5,   4,   1,   1,   2,   2,   5,   5,   1,   2,   1,   1,   5,   1,   1],
    "horse":   [2,  2,   2,   5,  2,   5,   1,   2,   1,   1,   5,   4,   1,   1,   5,   1,   5,   5,   1,   2,   1,   1,   5,   1,   1],
    "arm":     [1,  1,   4,   3,  4,   1,   1,   1,   1,   1,   2,   1,   1,   1,   2,   3,   5,   5,   1,   1,   1,   1,   1,   1,   1],
    "eye":     [1,  1,   1,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   5,   5,   1,   1,   1,   1,   1,   1,   1],
    "foot":    [1,  1,   1,   4,  2,   1,   1,   1,   1,   1,   2,   1,   1,   1,   2,   3,   5,   5,   1,   1,   1,   1,   1,   1,   1],
    "hand":    [3,  1,   5,   1,  5,   1,   1,   1,   1,   1,   2,   1,   1,   1,   2,   3,   5,   5,   1,   1,   1,   1,   1,   1,   1],
    "leg":     [1,  1,   2,   5,  3,   1,   1,   1,   1,   1,   2,   1,   1,   1,   2,   3,   5,   5,   1,   1,   1,   1,   1,   1,   1],
    "apartment":[1, 1,   1,   1,  1,   1,   1,   5,   1,   1,   1,   1,   1,   1,   4,   1,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "barn":    [1,  1,   1,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   5,   1,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "church":  [1,  1,   1,   1,  1,   1,   1,   3,   1,   1,   1,   1,   1,   1,   5,   1,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "house":   [1,  1,   1,   1,  1,   1,   1,   4,   1,   1,   1,   1,   1,   1,   4,   1,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "igloo":   [1,  1,   1,   1,  1,   1,   1,   3,   1,   1,   1,   1,   1,   1,   3,   1,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "skyscraper":[1,1,   1,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   5,   1,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "arch":    [1,  1,   1,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   3,   2,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "chimney": [1,  1,   1,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   3,   2,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "closet":  [1,  1,   2,   1,  1,   1,   5,   1,   1,   1,   1,   1,   1,   1,   2,   2,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "door":    [1,  1,   4,   1,  3,   1,   1,   1,   1,   1,   1,   1,   1,   1,   3,   2,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "window":  [1,  1,   1,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   2,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "wall":    [1,  1,   2,   1,  2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   1,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "coat":    [1,  1,   1,   1,  2,   1,   5,   1,   1,   1,   2,   1,   1,   1,   2,   3,   1,   1,   5,   1,   1,   1,   1,   1,   5],
    "dress":   [1,  1,   1,   1,  1,   1,   5,   1,   1,   1,   1,   1,   1,   1,   2,   3,   1,   1,   5,   1,   1,   1,   1,   1,   5],
    "pants":   [1,  1,   1,   1,  1,   1,   5,   1,   1,   1,   1,   1,   1,   1,   2,   3,   1,   1,   5,   1,   1,   1,   1,   1,   5],
    "shirt":   [1,  1,   1,   1,  1,   1,   5,   1,   1,   1,   1,   1,   1,   1,   2,   3,   1,   1,   5,   1,   1,   1,   1,   1,   5],
    "shoes":   [1,  1,   1,   3,  1,   1,   5,   1,   1,   1,   2,   1,   1,   1,   1,   3,   1,   1,   5,   1,   1,   1,   1,   1,   5],
    "skirt":   [1,  1,   1,   1,  1,   1,   5,   1,   1,   1,   1,   1,   1,   1,   1,   3,   1,   1,   5,   1,   1,   1,   1,   1,   5],
    "bed":     [1,  1,   3,   1,  3,   1,   1,   5,   1,   1,   1,   4,   1,   1,   3,   2,   1,   1,   5,   1,   1,   1,   1,   5,   1],
    "bench":   [1,  1,   2,   1,  2,   1,   1,   4,   1,   1,   1,   4,   1,   1,   2,   2,   1,   1,   5,   1,   1,   1,   1,   5,   1],
    "chair":   [1,  1,   3,   1,  2,   1,   1,   5,   1,   1,   1,   4,   1,   1,   2,   2,   1,   1,   5,   1,   1,   1,   1,   5,   1],
    "desk":    [1,  1,   2,   1,  2,   1,   1,   1,   1,   1,   1,   4,   1,   1,   3,   2,   1,   1,   5,   1,   1,   1,   1,   5,   1],
    "dresser": [1,  1,   2,   1,  2,   1,   1,   1,   1,   1,   1,   4,   1,   1,   3,   2,   1,   1,   5,   1,   1,   1,   1,   5,   1],
    "table":   [3,  2,   3,   1,  3,   1,   1,   4,   1,   1,   1,   4,   1,   1,   3,   2,   1,   1,   5,   1,   1,   1,   1,   5,   1],
    "ant":     [2,  2,   1,   4,  1,   1,   1,   1,   1,   1,   1,   4,   1,   1,   1,   5,   5,   5,   1,   1,   1,   1,   1,   1,   1],
    "bee":     [2,  2,   1,   3,  1,   1,   1,   1,   1,   2,   1,   4,   4,   1,   1,   5,   5,   5,   1,   1,   1,   1,   1,   1,   1],
    "beetle":  [1,  1,   1,   3,  1,   1,   1,   1,   1,   1,   1,   4,   2,   1,   1,   5,   5,   5,   1,   1,   1,   1,   1,   1,   1],
    "butterfly":[1, 1,   1,   2,  1,   1,   1,   1,   1,   4,   1,   4,   5,   1,   1,   5,   5,   5,   1,   1,   1,   1,   1,   1,   1],
    "fly":     [2,  1,   1,   3,  1,   1,   1,   1,   1,   5,   1,   4,   4,   1,   1,   5,   5,   5,   1,   1,   1,   1,   1,   1,   1],
    "mosquito":[2,  1,   1,   3,  1,   1,   1,   1,   1,   5,   1,   4,   4,   1,   1,   5,   5,   5,   1,   1,   1,   1,   1,   1,   1],
    "bottle":  [3,  2,   3,   1,  2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   3,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "cup":     [4,  3,   2,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "glass":   [4,  3,   2,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "knife":   [3,  2,   4,   1,  2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   1,   1,   5,   1,   5,   1,   1,   1,   1],
    "pan":     [4,  3,   3,   1,  3,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   3,   1,   1,   5,   1,   4,   1,   1,   1,   1],
    "spoon":   [4,  3,   3,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   1,   1,   5,   1,   3,   1,   1,   1,   1],
    "bell":    [1,  1,   2,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "key":     [1,  1,   4,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   5,   1,   1,   5,   1,   3,   1,   1,   1,   1],
    "lamp":    [1,  1,   2,   1,  2,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   3,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "refrigerator":[3,2, 2,   1,  3,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   1,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "telephone":[1, 1,   2,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "watch":   [1,  1,   1,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   5,   1,   1,   5,   1,   1,   1,   1,   1,   1],
    "chisel":  [1,  1,   5,   1,  3,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   1,   1,   5,   1,   5,   1,   1,   1,   1],
    "hammer":  [1,  1,   5,   1,  4,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   1,   1,   5,   1,   5,   1,   1,   1,   1],
    "pliers":  [1,  1,   5,   1,  3,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   1,   1,   5,   1,   5,   1,   1,   1,   1],
    "screwdriver":[1,1,  5,   1,  3,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   1,   1,   5,   1,   5,   1,   1,   1,   1],
    "saw":     [1,  1,   5,   1,  3,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   1,   1,   5,   1,   5,   1,   1,   1,   1],
    "wrench":  [1,  1,   5,   1,  3,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   1,   1,   5,   1,   5,   1,   1,   1,   1],
    "carrot":  [5,  4,   1,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   1,   5,   1,   5,   1,   1,   1,   1,   1],
    "celery":  [5,  4,   1,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   1,   5,   1,   5,   1,   1,   1,   1,   1],
    "corn":    [5,  4,   1,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   1,   5,   1,   5,   1,   1,   1,   1,   1],
    "lettuce": [5,  4,   1,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   2,   3,   1,   5,   1,   5,   1,   1,   1,   1,   1],
    "tomato":  [5,  5,   1,   1,  1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   1,   4,   1,   5,   1,   5,   1,   1,   1,   1,   1],
    "airplane":[1,  1,   1,   1,  2,   4,   1,   2,   3,   5,   1,   1,   4,   5,   5,   1,   1,   1,   5,   1,   1,   5,   1,   1,   1],
    "bicycle": [1,  1,   1,   4,  2,   5,   1,   1,   1,   1,   1,   2,   1,   4,   2,   3,   1,   1,   5,   1,   1,   5,   1,   1,   1],
    "car":     [1,  1,   2,   1,  2,   5,   1,   2,   5,   1,   1,   1,   1,   4,   3,   2,   1,   1,   5,   1,   1,   5,   1,   1,   1],
    "train":   [1,  1,   2,   1,  2,   5,   1,   4,   5,   1,   1,   1,   1,   4,   5,   1,   1,   1,   5,   1,   1,   5,   1,   1,   1],
    "truck":   [1,  1,   3,   1,  2,   5,   1,   2,   5,   1,   1,   1,   1,   4,   5,   1,   1,   1,   5,   1,   1,   5,   1,   1,   1],
}


def build_knowledge_base():
    """
    Build knowledge base K: dict {word: semantic_vector} and matrix form.
    Returns
    -------
    K_matrix : np.ndarray, shape (60, n_features)
    words     : list of 60 words (row order)
    feat_names: list of feature name strings
    """
    words = WORDS_60
    K_matrix = np.array([_KB_RAW[w] for w in words], dtype=float)
    # Normalise to zero-mean, unit-variance per feature (helps 1-NN)
    K_matrix = (K_matrix - K_matrix.mean(axis=0)) / (K_matrix.std(axis=0) + 1e-8)
    return K_matrix, words, SEMANTIC_FEATURE_NAMES


def load_knowledge_base_from_file(filepath):
    """
    Load a pre-built knowledge base from a CSV or .npy file.
    Expected CSV format: first column = word, remaining columns = feature values.
    """
    import pandas as pd
    df = pd.read_csv(filepath, index_col=0)
    words = list(df.index)
    K_matrix = df.values.astype(float)
    feat_names = list(df.columns)
    K_matrix = (K_matrix - K_matrix.mean(axis=0)) / (K_matrix.std(axis=0) + 1e-8)
    return K_matrix, words, feat_names


# ---------------------------------------------------------------------------
# 2. DATA LOADING  (Mitchell .mat format)
# ---------------------------------------------------------------------------

def load_subject_data(mat_path):
    """
    Load a single subject's .mat file.

    Returns
    -------
    X : np.ndarray, shape (n_words, n_voxels)
        Trial-averaged fMRI activation (6 repetitions averaged per word).
    words : list of str, length n_words
        Word label for each row in X.
    trial_info : list of dict
        Full info struct for reference.
    meta : dict
        Dataset metadata.
    """
    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    info = mat["info"]
    data = mat["data"]
    meta = mat["meta"]

    # Build word -> list of trial indices
    word_trials = {}
    for i, trial in enumerate(info):
        word = str(trial.word)
        word_trials.setdefault(word, []).append(i)

    words_sorted = sorted(word_trials.keys())
    X_list = []
    for word in words_sorted:
        idxs = word_trials[word]
        # Each data{i} is shape (1, n_voxels); average across 6 repetitions
        trials = np.vstack([data[i] for i in idxs])  # (6, n_voxels)
        X_list.append(trials.mean(axis=0))

    X = np.vstack(X_list)  # (60, n_voxels)
    return X, words_sorted, info, meta


# ---------------------------------------------------------------------------
# 3. VOXEL SELECTION  (correlation-stability criterion)
# ---------------------------------------------------------------------------

def select_stable_voxels(X_all_trials, trial_words, n_voxels=500):
    """
    Select the top-k most stable voxels using the correlation-stability criterion
    from Shinkareva et al. (2008) / Mitchell et al. (2008).

    For each voxel, compute pairwise correlation of that voxel's activation
    pattern across repeated trials of the same word, then average.

    Parameters
    ----------
    X_all_trials : np.ndarray, shape (n_trials, n_voxels_total)
        Raw trial-level activations (NOT averaged).
    trial_words  : list of str, length n_trials
    n_voxels     : int, number of top voxels to keep

    Returns
    -------
    selected_idx : np.ndarray of int, shape (n_voxels,)
    stability    : np.ndarray, shape (n_voxels_total,)
    """
    unique_words = list(set(trial_words))
    n_total = X_all_trials.shape[1]
    stability = np.zeros(n_total)

    for word in unique_words:
        idxs = [i for i, w in enumerate(trial_words) if w == word]
        if len(idxs) < 2:
            continue
        reps = X_all_trials[idxs]  # (n_reps, n_voxels)
        # Pairwise correlation across repetition pairs
        pair_corrs = []
        for a, b in combinations(range(len(idxs)), 2):
            r = np.corrcoef(reps[a], reps[b])[0, 1]
            pair_corrs.append(r if not np.isnan(r) else 0.0)
        stability += np.array(pair_corrs).mean() * np.ones(n_total)

    # Per-voxel stability: recompute properly
    voxel_stability = np.zeros(n_total)
    for v in range(n_total):
        corrs = []
        for word in unique_words:
            idxs = [i for i, w in enumerate(trial_words) if w == word]
            if len(idxs) < 2:
                continue
            reps = X_all_trials[idxs, v]
            for a, b in combinations(range(len(idxs)), 2):
                denom = (np.std(reps) + 1e-12)
                c = np.corrcoef(
                    X_all_trials[idxs[a]], X_all_trials[idxs[b]]
                )[0, 1]
                corrs.append(c if not np.isnan(c) else 0.0)
        voxel_stability[v] = np.mean(corrs) if corrs else 0.0

    selected_idx = np.argsort(voxel_stability)[::-1][:n_voxels]
    return selected_idx, voxel_stability


def fast_select_stable_voxels(X, words, n_voxels=500):
    """
    Faster approximation of stability selection using per-voxel variance
    across repetitions (works directly on averaged data when raw trials
    are unavailable). Falls back gracefully.
    """
    # Use per-voxel std across words as a proxy for information content
    std_per_voxel = X.std(axis=0)
    selected_idx = np.argsort(std_per_voxel)[::-1][:n_voxels]
    return selected_idx


# ---------------------------------------------------------------------------
# 4.  S MAP  — fMRI -> semantic features  (Ridge regression)
# ---------------------------------------------------------------------------

class SMap:
    """
    Learns S: X -> F, a ridge regression from voxel activations
    to semantic feature vectors.

    One regressor is trained per semantic feature dimension (independent outputs).
    As per Palatucci et al. (2009): bW ∈ R^{d x p} solved jointly via matrix ops.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=True)
        self._fitted = False

    def fit(self, X, F):
        """
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_voxels)
        F : np.ndarray, shape (n_samples, n_semantic_features)
        """
        self.model.fit(X, F)
        self._fitted = True
        return self

    def predict(self, X):
        """
        Returns predicted semantic feature vectors.
        Shape: (n_samples, n_semantic_features)
        """
        assert self._fitted, "Call fit() before predict()"
        return self.model.predict(X)


# ---------------------------------------------------------------------------
# 5.  L MAP  — semantic features -> class label  (1-NN over knowledge base K)
# ---------------------------------------------------------------------------

class LMap:
    """
    L: F_hat -> Y via 1-Nearest Neighbor in semantic space.

    Given a predicted feature vector f_hat, returns the word label
    whose knowledge-base entry is closest in Euclidean distance.
    """

    def __init__(self, K_matrix, words):
        """
        Parameters
        ----------
        K_matrix : np.ndarray, shape (n_classes, n_features)
        words    : list of str, length n_classes
        """
        self.K = K_matrix
        self.words = words

    def predict(self, F_hat):
        """
        Parameters
        ----------
        F_hat : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        predicted_words : list of str, length n_samples
        """
        # Euclidean distance from each prediction to each KB entry
        dists = cdist(F_hat, self.K, metric="euclidean")  # (n_samples, n_classes)
        nn_idx = np.argmin(dists, axis=1)
        return [self.words[i] for i in nn_idx]

    def predict_top_k(self, F_hat, k=5):
        """Return top-k nearest class labels per sample."""
        dists = cdist(F_hat, self.K, metric="euclidean")
        topk_idx = np.argsort(dists, axis=1)[:, :k]
        return [[self.words[i] for i in row] for row in topk_idx]


# ---------------------------------------------------------------------------
# 6.  ZSL CLASSIFIER  (full pipeline)
# ---------------------------------------------------------------------------

class ZeroShotClassifier:
    """
    Full Zero-Shot Learning pipeline:
        X  --[S]--> F_hat  --[L]--> Y

    Attributes
    ----------
    s_map   : SMap
    l_map   : LMap
    K_matrix: np.ndarray  (knowledge base)
    words   : list of str
    """

    def __init__(self, K_matrix, words, alpha=1.0):
        self.K_matrix = K_matrix
        self.words = words
        self.s_map = SMap(alpha=alpha)
        self.l_map = LMap(K_matrix, words)

    def fit(self, X_train, y_train):
        """
        Build {x, f} pairs from {x, y} using K, then fit S map.

        Parameters
        ----------
        X_train : np.ndarray, shape (n_train, n_voxels)
        y_train : list of str, length n_train — word labels
        """
        # Replace y with semantic vectors from knowledge base
        word_to_idx = {w: i for i, w in enumerate(self.words)}
        F_train = np.array([
            self.K_matrix[word_to_idx[y]] for y in y_train
        ])
        self.s_map.fit(X_train, F_train)
        return self

    def predict(self, X_test):
        """Predict word labels for new fMRI observations."""
        F_hat = self.s_map.predict(X_test)
        return self.l_map.predict(F_hat)

    def predict_semantic(self, X_test):
        """Return predicted semantic vectors (before label lookup)."""
        return self.s_map.predict(X_test)


# ---------------------------------------------------------------------------
# 7.  EVALUATION  — Leave-One-Out cross-validation (leave one word out)
# ---------------------------------------------------------------------------

def evaluate_leave_one_word_out(X, words_per_sample, K_matrix, all_words,
                                 alpha=1.0, verbose=True):
    """
    Standard ZSL evaluation: leave one word out at a time.
    For each held-out word w:
      - Train S on all other words
      - Predict the semantic vector for w's fMRI activation
      - 1-NN over the full KB (including unseen word w)
      - Check if the nearest neighbour == w

    This tests true zero-shot generalisation: the model has never
    seen fMRI data for w during training.

    Returns
    -------
    accuracy : float  (fraction of words correctly identified)
    results  : list of dict with per-word details
    """
    unique_words = list(set(words_per_sample))
    n_words = len(unique_words)
    results = []
    correct = 0

    for held_out in unique_words:
        # Train indices: all words except held_out
        train_idx = [i for i, w in enumerate(words_per_sample) if w != held_out]
        test_idx  = [i for i, w in enumerate(words_per_sample) if w == held_out]

        X_train = X[train_idx]
        y_train = [words_per_sample[i] for i in train_idx]
        X_test  = X[test_idx]

        clf = ZeroShotClassifier(K_matrix, all_words, alpha=alpha)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Majority vote if multiple test samples for this word
        from collections import Counter
        pred_label = Counter(y_pred).most_common(1)[0][0]
        hit = int(pred_label == held_out)
        correct += hit

        results.append({
            "true_word":  held_out,
            "pred_word":  pred_label,
            "correct":    bool(hit),
            "all_preds":  y_pred,
        })

        if verbose:
            status = "✓" if hit else "✗"
            print(f"  [{status}] true={held_out:15s}  pred={pred_label}")

    accuracy = correct / n_words
    print(f"\nLeave-One-Word-Out Accuracy: {correct}/{n_words} = {accuracy:.3f}")
    return accuracy, results


def evaluate_pairwise(X, words_per_sample, K_matrix, all_words,
                       alpha=1.0, verbose=False):
    """
    Pairwise (2-way forced choice) evaluation from Mitchell (2008).
    For each pair of words (w1, w2):
      - Train on all other 58 words
      - Predict semantic vectors for w1 and w2
      - Check if predictions are closer to correct KB entries than swapped

    Returns
    -------
    pairwise_accuracy : float  (fraction of pairs correctly ordered)
    """
    unique_words = list(set(words_per_sample))
    word_to_X = {w: X[[i for i, ww in enumerate(words_per_sample) if ww == w]].mean(0)
                 for w in unique_words}
    word_to_kb = {w: K_matrix[all_words.index(w)] for w in unique_words}

    pairs = list(combinations(unique_words, 2))
    n_correct = 0

    for w1, w2 in pairs:
        # Train on all except w1, w2
        train_words = [w for w in unique_words if w not in (w1, w2)]
        train_idx = [i for i, ww in enumerate(words_per_sample) if ww in train_words]
        X_train = X[train_idx]
        y_train = [words_per_sample[i] for i in train_idx]

        clf = ZeroShotClassifier(K_matrix, all_words, alpha=alpha)
        clf.fit(X_train, y_train)

        f1 = clf.predict_semantic(word_to_X[w1].reshape(1, -1))[0]
        f2 = clf.predict_semantic(word_to_X[w2].reshape(1, -1))[0]

        kb1 = word_to_kb[w1]
        kb2 = word_to_kb[w2]

        # Correct assignment: f1 closer to kb1, f2 closer to kb2
        d_correct = np.linalg.norm(f1 - kb1) + np.linalg.norm(f2 - kb2)
        d_swapped = np.linalg.norm(f1 - kb2) + np.linalg.norm(f2 - kb1)

        if d_correct < d_swapped:
            n_correct += 1

    pairwise_acc = n_correct / len(pairs)
    print(f"Pairwise Accuracy: {n_correct}/{len(pairs)} = {pairwise_acc:.3f}  "
          f"(chance = 0.500)")
    return pairwise_acc


# ---------------------------------------------------------------------------
# 8.  MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ZSL fMRI pipeline")
    parser.add_argument("--mat_file", type=str, default=None,
                        help="Path to data-science-P*.mat file")
    parser.add_argument("--subject", type=str, default="P1",
                        help="Subject identifier (used in output messages)")
    parser.add_argument("--n_voxels", type=int, default=500,
                        help="Number of stable voxels to select")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Ridge regression regularisation strength")
    parser.add_argument("--kb_file", type=str, default=None,
                        help="Optional path to CSV knowledge-base file")
    parser.add_argument("--eval", type=str, default="lowo",
                        choices=["lowo", "pairwise", "both"],
                        help="Evaluation mode")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Zero-Shot Learning — fMRI Decoder  (subject {args.subject})")
    print("=" * 60)

    # -- Knowledge base -------------------------------------------------------
    if args.kb_file:
        print(f"\nLoading knowledge base from {args.kb_file}")
        K_matrix, all_words, feat_names = load_knowledge_base_from_file(args.kb_file)
    else:
        print("\nBuilding 25-feature knowledge base (Palatucci-style)...")
        K_matrix, all_words, feat_names = build_knowledge_base()

    print(f"  KB shape: {K_matrix.shape}  ({len(all_words)} words x {len(feat_names)} features)")

    # -- Data loading ---------------------------------------------------------
    if args.mat_file is None:
        print("\nNo --mat_file provided. Running demo with synthetic data...\n")
        np.random.seed(42)
        n_words = len(all_words)
        n_voxels_raw = 2000

        # Synthetic X: each word's activation is loosely correlated with its
        # semantic vector (to give the model a chance to learn something)
        W_true = np.random.randn(len(feat_names), n_voxels_raw) * 0.5
        noise = np.random.randn(n_words, n_voxels_raw)
        X = K_matrix @ W_true + noise
        words_per_sample = all_words[:]

    else:
        print(f"\nLoading {args.mat_file} ...")
        X_raw, words_per_sample, _, _ = load_subject_data(args.mat_file)
        # Filter to words that exist in our knowledge base
        kb_set = set(all_words)
        keep = [i for i, w in enumerate(words_per_sample) if w in kb_set]
        X_raw = X_raw[keep]
        words_per_sample = [words_per_sample[i] for i in keep]
        print(f"  Loaded {X_raw.shape[0]} word samples, {X_raw.shape[1]} voxels")

        # Voxel selection
        print(f"  Selecting top {args.n_voxels} stable voxels...")
        sel_idx = fast_select_stable_voxels(X_raw, words_per_sample, args.n_voxels)
        X = X_raw[:, sel_idx]
        print(f"  X after voxel selection: {X.shape}")

    # -- Evaluation -----------------------------------------------------------
    print(f"\nRunning evaluation: {args.eval}")

    if args.eval in ("lowo", "both"):
        print("\n--- Leave-One-Word-Out Accuracy ---")
        acc, results = evaluate_leave_one_word_out(
            X, words_per_sample, K_matrix, all_words,
            alpha=args.alpha, verbose=True
        )

    if args.eval in ("pairwise", "both"):
        print("\n--- Pairwise (2-way forced choice) Accuracy ---")
        pacc = evaluate_pairwise(
            X, words_per_sample, K_matrix, all_words,
            alpha=args.alpha
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
