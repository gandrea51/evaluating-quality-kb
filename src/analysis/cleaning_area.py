from sentence_transformers import SentenceTransformer
from collections import Counter
from pathlib import Path
from analysis.core import build_valid_pairs, candidate_summary, cosine_matrix, find_contradiction_candidates
from analysis.core import nearest_neighbor_sim, prepare_chunks, redundancy_at, redundancy_sweep, compatible_mask, DEFAULT_ENTITIES
from analysis.utils import read_jsonl, pca3, MODEL
import numpy as np
import time
import os

"""
    Prepare content analysis for the view.
    - reads JSONL chunks and separates short fragments from parsable texts;
    - calculates PCA embeddings and projections, reusing cached models and results;
    - similar measure, neighbors, groups of quasi-duplicates and discordant candidates;
    - builds summaries, text samples, and geometric data for visualization;
    - optionally select the pairs to send to the LLM judgment.

    Returns the aggregated data for the dashboard and, separately, the information needed to start the asynchronous evaluation job.
"""

os.environ.setdefault("MPLBACKEND", "Agg")
DEFAULT_THRESHOLD = (0.80, 0.85, 0.90, 0.92, 0.95, 0.98, 0.99)

_models = {}
_embedding_cache = {}


# (group one)



