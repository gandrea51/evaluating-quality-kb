from sentence_transformers import SentenceTransformer
from collections import Counter
from pathlib import Path
from kb_analysis.core import build_valid_pairs, candidate_summary, cosine_matrix, find_contradiction_candidates
from kb_analysis.core import nearest_neighbor_sim, prepare_chunks, redundancy_at, redundancy_sweep, DEFAULT_ENTITIES
from kb_analysis.utils import read_jsonl, pca3, MODEL
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

# (sentence-transformer)
def _get_model(model_name=MODEL):
    if model_name not in _models:
        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


# (embedding, pca)
def _embed_cache(texts, cache_key, model_name=MODEL, batch_size=32):
    key = (cache_key, model_name)

    if key not in _embedding_cache:
        model = _get_model(model_name)
        
        X = np.asarray(model.encode(
            texts, batch_size=batch_size, show_progress_bar=False
        ), dtype=np.float32)

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-12)

        C, explained_variance = pca3(X)
        _embedding_cache[key] = (X, C, explained_variance)

    return _embedding_cache[key]


# (short text)
def _snippet(text, length=110):
    return str(text)[:length].replace("\n", " ")

# (candidate pair)
def _pair_key(pair):
    return f'{pair["a"]}|{pair["b"]}'

# (cache key)
def _embedding_cache_key(chunks_path):
    path = Path(chunks_path)
    return str(path), path.stat().st_mtime


# (group one)
def run_contents(chunks_path="out/chunks.jsonl", out_dir="out", threshold=0.92, cand_threshold=0.85, min_chars=40, judge=None, max_pairs=10, max_show=50, model_name=MODEL):
    started = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # (chunks)
    chunks = read_jsonl(chunks_path)
    data = prepare_chunks(chunks, min_chars=min_chars)
    data["min_chars"] = min_chars

    texts = data["texts"]
    ids = data["ids"]
    n = data["n"]
    docs = data["docs"]
    doc_names = data["doc_names"]
    fragments = data["fragments"]

    fragment_counter = Counter(doc_names[doc_index] for doc_index in docs[fragments])

    # (embedding, pca)
    cache_key = _embedding_cache_key(chunks_path)
    X, C, explained_variance = _embed_cache(texts, cache_key, model_name)
    
    similarity = cosine_matrix(X)

    # (pairs, knn)
    I, J = build_valid_pairs(data)
    S = similarity[I, J]

    nearest = nearest_neighbor_sim(I, J, S, n)
    nn = nearest[nearest > -1]
    
    # (redundancy)
    redundancy_metrics, groups, edges = redundancy_at(threshold, I, J, S, docs, n, data["n_effective"])
    sweep = redundancy_sweep(DEFAULT_THRESHOLD, I, J, S, docs, n, data["n_effective"])

    # (discordance)
    candidates = find_contradiction_candidates(texts, ids, docs, I, J, S, cand_threshold, DEFAULT_ENTITIES)
    discordance_metrics = candidate_summary(candidates)
    
    # (pairs 4 llm)
    pairs_to_judge = candidates[:max_pairs] if judge and candidates else []
    todo = {
        "model": judge, "pairs": pairs_to_judge, "texts": texts, "ids": ids, "out_dir": str(out_dir)
    } if pairs_to_judge else None

    keys_to_judge = {_pair_key(pair) for pair in pairs_to_judge}
    discordance_metrics["to_judge"] = len(pairs_to_judge)

    # (data 4 ui)
    n_show = max(max_show, max_pairs)
    plot_candidates = candidates[:n_show]
    group_by_chunk = {
        chunk_index: group_index for group_index, group in enumerate(groups) for chunk_index in group
    }

    mode = Path(chunks_path).stem.removeprefix("chunks_") 

    result = {
        "params": {
            "path": mode,
            "threshold": threshold, 
            "cand_threshold": cand_threshold, 
            "min_chars": min_chars, 
            "judge": judge, 
            "max_pairs": max_pairs, 
            "model": model_name
        },
        "summary": {
            "n_chunks": n, 
            "n_analyzed": data["n_effective"], 
            "n_fragments": int(fragments.sum()), 
            "n_documents": len(doc_names), 
            "var3": float(explained_variance[:3].sum()),
            "seconds": round(time.perf_counter() - started, 1)
        },
        "redundancy": redundancy_metrics, 
        "sweep": sweep,
        "groups": [{
            "size": len(group), "members": [{
                "id": ids[chunk_index], "doc": doc_names[docs[chunk_index]], "text": _snippet(texts[chunk_index])} for chunk_index in group]}
                   for group in sorted(groups, key=len, reverse=True)[:25]],
        "discordance": discordance_metrics,
        "candidates": [{
            "key": _pair_key(pair), "a": pair["a"], "b": pair["b"], "sim": pair["sim"], "cross_doc": pair["cross_doc"],
            "numeric_diff": pair["numeric_diff"], "to_judge": _pair_key(pair) in keys_to_judge, 
            "ta": _snippet(texts[pair["i"]]), "tb": _snippet(texts[pair["j"]])} 
            for pair in candidates[:n_show]],
        "fragments_by_doc": dict(fragment_counter.most_common(10)),
        "plot": {
            "points": [{
                "id": ids[index], "doc": doc_names[docs[index]], 
                "x": float(C[index, 0]), "y": float(C[index, 1]), "z": float(C[index, 2]), 
                "frag": bool(fragments[index]), "group": group_by_chunk.get(index, -1), 
                "text": _snippet(texts[index], 90)} 
                for index in range(n)
            ],
            "edges": [[
                int(i), int(j)] for i, j in edges[:3000]
            ],
            "cands": [{
                "i": pair["i"], "j": pair["j"], "key": _pair_key(pair)
            } for pair in plot_candidates],
        },
    }
    return result, todo


