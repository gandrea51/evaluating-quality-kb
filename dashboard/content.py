from sentence_transformers import SentenceTransformer
from collections import Counter
from utils import words, numbers, position, read_jsonl, content_chars, pca3
from embedding import MODEL, redundancy_at, judge_pairs
from pathlib import Path
import numpy as np
import time
import os


os.environ.setdefault("MPLBACKEND", "Agg")
CITTA = ["agra", "jaipur", "kanpur", "delhi", "mumbai", "kolkata", "nagpur", "surat", "bangladesh"]

_model = None
_emb_cache = {}

def _embed(texts):
    global _model
    if _model is None: _model = SentenceTransformer(MODEL)
    X = np.asarray(_model.encode(texts, batch_size=32, show_progress_bar=False), dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

def _snip(t, n=110):
    return t[:n].replace("\n", " ")

def _key(p):
    return f'{p["a"]}|{p["b"]}'

def run_contents(chunks_path="out/chunks.jsonl", out_dir="out", threshold=0.92, cand_threshold=0.85, min_chars=40, judge=None, max_pairs=10, max_show=50):
    t0 = time.perf_counter()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    chunks = read_jsonl(chunks_path)
    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    n = len(chunks)
    doc_names = sorted({c["doc_id"] for c in chunks})
    docs = np.array([doc_names.index(c["doc_id"]) for c in chunks])
    pos = np.array([position(c["chunk_id"]) for c in chunks])

    ignore = np.array([content_chars(t) < min_chars for t in texts])
    n_eff = int((~ignore).sum())
    frag_by_doc = Counter(doc_names[d] for d in docs[ignore])

    # Embedding
    key = (str(chunks_path), os.path.getmtime(chunks_path))
    if key not in _emb_cache:
        X = _embed(texts); C, var = pca3(X)
        _emb_cache[key] = (X, C, var)
    X, C, var = _emb_cache[key]

    sim = X @ X.T
    I, J = np.triu_indices(n, 1)
    adjacent = (docs[I] == docs[J]) & (np.abs(pos[I] - pos[J]) == 1)
    valid = ~adjacent & ~(ignore[I] | ignore[J])
    I, J = I[valid], J[valid]
    S = sim[I, J]

    # Ridondanza
    red, groups, edges = redundancy_at(threshold, I, J, S, docs, n, n_eff)
    sweep = [redundancy_at(t, I, J, S, docs, n, n_eff)[0] for t in (0.8, 0.85, 0.9, 0.92, 0.95, 0.98, 0.99)]

    # Discordanza - candidati
    cand = []
    for k in np.where(S >= cand_threshold)[0]:
        i, j = int(I[k]), int(J[k])
        if words(texts[i]) == words(texts[j]): continue

        ci = {c for c in CITTA if c in texts[i].lower()}
        cj = {c for c in CITTA if c in texts[j].lower()}
        if ci and cj and not (ci & cj): continue

        ni, nj = numbers(texts[i]), numbers(texts[j])
        cand.append({"i": i, "j": j, "a": ids[i], "b": ids[j], "sim": float(S[k]), 
                     "cross_doc": bool(docs[i] != docs[j]), "numeric_diff": bool(ni and nj and ni != nj)})

    cand.sort(key=lambda p: (-p["cross_doc"], -p["numeric_diff"], -p["sim"]))

    # LLM in background
    judge_list = cand[:max_pairs] if (judge and cand) else []
    todo = ({"model": judge, "pairs": judge_list, "texts": texts, "ids": ids, "out_dir": out_dir} if judge_list else None)
    judge_key = {_key(p) for p in judge_list}
    
    disc = {"candidate_pairs": len(cand), "candidate_numeric_diff": sum(p["numeric_diff"] for p in cand),
            "candidate_cross_doc": sum(p["cross_doc"] for p in cand), "to_judge": len(judge_list)}

    n_show = max(max_show, max_pairs)
    plot_cands = cand[:n_show]
    in_group = {i: gi for gi, g in enumerate(groups) for i in g}

    result = {
        "params": {
            "threshold": threshold, "cand_threshold": cand_threshold, "min_chars": min_chars, "judge": judge, "max_pairs": max_pairs, "model": MODEL
        },
        "summary": {
            "n_chunks": n, "n_analyzed": n_eff, "n_fragments": int(ignore.sum()), "n_documents": len(doc_names), "var3": float(var[:3].sum()),
            "seconds": round(time.perf_counter()-t0, 1)
        },
        "redundancy": red, "sweep": sweep,
        "groups": [{"size": len(g), "members": [{"id": ids[i], "doc": doc_names[docs[i]], "text": _snip(texts[i])} for i in g]}
                   for g in sorted(groups, key=len, reverse=True)[:25]],
        "discordance": disc,
        "candidates": [{"key": _key(p), "a": p["a"], "b": p["b"], "sim": p["sim"], "cross_doc": p["cross_doc"], "numeric_diff": p["numeric_diff"], 
                        "to_judge": _key(p) in judge_key, "ta": _snip(texts[p["i"]]), "tb": _snip(texts[p["j"]])} for p in cand[:n_show]],
        "fragments_by_doc": dict(frag_by_doc.most_common(10)),
        "plot": {
            "points": [{"id": ids[i], "doc": doc_names[docs[i]], "x": float(C[i, 0]), "y": float(C[i, 1]), "z": float(C[i, 2]), "frag": bool(ignore[i]), 
                        "group": in_group.get(i, -1), "text": _snip(texts[i], 90)} for i in range(n)],
            "edges": [[int(i), int(j)] for i, j in edges[:3000]],
            "cands": [{"i": p["i"], "j": p["j"], "key": _key(p)} for p in plot_cands],
        },
    }
    return result, todo

