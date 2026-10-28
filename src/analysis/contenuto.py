from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from collections import Counter
from pathlib import Path
from analysis.core import build_valid_pairs, candidate_summary, cosine_matrix, find_contradiction_candidates, nearest_neighbor_sim
from analysis.core import prepare_chunks, redundancy_at, redundancy_sweep, compatible_mask, coherence_metrics, DEFAULT_ENTITIES
from analysis.utils import read_jsonl, pca3, MODEL
import numpy as np
import time
import os

os.environ.setdefault("MPLBACKEND", "Agg")
DEFAULT_THRESHOLD = (0.80, 0.85, 0.90, 0.92, 0.95, 0.98, 0.99)
SHARE_FACTOR = 0.5

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

def _embedding_cache_key(chunks_path):
    path = Path(chunks_path)
    return str(path), path.stat().st_mtime

# (short text)
def _snippet(text, length=110):
    return str(text)[:length].replace("\n", " ")

# (candidate pair)
def _pair_key(pair):
    return f'{pair["a"]}|{pair["b"]}'

## Copertura

# (evaluate clusters with silhouette-inertia)
def scan_k(X, k_max=12, n_init=5, random_state=0):
    n = len(X)
    if n < 2:
        return []
    
    k_min = 3 if n >= 6 else 2
    k_upper = max(k_min, min(k_max, n - 1))
    sample_size = 2000 if n > 2000 else None
    rows = []

    for k in range(k_min, k_upper + 1):
        model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state).fit(X)
        silhouette = silhouette_score(X, model.labels_, sample_size=sample_size, random_state=random_state)
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette": float(silhouette)})
    return rows

def cluster_terms(texts, labels, k, n_terms=5):
    try:
        vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, token_pattern=r"(?u)\b[^\W\d_]{4,}\b")
        matrix = vectorizer.fit_transform(texts)
        vocabulary = np.array(vectorizer.get_feature_names_out())
        result = []

        for cluster in range(k):
            mask = labels == cluster
            if not mask.any():
                result.append([])
                continue
            
            mean_tfidf = np.asarray(matrix[mask].mean(axis=0)).ravel()
            top_indices = np.argsort(-mean_tfidf)[:n_terms]
        
            result.append([str(vocabulary[index]) for index in top_indices if mean_tfidf[index] > 0])
        return result
    
    except ValueError:
        return [[] for _ in range(k)]

def topic_coverage(X_valid, ids_valid, expected_topics, model_name=MODEL, topic_threshold=0.55):
    # Cp(a) = #argomenti attesi presenti nella base / #argomenti attesi per a
    # Argomento presente <=> Almeno un chunk con cos(arg, chunk) >= topic_thresh
    # expected_topics: {"area": ["arg1", "arg2", ...]}
    pairs = [(area, topic) for area, topics in expected_topics.items() for topic in topics]
    if not pairs:
        return None
 
    model = _get_model(model_name)
    T = np.asarray(model.encode([t for _, t in pairs], show_progress_bar=False), dtype=np.float32)
    T = T / np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
 
    sims = T @ X_valid.T
    best = sims.max(axis=1)
    best_idx = sims.argmax(axis=1)
 
    rows = [{"area": area, "topic": topic, "best_sim": float(best[k]),
             "best_chunk": ids_valid[int(best_idx[k])], 
             "covered": bool(best[k] >= topic_threshold)} for k, (area, topic) in enumerate(pairs)]
 
    per_area = {}
    for row in rows:
        entry = per_area.setdefault(row["area"], {"expected": 0, "covered": 0})
        entry["expected"] += 1
        entry["covered"] += int(row["covered"])
    for entry in per_area.values():
        entry["cp"] = entry["covered"] / entry["expected"]
 
    covered_total = sum(row["covered"] for row in rows)
    return {"cp_kb": covered_total / len(rows), 
            "n_expected": len(rows), "n_covered": covered_total,
            "topic_threshold": topic_threshold, "per_area": per_area, 
            "topics": rows, "missing": [row for row in rows if not row["covered"]]
    }

def analyze_coverage(X, C, texts, ids, docs, doc_names, fragments, k=0, k_max=12, min_share=None, expected_topics=None, model_name=MODEL, topic_threshold=0.55):
    valid_indices = np.where(~fragments)[0]
    n_analyzed = len(valid_indices)

    if n_analyzed < 4: 
        raise ValueError("Copertura: at least 4 chunks")
    
    X_valid = X[valid_indices]
    k_scan = scan_k(X_valid, k_max)
    best_k = max(k_scan, key=lambda item: item["silhouette"])["k"]
    
    auto_k = k <= 0
    selected_k = best_k if auto_k else max(2, min(k, n_analyzed - 1))

    model = KMeans(n_clusters=selected_k, n_init=10, random_state=0).fit(X_valid)
    labels, centers = model.labels_, model.cluster_centers_

    # Reorder the clusters in descending order by size
    order = np.argsort(-np.bincount(labels, minlength=selected_k))
    remap = np.empty(selected_k, dtype=int)
    remap[order] = np.arange(selected_k)
    labels, centers = remap[labels], centers[order]
 
    normalized_center = centers / np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    cohesion_per_chunk = (X_valid * normalized_center[labels]).sum(axis=1)      # Densità cluster
    terms = cluster_terms([texts[index] for index in valid_indices], labels, selected_k)
 
    sizes = np.bincount(labels, minlength=selected_k)
    shares = sizes / n_analyzed

    positive_shares = shares[shares > 0]
    balance = float(-(positive_shares * np.log(positive_shares)).sum() / np.log(selected_k)) if selected_k > 1 else 1.0
    
    t = min_share if min_share is not None else SHARE_FACTOR / selected_k
    multi_doc = len(doc_names) > 1

    cluster_rows = []
    for cluster in range(selected_k):
        mask = labels == cluster
        if not mask.any(): 
            continue
        
        document_count = Counter(docs[valid_indices[mask]])
        top_doc, top_doc_count = document_count.most_common(1)[0]
        
        cluster_rows.append({
            "cluster": cluster, 
            "name": f"C{cluster + 1}", 
            "terms": terms[cluster],
            "size": int(sizes[cluster]), 
            "share": float(shares[cluster]),
            "density": float(cohesion_per_chunk[mask].mean()),
            "n_docs": len(document_count), "top_doc": doc_names[top_doc],
            "top_doc_share": top_doc_count / int(sizes[cluster]),
            "single_doc": bool(multi_doc and len(document_count) == 1),
            "status": "coperta" if shares[cluster] >= t else "sottorappresentata"
        })
 
    silhouette = silhouette_score(X_valid, labels, sample_size=2000 if n_analyzed > 2000 else None, random_state=0)
    
    # heatmap cluster x documento
    document_totals = Counter(docs[valid_indices])
    top_docs = [doc for doc, _ in document_totals.most_common(30)]
    heatmap = [[
        int(np.sum((labels == cluster) & (docs[valid_indices] == doc))) for doc in top_docs] 
        for cluster in range(selected_k)
    ]
 
    # centroidi
    centroids_3d = []
    for cluster in range(selected_k):
        mask = labels == cluster
        if mask.any():
            cluster_coordinates = C[valid_indices[mask]]
            centroids_3d.append({
                "cluster": cluster,
                "x": float(cluster_coordinates[:, 0].mean()), 
                "y": float(cluster_coordinates[:, 1].mean()),
                "z": float(cluster_coordinates[:, 2].mean()),
            })

    topics = (topic_coverage(X_valid, [ids[i] for i in valid_indices], expected_topics, model_name, topic_threshold)
              if expected_topics else None)
 
    full_labels = np.full(len(texts), -1, dtype=int)
    full_labels[valid_indices] = labels

    def cluster_label(c):
        label_terms = ", ".join(terms[c][:3])
        return f"C{c + 1}" + (f": {label_terms}" if label_terms else "")
 
    return {
        "summary": {
            "k": selected_k, "k_auto": best_k, "auto": auto_k, "threshold": float(t),
            "n_analyzed": n_analyzed, "silhouette": float(silhouette), "balance": balance,
            "n_under": sum(r["status"] == "sottorappresentata" for r in cluster_rows),
            "n_single_doc": sum(r["single_doc"] for r in cluster_rows),
            "mean_density": float(np.mean([r["density"] for r in cluster_rows])),
            "cp_kb_topics": topics["cp_kb"] if topics else None},
        "areas": cluster_rows,
        "under": [r for r in cluster_rows if r["status"] == "sottorappresentata"],
        "topics": topics,
        "labels": full_labels,
        "plot": {
            "names": [cluster_label(c) for c in range(selected_k)], "centroids": centroids_3d,
            "sizes": [int(s) for s in sizes], "uniform": n_analyzed / selected_k,
            "scan": k_scan, "k": selected_k, 
            "heat": {
                "docs": [doc_names[d] for d in top_docs], "z": heatmap
            }},
    }

## Coerenza

def finalize_coherence(result, judged_pairs, texts):
    """Da chiamare quando il job LLM in background ha finito: aggiorna result['coerenza'] in place."""
    n_eff = result["summary"]["n_analyzed"]
    metrics = coherence_metrics(judged_pairs, n_eff)
    conflicts = [{
        "a": p["a"], "b": p["b"], "sim": p["sim"], "cross_doc": p["cross_doc"],
        "text_a": texts[p["i"]], "text_b": texts[p["j"]],
        "relation": p["relation"], "explanation": p.get("explanation", "")}
        for p in judged_pairs if p.get("relation") == "contraddizione"]
    conflicts.sort(key=lambda c: -c["sim"])
 
    result["coerenza"].update(metrics)
    result["coerenza"]["conflicts"] = conflicts
    result["coerenza"]["evaluated"] = True
    return result

## Entry point
def run_contents(chunks_path="out/chunks.jsonl", out_dir="out", threshold=0.92, cand_threshold=0.85, min_chars=40, k=0, 
                 k_max=12, min_share=None, expected_topics=None, topic_threshold=0.55, judge=None, max_pairs=10, max_show=50, model_name=MODEL):
    started = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # chunks, embedding, pca
    chunks = read_jsonl(chunks_path)
    data = prepare_chunks(chunks, min_chars=min_chars)
    data["min_chars"] = min_chars

    texts = data["texts"]
    ids = data["ids"]
    n = data["n"]
    docs = data["docs"]
    doc_names = data["doc_names"]
    fragments = data["fragments"]
    if data["n_effective"] < 4:
        raise ValueError("Contenuto: at least 4 chunks.")

    # (embedding, pca)
    X, C, explained_variance = _embed_cache(texts, _embedding_cache_key(chunks_path), model_name)    
    similarity = cosine_matrix(X)

    # (pairs, knn)
    I, J = build_valid_pairs(data)
    S = similarity[I, J]
    S_red = np.where(compatible_mask(texts, I, J, S, min(DEFAULT_THRESHOLD), DEFAULT_ENTITIES), S, -1.0)

    nearest = nearest_neighbor_sim(I, J, S, n)
    nn = nearest[nearest > -1]

    # Copertura
    coverage = analyze_coverage(X, C, texts, ids, docs, doc_names, fragments, k, k_max, min_share, expected_topics, model_name, topic_threshold)
    labels = coverage.pop("labels")

    # Ridondanza
    redundancy_metrics, groups, edges = redundancy_at(threshold, I, J, S_red, docs, n, data["n_effective"])
    sweep = redundancy_sweep(DEFAULT_THRESHOLD, I, J, S_red, docs, n, data["n_effective"])
    sweep_raw = redundancy_sweep(DEFAULT_THRESHOLD, I, J, S, docs, n, data["n_effective"])
    group_by_chunk = {c: g for g, group in enumerate(groups) for c in group}

    redundancy = {
        "R": redundancy_metrics["redundant_chunk_share"],   # R = #chunk in gruppi >= 2 / #chunk analizzati
        "metrics": redundancy_metrics, "sweep": sweep, "sweep_raw": sweep_raw,
        "nn": [float(v) for v in nn],
        "groups": [{"size": len(g), "members": [{"id": ids[c], "doc": doc_names[docs[c]], "text": _snippet(texts[c])} for c in g]}
                   for g in sorted(groups, key=len, reverse=True)[:25]]
    }


    # Coerenza
    candidates = find_contradiction_candidates(texts, ids, docs, I, J, S, cand_threshold, DEFAULT_ENTITIES)
    discordance_metrics = candidate_summary(candidates)
    
    pairs_to_judge = candidates[:max_pairs] if judge and candidates else []
    todo = {
        "model": judge, "pairs": pairs_to_judge, "texts": texts, "ids": ids, "out_dir": str(out_dir)
    } if pairs_to_judge else None

    keys_to_judge = {_pair_key(pair) for pair in pairs_to_judge}
    discordance_metrics["to_judge"] = len(pairs_to_judge)
    discordance_metrics["n_valid_pairs"] = int(len(I))

    # (data 4 ui)
    n_show = max(max_show, max_pairs)
    plot_candidates = candidates[:n_show]
    coherence = {
        "evaluated": False, "C_pairs": None, "C_chunks": None, "coerenza": None,
        "metrics": discordance_metrics, "conflicts": [],
        "candidates": [{
            "key": _pair_key(p), "a": p["a"], "b": p["b"], "sim": p["sim"], "cross_doc": p["cross_doc"],
            "numeric_diff": p["numeric_diff"], "to_judge": _pair_key(p) in keys_to_judge,
            "ta": _snippet(texts[p["i"]]), "tb": _snippet(texts[p["j"]])} for p in candidates[:n_show]]
    }
    mode = Path(chunks_path).stem.removeprefix("chunks_") 

    result = {
        "params": {
            "path": mode,
            "threshold": threshold, 
            "cand_threshold": cand_threshold, 
            "min_chars": min_chars, 
            "k": k,
            "min_share": coverage["summary"]["threshold"],
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
        "copertura": coverage,
        "ridondanza": redundancy, 
        "coerenza": coherence,
        "plot": {
            "points": [{
                "id": ids[index], "doc": doc_names[docs[index]], 
                "x": float(C[index, 0]), "y": float(C[index, 1]), "z": float(C[index, 2]), 
                "frag": bool(fragments[index]), "cluster": int(labels[index]),
                "group": group_by_chunk.get(index, -1), 
                "text": _snippet(texts[index], 90)} 
                for index in range(n)
            ],
            "edges": [[
                int(i), int(j)] for i, j in edges[:3000]
            ],
            "cands": [{
                "i": pair["i"], "j": pair["j"], "key": _pair_key(pair)
            } for pair in plot_candidates],
            "cov": coverage["plot"]
        },
    }
    return result, todo


def get_report(result):
    s, cov, red, coe = result["summary"], result["copertura"], result["ridondanza"], result["coerenza"]
    pct = lambda v: "n/d" if v is None else f"{v:.1%}"

    print("-" * 70)
    print(f"Gruppo 1 - Contenuto | {s['n_analyzed']}/{s['n_chunks']} chunk analizzati, {s['n_documents']} documenti, {s['seconds']} s")
 
    print("\n[1] Copertura Tematica")
    cs = cov["summary"]
    print(f"  k = {cs['k']}{' (auto)' if cs['auto'] else ''} | soglia t = {cs['threshold']:.1%} | silhouette = {cs['silhouette']:.3f} | aree sottorappresentate: {cs['n_under']}")
    print(f"  {'area':<6}{'#chunk':>8}{'% tot':>9}{'densita':>9} stato")
    for a in cov["areas"]:
        print(f"  {a['name']:<6}{a['size']:>8}{a['share']:>9.1%}{a['density']:>9.3f} {a['status']} [{', '.join(a['terms'][:3])}]")
    if cov["topics"]:
        t = cov["topics"]
        print(f"  Cp(kb) argomenti attesi: {t['n_covered']}/{t['n_expected']} = {pct(t['cp_kb'])}")
        for area, e in t["per_area"].items():
            print(f"  {area}: {e['covered']}/{e['expected']} = {pct(e['cp'])}")
        for m in t["missing"]:
            print(f"  mancante [{m['area']}] {m['topic']} (best cos {m['best_sim']:.2f})")
 
    print("\n[2] Ridondanza")
    m = red["metrics"]
    print(f"  R = {pct(red['R'])} a soglia {m['threshold']:.2f} | gruppi: {m['n_groups']} | max {m['max_group_size']} | chunk ridondanti: {m['redundant_chunk_count'] if 'redundant_chunk_count' in m else m['redundant_chunks']}")
    for g in red["groups"][:5]:
        print(f" gruppo da {g['size']}:")
        for mem in g["members"][:4]:
            print(f"  - {mem['id']}: {mem['text'][:70]}")
 
    print("\n[3] Coerenza Informativa")
    d = coe["metrics"]
    print(f"  candidati: {d['candidate_pairs']} (cross-doc {d['candidates_cross_doc']}, diff. numerica {d['candidate_numeric_diff']}) | da giudicare: {d['to_judge']}")
    if not coe["evaluated"]:
        print("  giudizio LLM non ancora disponibile: C e Coerenza = n/d")
    else:
        print(f"  giudicate: {coe['n_judged']} | contraddittorie: {coe['n_contradictions']}")
        print(f"  C (coppie) = {pct(coe['C_pairs'])} | C (chunk) = {pct(coe['C_chunks'])} | Coerenza = 1 - C = {pct(coe['coerenza'])}")
        for c in coe["conflicts"][:5]:
            print(f" CONFLITTO {c['a']} <> {c['b']} (cos {c['sim']:.3f}): {c['explanation'][:100]}")
    print("-" * 70)
