from collections import Counter
from pathlib import Path
from sentence_transformers import SentenceTransformer
from analysis.utils import words, numbers, position, read_jsonl, components, content_chars, JUDGE, MODEL
import urllib.request
import numpy as np
import hashlib
import json
import time

"""
    Central logic for measuring redundancies and possible discrepancies between chunks.
    - Preparation: Organize texts, documents, and locations; identify short fragments and construct comparable pairs, excluding adjacent fragments and chunks.
    - Similarity and redundancy: Generate embeddings, calculate cosine similarity, and group quasi-duplicates at one or more thresholds.
    - Candidates for discordance: Select sufficiently similar pairs and sort them by considering numerical differences and provenance from distinct documents.
    - Optional evaluation: Query Ollama to classify pairs, normalize returned relationships, and reuse valid judgments stored in the JSONL cache.
    - Results: Calculate summaries and metrics and save candidate pairs in JSONL.

    The module coordinates these phases and relies on the common functions of utils for reading, text analysis, and grouping.
"""

# (timing)
def mark_timing(label, started):
    now = time.perf_counter()
    print(f"[tempo] {label}: {now - started:.2f} s", flush=True)
    return now

# (embedding)
def generate_embeddings(texts, model_name=MODEL, batch_size=32):
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    
    print(f"[memo] load model {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"[memo] embedding for {len(texts)} chunk ...")
    started = time.perf_counter()
    X = np.asarray(model.encode(texts, batch_size=batch_size, show_progress_bar=True), dtype=np.float32)
    print(f"Embedding completed in {time.perf_counter() - started:.2f} s")

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-12)

def cosine_matrix(X):
    return X @ X.T

def compatible_pair(text_a, text_b, entities=None):
    if has_conflicting_city_entities(text_a, text_b, entities):
        return False
    return numbers(text_a) == numbers(text_b)

def compatible_mask(texts, I, J, S, min_sim, entities=None):
    mask = np.ones(len(S), dtype=bool)
    for k in np.flatnonzero(S >= min_sim):
        mask[k] = compatible_pair(texts[I[k]], texts[J[k]], entities)
    return mask


# (redundancy) Calcolo delle metriche a una data soglia
def redundancy_at(threshold, I, J, S, docs, n, n_eff=None):
    n_eff = n_eff if n_eff is not None else n

    keep = S >= threshold
    edges = list(zip(I[keep].tolist(), J[keep].tolist()))
    groups = components(n, edges)
    sizes = [len(group) for group in groups]

    # Occorrenze ridondanti rispetto al dato unico
    excess = sum(size - 1 for size in sizes)
    cross_doc = int(sum(docs[i] != docs[j] for i, j in edges))

    # Metriche, gruppi quasi duplicati, archi usati
    metrics = {"threshold": threshold, 
               "redundant_chunks": sum(sizes), 
               "redundant_chunk_share": sum(sizes) / n_eff if n_eff else 0.0,
                "n_groups": len(groups), 
                "max_group_size": max(sizes, default=0),
                "mean_group_size": float(np.mean(sizes)) if sizes else 0.0,
                "excess_units": excess, 
                "redundancy_rate": excess / n_eff if n_eff else 0.0,
                "pairs_cross_doc": cross_doc, 
                "pairs_same_doc": len(edges) - cross_doc}
    return metrics, groups, edges

# (metrics) Più soglie
def redundancy_sweep(thresholds, I, J, S, docs, n, n_eff=None):
    return [redundancy_at(threshold, I, J, S, docs, n, n_eff)[0] for threshold in thresholds]


# (preparation) Indici, documenti, posizioni, maschera dei frammenti
def prepare_chunks(chunks, min_chars=40):
    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    n = len(chunks)

    doc_names = sorted({chunk["doc_id"] for chunk in chunks})
    doc_to_index = {name: i for i, name in enumerate(doc_names)}

    docs = np.array([doc_to_index[chunk["doc_id"]] for chunk in chunks])
    positions = np.array([position(chunk["chunk_id"]) for chunk in chunks])

    fragments = np.array([content_chars(text) < min_chars for text in texts], dtype=bool)

    # Array per le analisi successive
    return {"texts": texts, "ids": ids, "n": n, "doc_names": doc_names, "docs": docs,
            "positions": positions, "fragments": fragments, "n_effective": int((~fragments).sum())}

# (pairs) Pair construction without: parsing fragments and adjacent chunks within the same document
def build_valid_pairs(data):
    n = data["n"]
    I, J = np.triu_indices(n, k=1)

    adjacent = (data["docs"][I] == data["docs"][J]) & (np.abs(data["positions"][I] - data["positions"][J]) == 1)
    
    fragment_pair = data["fragments"][I] | data["fragments"][J]
    valid = ~(adjacent | fragment_pair)

    return I[valid], J[valid]

# (knn)
def nearest_neighbor_sim(I, J, S, n):
    nn = np.full(n, -1.0, dtype=float)
    np.maximum.at(nn, I, S)
    np.maximum.at(nn, J, S)
    return nn

# Discordance

DEFAULT_ENTITIES = {"agra", "jaipur", "kanpur", "delhi", "mumbai", "kolkata", "nagpur", "surat", "bangladesh"}
def has_conflicting_city_entities(text_a, text_b, entities=None):
    entities = entities or DEFAULT_ENTITIES

    text_a = text_a.lower()
    text_b = text_b.lower()
    entities_a = {entity for entity in entities if entity in text_a}
    entities_b = {entity for entity in entities if entity in text_b}

    return bool(entities_a and entities_b and not (entities_a & entities_b))

def find_contradiction_candidates(texts, ids, docs, I, J, S, cand_threshold=0.85, city_entities=None):
    # Select pairs for analysis
    candidates = []
    for k in np.where(S >= cand_threshold)[0]:
        i, j = int(I[k]), int(J[k])

        if words(texts[i]) == words(texts[j]):
            continue
        if has_conflicting_city_entities(texts[i], texts[j], city_entities):
            continue

        numbers_i, numbers_j = numbers(texts[i]), numbers(texts[j])

        candidates.append({"i": i, "j": j, "a": ids[i], "b": ids[j], "sim": float(S[k]),
                           "cross_doc": bool(docs[i] != docs[j]), "numeric_diff": bool(numbers_i and numbers_j and numbers_i != numbers_j),})

    candidates.sort(key=lambda pair: (-pair["cross_doc"], -pair["numeric_diff"], -pair["sim"],))
    return candidates

# (stats)
def candidate_summary(candidates):
    return {
        "candidate_pairs": len(candidates),
        "candidate_numeric_diff": sum(pair["numeric_diff"] for pair in candidates),
        "candidates_cross_doc": sum(pair["cross_doc"] for pair in candidates),
    }

# (ollama)
def ollama_chat(model, system_prompt, user_prompt, timeout=1800):
    body = {
        "model": model, "stream": False, "format": "json", "keep_alive": "30m",
        "options": {
            "temperature": 0, "num_ctx": 4096
        },
        "messages": [{
            "role": "system", "content": system_prompt}, 
            {"role": "user", "content": user_prompt}
        ]
    }
    request = urllib.request.Request(
        "http://localhost:11434/api/chat", 
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["message"]["content"]

RELATION_ALIASES = {
    "equivalente": "equivalente", "equivalent": "equivalente",
    "contraddizione": "contraddizione", "contradiction": "contraddizione", "contradictory": "contraddizione",
    "complementare": "complementare", "complementary": "complementare",
    "non correlato": "non correlato", "unrelated": "non correlato",
}

# (llm) Normalization
def normalize_relation(value):
    # Relazione restituita dall'LLM
    value = str(value or "").strip().lower()
    return RELATION_ALIASES.get(value, "invalid")

# (llm) Extraction results 
def parse_judgment(raw):
    try:
        data = json.loads(raw)

        raw_relation = data.get("relazione", data.get("relation", ""))
        relation = normalize_relation(raw_relation)

        explanation = str(data.get("explanation", data.get("spiegazione", "")))
        return relation, explanation

    except Exception as exc:
        return "invalid", f"Errore parsing: {exc} | Raw: {raw[:150]}"

# (llm) Cache key
def judgment_key(model, text_a, text_b):
    payload = f"{model}|{text_a}|{text_b}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()

# (llm) Upload cache
def load_judgment_cache(path):
    path = Path(path)
    if not path.exists():
        return {}

    return {record["key"]: record for record in read_jsonl(path)}

# (llm) Evaluation
def judge_pairs(model, pairs, texts, ids, out_dir, system_prompt=JUDGE, on_result=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    judgements_path = out_dir / "judgements.jsonl"
    cache = load_judgment_cache(judgements_path)
    
    with judgements_path.open("a", encoding="utf-8") as output_file:
        for index, pair in enumerate(pairs, 1):
            a, b = pair["i"], pair["j"]

            key = judgment_key(model, texts[a], texts[b])
            elapsed = None
            
            if key in cache:
                result = cache[key]
            else:
                started = time.perf_counter()

                # Input per LLM
                user_prompt =(f"Testo A:\n{texts[a]}\n\n"
                              f"Testo B:\n{texts[b]}")
                
                raw = ollama_chat(model, system_prompt, user_prompt)
                relation, explanation = parse_judgment(raw)
                elapsed = time.perf_counter() - started

                result = {
                    "key": key, "a": ids[a], "b": ids[b], "relation": relation, "explanation": explanation
                }

                if relation != "invalid":
                    cache[key] = result
                    output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                    output_file.flush()

            pair["relation"] = result["relation"]
            pair["explanation"] = result["explanation"]

            if on_result:
                on_result(index, pair)

            duration = f"{elapsed:.2f} s" if elapsed is not None else " (cache)"
            print(f" giudizio {index}/{len(pairs)}: {pair['relation']} {duration}", flush=True)
    return pairs

# (llm) Final stats
def judged_summary(candidates):
    relations = Counter(pair["relation"] for pair in candidates if "relation" in pair)
    judged = sum(relations.values())

    return {
        "judged": judged,
        **{f"judged_{relation}": count for relation, count in relations.items()},
        "contradiction_rate_judged": relations.get("contraddizione", 0) / judged if judged else 0.0
    }

# (output)
def write_candidate_pairs(candidates, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / "candidate_pairs.jsonl"
    try:
        with path.open("w", encoding="utf-8") as fp:
            for pair in candidates:
                record = {
                    key: value for key, value in pair.items() if key not in ("i", "j")
                }
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    except PermissionError:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"candidate_pairs_{timestamp}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for pair in candidates:
                record = {
                    key: value for key, value in pair.items() if key not in ("i", "j")
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[warning] impossible rewrite candidate_pairs.jsonl; store in {path}", flush=True)
    return path

# (metrics) Dict to json
def build_metrics(model_name, data, redundancy_metrics, discordance_metrics):
    fragments = data["fragments"]
    n = data["n"]

    fragment_counter = Counter(data["doc_names"][doc_index] for doc_index in data["docs"][fragments])

    return {
        "embedder": model_name,
        "n_chunks": n,
        "n_chunks_analyzed": data["n_effective"],
        "n_documents": len(data["doc_names"]),
        "fragments": {
            "min_chars": data["min_chars"],
            "n_fragments": int(fragments.sum()),
            "fragment_share": float(fragments.mean()) if n else 0.0,
            "by_document": dict(fragment_counter.most_common(10)),
        },
        "redundancy": redundancy_metrics,
        "discordance": discordance_metrics,
    }

def coherence_metrics(judged_pairs, n_effective):
    valid = [p for p in judged_pairs if p.get("relation") not in (None, "invalid")]
    contra = [p for p in valid if p["relation"] == "contraddizione"]
    chunks_in_contra = {p["i"] for p in contra} | {p["j"] for p in contra}

    c_pairs = len(contra) / len(valid) if valid else None
    c_chunks = len(chunks_in_contra) / n_effective if n_effective else None

    return {
        "n_judged": len(valid),
        "n_invalid": len(judged_pairs) - len(valid),
        "n_contradictions": len(contra),
        "n_chunks_in_contradiction": len(chunks_in_contra),
        "C_pairs": c_pairs,
        "C_chunks": c_chunks,
        "coerenza": None if c_pairs is None else 1 - c_pairs,
        "coerenza_chunks": None if c_chunks is None else 1 - c_chunks
    }