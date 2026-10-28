from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from collections import Counter
from pathlib import Path
from analysis.utils import read_jsonl, MODEL
from analysis.core import prepare_chunks
from analysis.cleaning_area import _embed_cache, _embedding_cache_key, _get_model
import numpy as np
import time
import re
import os

"""
    Analyze the thematic coverage and granularity of chunks.
    - Thematic coverage: clustering (K-means), evaluates the #cluster via silhouette and describes the groups with TF-IDF terms, cohesion, distribution by document and centroid coordinates.
    - Granularite and segmentation: measures lengths in characters, words, and tokens; 
        reports short chunks, snippets, text beyond the template's limit, and possible boundary anomalies, such as missing final punctuation or lowercase initial.
    
    Results: Aggregate summaries, statistics by document, data by graphs, and phase times. 
        Embeddings and latent projection can be reused from the shared cache with content analysis.
"""

os.environ.setdefault("MPLBACKEND", "Agg")
SENTENCE_ENDINGS = tuple('.!?…:;)]"\'»”’')


# (latent space)
def latente_space(chunks_path, texts, model_name=MODEL):
    cache_key = _embedding_cache_key(chunks_path)
    return _embed_cache(texts, cache_key, model_name)


# Coverage



# (termini tfidf)



# (k-means)

# Granularità

# (stats)
def distribution_stats(values):
    values = np.asarray(values, dtype=float)

    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0

    p5, p25, p50, p75, p95 = (float(value) for value in np.percentile(values, [5, 25, 50, 75, 95]))
    
    return {
        "mean": mean,
        "median": p50,
        "std": std,
        "cv": std / mean if mean else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "p5": p5, "p25": p25, "p75": p75, "p95": p95,
        "iqr": p75 - p25,
        "skew": (float(np.mean(((values - mean) / std) ** 3)) if std > 0 else 0.0),
    }


def token_counts(texts, words_counts, model=None):
    if model is None:
        model = _get_model()

    try:
        encoded = model.tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
        return np.array([len(tokens) for tokens in encoded]), int(model.max_seq_length), "tokenizer"
    except Exception:
        return np.ceil(words_counts * 1.3).astype(int), 128, "stima"
    
def starts_lowercase(text):
    fisrt_character = text.lstrip(" \t\r\n-•*·–—\"“(")[:1]
    return fisrt_character.isalpha() and fisrt_character.islower()

# (granularity, segmentation quality)
def analyze_granularity(texts, docs, doc_names, fragments, min_chars=40, short_chars=100, model=None):
    n = len(texts)
    char_counts = np.array([len(text) for text in texts])
    words_counts = np.array([len(re.findall(r"\w+", text)) for text in texts])
    token_counts_array, token_limit, token_source = token_counts(texts, words_counts, model)
 
    short = (char_counts < short_chars) & ~fragments
    over_limit = token_counts_array > token_limit
    
    no_final_punctuation = np.array([not text.rstrip().endswith(SENTENCE_ENDINGS) for text in texts]) & ~fragments
    lowercase_sart = np.array([starts_lowercase(text) for text in texts]) & ~fragments
    
    anomalous = fragments | short | over_limit | no_final_punctuation | lowercase_sart
 
    flags = [
        {"label": f"Fragments (< {min_chars} useful char)", "count": int(fragments.sum())},
        {"label": f"Short (< {short_chars} char)", "count": int(short.sum())},
        {"label": f"Beyond the limits of the model (> {token_limit} token)", "count": int(over_limit.sum())},
        {"label": "A punctuation-free ending", "count": int(no_final_punctuation.sum())},
        {"label": "Lower-case first letter", "count": int(lowercase_sart.sum())},
        {"label": "No anomalies", "count": int((~anomalous).sum())},
    ]
    for flag in flags:
        flag["share"] = flag["count"] / n
 
    # documenti con più chunk: tabella + boxplot
    document_totals = Counter(docs)

    per_document, boxplot_data = [], {}
    for doc_index, count in document_totals.most_common(15):
        mask = docs == doc_index
        doc_chars = char_counts[mask]
        doc_tokens = token_counts_array[mask]

        char_mean = float(doc_chars.mean()) if len(doc_chars) else 0.0
        char_std = float(doc_chars.std(ddof=1)) if len(doc_chars) > 1 else 0.0

        per_document.append({
            "doc": doc_names[doc_index], 
            "n": int(count), 
            "median_chars": float(np.median(doc_chars)),
            "cv": char_std / char_mean if char_mean else 0.0,
            "over": float(over_limit[mask].mean()), 
            "frag": float(fragments[mask].mean()),
            "anomalous": float(anomalous[mask].mean())
        })
        
        boxplot_data[doc_names[doc_index]] = [int(value) for value in doc_tokens]
 
    return {
        "summary": {
            "n": n, 
            "token_source": token_source, 
            "limit": token_limit, 
            "clean_share": float((~anomalous).mean()) if n else 0.0,
            "chars": distribution_stats(char_counts), 
            "words": distribution_stats(words_counts), 
            "tokens": distribution_stats(token_counts_array)
        },
        "flags": flags,
        "per_doc": per_document,
        "plot": {
            "tokens": [int(value) for value in token_counts_array],
            "chars": [int(value) for value in char_counts], 
            "limit": token_limit,
            "flags": flags, 
            "box": boxplot_data
        },
    }


# (group two)
def run_structure(chunks_path="out/chunks.jsonl", k=0, k_max=12, min_chars=40, short_chars=100, model_name=MODEL):
    started = time.perf_counter()
    chunks = read_jsonl(chunks_path)

    data = prepare_chunks(chunks, min_chars)
    data["min_chars"] = min_chars
    
    texts = data["texts"]
    docs = data["docs"]
    doc_names = data["doc_names"]
    fragments = data["fragments"]

    if len(texts) < 4:
        raise ValueError("group 2: at lest 4 chunks")

    X, C, explained_variance = latente_space(chunks_path, texts, model_name)
    embedding_time = time.perf_counter()

    coverage = analyze_coverage(X, C, texts, docs, doc_names, fragments, k, k_max)
    coverage_time = time.perf_counter()

    granularity = analyze_granularity(texts, docs, doc_names, fragments, min_chars, short_chars)
    granularity_time = time.perf_counter()

    mode = Path(chunks_path).stem.removeprefix("chunk_") 

    return {
        "params": {
            "path": mode,
            "k": k, 
            "k_max": k_max, 
            "min_chars": min_chars, 
            "short_chars": short_chars, 
            "model": model_name
        },
        "summary": {
            "n_chunks": len(texts), 
            "n_documents": len(doc_names), 
            "var3": float(explained_variance[:3].sum()),
            "seconds": round(granularity_time - started, 1),
            "t_embed": round(embedding_time - started, 1), 
            "t_cov": round(coverage_time - embedding_time, 1), 
            "t_gran": round(granularity_time - coverage_time, 1)
        },
        "coverage": coverage,
        "granularity": granularity,
        "plot": {
            "cov": coverage["plot"], 
            "gran": granularity["plot"]
        }
    }
