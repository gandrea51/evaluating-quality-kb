from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import json
import re

"""
    Shared utilities for chunk analysis and visualization.

    - LLM Text and Judgment: Defines the comparison prompt and functions to extract meaningful words, numbers, position, and characters from chunks.
    - Input: Reads JSONL records and reports malformed or incorrectly structured rows; also converts parameters received from web forms.
    - Structure and visualization: Group linked indexes into components, project embeddings into three dimensions with PCA, and generate analysis graphs.

    Functions are reused by analysis modules to maintain consistent data preparation, checks, and result representation.
"""

# (llm) Prompt
JUDGE = (
    "Sei un assistente IA che analizza i documenti aziendali. Il tuo compito è confrontare due testi e stabilire la loro relazione. "
    "Rispondi TASSATIVAMENTE ed ESCLUSIVAMENTE con un oggetto JSON valido, senza testo introduttivo o conclusivo. "
    "Il JSON deve contenere solo ed esclusivamente queste due chiavi: 'relazione' e 'explanation'.\n\n"
    "I valori ammessi per la chiave 'relazione' sono soltanto:\n"
    "- 'equivalente': i due testi esprimono lo stesso identico concetto.\n"
    "- 'contraddizione': i due testi contengono valori, cifre o affermazioni incompatibili sullo stesso argomento (es. tassi di interesse diversi).\n"
    "- 'complementare': parlano dello stesso argomento ma forniscono informazioni diverse e compatibili.\n"
    "- 'non correlato': parlano di argomenti completamente diversi.\n\n"
    "Esempio di output richiesto:\n"
    '{"relazione": "contraddizione", "explanation": "Il testo A riporta un tasso del 3%, mentre il testo B indica il 4% per lo stesso mutuo."}'
)

# (embedding) Hugging Face
MODEL = "BAAI/bge-m3"

# Utility

def words(s):
    # Extraction of lower-case words for text analysis
    if s is None:
        return []
    return re.findall(r"\w+", str(s).lower())

def numbers(s):
    # Selection of numbers for checks on quantity discrepancies
    if s is None:
        return set()
    
    pattern = r"(?<!\w)[+-]?\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?"
    return set(re.findall(pattern, str(s)))

def position(chunk_id):
    # I retrieve the sequential chunk index in the document
    if chunk_id is None:
        return 0
    
    match = re.search(r"::(\d+)$", chunk_id)
    return int(match.group(1)) if match else 0


def content_chars(text):
    # I count the actual alphanumeric characters
    if text is None:
        return 0
    
    return len(re.sub(r"[\W_]+", "", str(text)))


# I/O
def read_jsonl(path):
    records = []

    with open(path, encoding="utf-8-sig") as f:        # utf-8-sig: ignora il BOM
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}, riga {n}: JSON non valido ({e.msg}, colonna {e.colno})")
            if not isinstance(record, dict):
                raise ValueError(f"{path}, riga {n}: atteso un oggetto JSON, trovato {type(record).__name__}")
            records.append(record)
    return records

# (graph) Componenti connesse per gruppare quasi-duplicati
# Return: gruppi con almeno due componenti
def components(n, edges):
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 0:
        raise ValueError("n: non-negative integer.")

    n = int(n)
    parent = list(range(n))
    size = [1] * n

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b): 
        root_a, root_b = find(a), find(b)
        if root_a == root_b: 
            return

        # Union by size
        if size[root_a] < size[root_b]:
            root_a, root_b = root_b, root_a

        parent[root_b] = root_a
        size[root_a] += size[root_b]

    for edge in edges:
        try: 
            i, j = edge
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid arc: {edge!r}. Attend pair (i, j).") from exc

        if not isinstance(i, (int, np.integer)) or isinstance(i, bool) or not isinstance(j, (int, np.integer)) or isinstance(j, bool):
            raise ValueError(f"Arc indices: integers: {edge!r}.")

        i, j = int(i), int(j)
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError(f"Out-of-bounds arc, n={n}: {edge!r}.")

        union(int(i), int(j))

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [group for group in sorted(groups.values(), key=lambda g: g[0]) if len(group) > 1]


# (pca) Dimensionality reduction
def pca3(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X: bidimensional matrix.")

    n_samples, n_features = X.shape

    if n_samples == 0:
        raise ValueError("PCA impossible: X without samples.")
    if n_features == 0:
        raise ValueError("PCA impossible: X without features.")
    if not np.isfinite(X).all():
        raise ValueError("X with NaN or infinite values.")

    X_centered = X - X.mean(axis=0)

    if n_samples < 2:
        return np.zeros((n_samples, 3)), np.zeros(3)
    
    _, singular_values, vt = np.linalg.svd(X_centered, full_matrices=False)

    variance = singular_values ** 2
    total_variance = variance.sum()
    explained_variance = variance / total_variance if total_variance > 0 else np.zeros_like(variance)

        
    k = min(3, vt.shape[0])
    component_3d = X_centered @ vt[:k].T
    if k < 3:
        component_3d = np.pad(component_3d, ((0, 0), (0, 3 - k)))

    variance_3d = np.zeros(3)
    variance_3d[:min(3, len(explained_variance))] = explained_variance[:3]
    return component_3d, variance_3d


# (plot) Spazio latente
def plot_3d(C, var, docs, red_groups, red_edges, cand_pairs, thr, path, show=False, frag_mask=None):
    C = np.asarray(C, dtype=float)
    var = np.asarray(var, dtype=float)
    docs = list(docs)

    if C.ndim != 2 or C.shape[1] != 3:
        raise ValueError("C: to have a shape (num_chunk, 3).")

    n = len(C)
    if len(docs) != n:
        raise ValueError("docs: at least one element per chunk.")
    if var.ndim != 1 or len(var) < 3:
        raise ValueError("var: at least three values.")
    if not np.isfinite(C).all() or not np.isfinite(var).all():
        raise ValueError("C, var: to not have Nan or infinites.")
    
    cmap = matplotlib.colormaps["tab20"]
    doc_keys = list(dict.fromkeys(docs))
    doc_colors = {key: cmap(i % cmap.N) for i, key in enumerate(doc_keys)}
    colors = [doc_colors[doc] for doc in docs]

    in_group = np.zeros(n, dtype=bool)
    for group in red_groups:
        indices = np.asarray(group, dtype=int)
        if np.any((indices < 0) | (indices >= n)):
            raise IndexError("A redundant group contains out-of-bounds indices.")
        
        in_group[indices] = True
    
    if frag_mask is None:
        frag = np.zeros(n, dtype=bool)
    else:
        frag = np.asarray(frag_mask, dtype=bool)
        if frag.shape != (n,):
            raise ValueError("frag_mask: a value for each chunk.")

    in_group &= ~frag
    plain = ~(in_group | frag)
    fig = plt.figure(figsize=(11, 8.5))

    try: 
        ax = fig.add_subplot(111, projection="3d")
        if plain.any():
            idx = np.flatnonzero(plain)
            ax.scatter(C[idx, 0], C[idx, 1], C[idx, 2], c=[colors[i] for i in idx], s=28, alpha=0.75)

        if in_group.any():
            idx = np.flatnonzero(in_group)
            ax.scatter(C[idx, 0], C[idx, 1], C[idx, 2], c=[colors[i] for i in idx], s=90, edgecolors="black", linewidths=1.4, alpha=0.95)

        if frag.any():
            idx = np.flatnonzero(frag)
            ax.scatter(C[idx, 0], C[idx, 1], C[idx, 2], c="gray", marker="x", s=35, alpha=0.8)

        for i, j in list(red_edges)[:300]:
            if not (0 <= i < n and 0 <= j < n):
                continue
            ax.plot([C[i, 0], C[j, 0]], [C[i, 1], C[j, 1]], [C[i, 2], C[j, 2]], color="black", alpha=0.35, linewidth=1)

        for pair in list(cand_pairs)[:200]:
            i, j = pair.get("i"), pair.get("j")
            relation = pair.get("relation")
            
            if not isinstance(i, (int, np.integer)) or not isinstance(j, (int, np.integer)) or not (0 <= i < n and 0 <= j < n):
                continue

            is_contradiction = relation == "contraddizione"

            ax.plot([C[i, 0], C[j, 0]], [C[i, 1], C[j, 1]], [C[i, 2], C[j, 2]], color="red" if is_contradiction else "orange", 
                    alpha=0.9 if is_contradiction else 0.45,
                    linewidth=2.4 if is_contradiction else 1, 
                    linestyle="-" if is_contradiction else "--")

        ax.set_xlabel(f"PC1 ({var[0]:.1%})")
        ax.set_ylabel(f"PC2 ({var[1]:.1%})")
        ax.set_zlabel(f"PC3 ({var[2]:.1%})")
        
        ax.set_title(f"Latent space of chunks (PCA 3D, "
                     f"{var[:3].sum():.1%} of the variance)\n"
                     f"Grouping chunks into near-duplicates: cosine >= {thr:.2f})")
    
        handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=9, label="Quasi-duplicate group"),
            Line2D([0], [0], marker="x", color="gray", linestyle="", markersize=7, label="PDF fragment excluded"),
            Line2D([0], [0], color="orange", linestyle="--", label="Candidate couple"),
            Line2D([0], [0], color="red", linewidth=2.4, label="Labeled contradiction")
        ]
            
        if len(doc_keys) <= 12:
            handles.extend(Line2D([0], [0], marker="o", color="w", markerfacecolor=doc_colors[key], markersize=8, label=str(key)[:28]) for key in doc_keys)

        ax.legend(handles=handles, loc="upper left", fontsize=7)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        if show:
            plt.show()
    finally:
        plt.close(fig)

# (plot) Calibrazione delle soglie di similarità
def plot_hist(nn, thr, cand_thr, path, show=False):
    nn = np.asarray(nn, dtype=float).ravel()
    
    if not np.isfinite(nn).all():
        raise ValueError("nn: contains Nan o infinite value.")
    if not np.isfinite(thr) or not np.isfinite(cand_thr):
        raise ValueError("threshold: finite numerical values.")
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    try: 
        if len(nn):
            ax.hist(nn, bins=40, color="steelblue", edgecolor="white")
        else:
            ax.text(0.5, 0.5, "No available data", transform=ax.transAxes, ha="center", va="center")

        ax.axvline(thr, color="black", linestyle="-", label=f"near-duplicate threshold ({thr:.2f})")
        ax.axvline(cand_thr, color="orange", linestyle="--", label=f"candidate threshold ({cand_thr:.2f})")
        ax.set_xlabel("Cosine similarity with the most similar chunk (excluding the document itself)")
        ax.set_ylabel("Number of chunks")
        ax.legend()
        ax.set_title("Calibration of thresholds in latent space PDF")

        fig.tight_layout()
        fig.savefig(path, dpi=150)
        if show:
            plt.show()
    finally:
        plt.close(fig)


# (read: int)
def get_int(form, name, default):
    raw = form.get(name, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default

# (read: float)
def get_float(form, name, default):
    raw = form.get(name, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default
