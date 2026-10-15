from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import json
import re

# Estrazione parole in minuscolo per analisi testuale
def words(s):
    return re.findall(r"\w+", str(s).lower())

# Estrazione numeri per controlli di discordanza quantitativa
def numbers(s):
    return set(re.findall(r"\d+(?:[.,]\d+)*", str(s)))

# Recupero l'indice progressivo del chunk nel documento
def position(chunk_id):
    m = re.search(r"::(\d+)$", chunk_id)
    return int(m.group(1)) if m else 0

# Lettura file dei chunks
def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

# Conto i caratteri alfanumerici reali
def content_chars(text):
    return len(re.sub(r"[\W_]+", "", str(text)))

# Componenti connessi per raggruppare i quasi-duplicati
def components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for i, j in edges:
        parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]

# Dimensionality reduction
def pca3(X):
    Xc = X - X.mean(axis=0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S**2) / (S**2).sum()
    k = min(3, Vt.shape[0])
    C = Xc @ Vt[:k].T
    if k < 3:
        C = np.hstack([C, np.zeros((len(C), 3 - k))])
    return C, var

# Grafico dello spazio latente
def plot_3d(C, var, docs, doc_names, red_groups, red_edges, cand_pairs, thr, path, show, frag_mask=None):
    if not show: matplotlib.use("Agg")

    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")
    colors = [cmap(d % 20) for d in docs]

    in_group = np.zeros(len(C), dtype=bool)
    for g in red_groups:
        in_group[g] = True
    frag = np.zeros(len(C), dtype=bool) if frag_mask is None else np.asarray(frag_mask, dtype=bool)

    # Chunk standard
    plain = np.where(~in_group & ~frag)[0]
    ax.scatter(*C[plain].T, c=[colors[i] for i in plain], s=28, alpha=0.75)

    # Chunk di un gruppo ridondante
    if in_group.any():
        ax.scatter(*C[in_group].T, c=[colors[i] for i in np.where(in_group)[0]], s=90,
                   edgecolors="black", linewidths=1.6, alpha=0.95)
    
    # Frammenti spuri (es: artefatti pdf senza reale contenuto)
    if frag.any():
        ax.scatter(*C[frag].T, c="gray", marker="x", s=30, alpha=0.7)

    # Linee di connessione tra i quasi-duplicati
    for i, j in red_edges[:300]:
        ax.plot(*zip(C[i], C[j]), color="black", alpha=0.35, lw=1)

    # Linne per coppie candidate alla discordanza-contraddizione
    for p in cand_pairs[:200]:
        rel = p.get("relation")
        col = {"contradiction": "red", "equivalent": "black"}.get(rel, "orange")
        ax.plot(*zip(C[p["i"]], C[p["j"]]), color=col, alpha=0.9 if rel == "contradiction" else 0.45,
                lw=2.4 if rel == "contradiction" else 1, ls="-" if rel == "contradiction" else "--")

    ax.set_xlabel(f"PC1 ({var[0]:.0%})")
    ax.set_ylabel(f"PC2 ({var[1]:.0%})")
    ax.set_zlabel(f"PC3 ({var[2]:.0%})" if len(var) > 2 else "PC3")
    ax.set_title(f"Spazio latente dei chunk (PCA 3D, {var[:3].sum():.0%} della varianza)\n"
                 f"cerchi neri = chunk in gruppi di quasi-duplicati (coseno >= {thr:.2f})")
    
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black", markersize=10, label="in gruppo di quasi-duplicati"),
               Line2D([0], [0], marker="x", color="gray", ls="", markersize=7, label="frammento pdf (escluso)"),
               Line2D([0], [0], color="orange", ls="--", label="coppia candidata alla discordanza"),
               Line2D([0], [0], color="red", lw=2.4, label="contraddizione confermata")]
    
    if len(doc_names) <= 12:
        handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(i % 20), markersize=8, label=n[:28]) for i, n in enumerate(doc_names)]

    ax.legend(handles=handles, loc="upper left", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

# Calibrazione delle soglie di similarità
def plot_hist(nn, thr, cand_thr, path):
    matplotlib.use("Agg") if not matplotlib.get_backend().lower().startswith("tk") else None
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.hist(nn, bins=40, color="steelblue", edgecolor="white")
    ax.axvline(thr, color="black", ls="-", label=f"soglia quasi-duplicati ({thr:.2f})")
    ax.axvline(cand_thr, color="orange", ls="--", label=f"soglia candidati ({cand_thr:.2f})")
    ax.set_xlabel("similarita' coseno con il chunk piu' simile (escluso lo stesso documento)")
    ax.set_ylabel("numero di chunk")
    ax.legend()
    ax.set_title("Calibrazione delle soglie sullo spazio latente pdf")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)