from collections import Counter
from sentence_transformers import SentenceTransformer
from utils import words, numbers, position, read_jsonl, components, content_chars
from utils import pca3, plot_3d, plot_hist
from pathlib import Path
import urllib.request
import numpy as np
import argparse
import hashlib
import json
import time
import re

# Prompt per LLM Judge
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


# Modello Hugging Face
MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

def mark_timing(label, started):
    now = time.perf_counter()
    print(f"[tempo] {label}: {now - started:.2f} s", flush=True)
    return now

# Coppie candidate per analisi di discordanza
def write_candidate_pairs(candidates, out_dir):
    path = out_dir/"candidate_pairs.jsonl"
    try:
        with open(path, "w", encoding="utf-8") as f:
            for pair in candidates:
                f.write(json.dumps({k: v for k, v in pair.items() if k not in ("i", "j")}, ensure_ascii=False) + "\n")
    except PermissionError:
        path = out_dir / f"candidate_pairs_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for pair in candidates:
                f.write(json.dumps({k: v for k, v in pair.items() if k not in ("i", "j")}, ensure_ascii=False) + "\n")
        print(f"[warning] impossibile sovrascrivere candidate_pairs.jsonl; salvato in {path}", flush=True)
    return path

# Normalizzazione L2 degli embedding con Sent. Transformer
def generate_embeddings(texts):
    print(f"[memo] caricamento modello {MODEL}...")
    model = SentenceTransformer(MODEL)

    print(f"[memo] generazione embedding per {len(texts)} chunk estratti...")
    batch_started = time.perf_counter()
    X = np.asarray(model.encode(texts, batch_size=32, show_progress_bar=True), dtype=np.float32)
    print(f"Embedding completati in {time.perf_counter()-batch_started:.2f} s")

    # Normalizzazione L2 per cosine similarity (prodotto cartesiano)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

# Ridondanza sulla soglia t
def redundancy_at(t, I, J, S, docs, n, n_eff=None):
    n_eff = n_eff or n
    keep = S >= t
    edges = list(zip(I[keep].tolist(), J[keep].tolist()))
    groups = components(n, edges)
    sizes = [len(g) for g in groups]

    # Occorrenze ridondanti rispetto al dato unico
    excess = sum(s - 1 for s in sizes)
    cross = int(sum(docs[i] != docs[j] for i, j in edges))
    return {"threshold": t, "redundant_chunks": sum(sizes), "redundant_chunk_share": sum(sizes) / n_eff,
            "n_groups": len(groups), "max_group_size": max(sizes, default=0),
            "mean_group_size": float(np.mean(sizes)) if sizes else 0.0,
            "excess_units": excess, "redundancy_rate": excess / n_eff,
            "pairs_cross_doc": cross, "pairs_same_doc": len(edges) - cross}, groups, edges

# Ollama per analizzare i due chunk estratti
def ollama_chat(model, prompt, user):
    body = {
        "model": model, "stream": False, "format": "json", "keep_alive": "30m",
        "options": {"temperature": 0, "num_ctx": 4096},
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": user}]}
    req = urllib.request.Request(
        "http://localhost:11434/api/chat", 
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=1800))["message"]["content"]

# Valutazione coppie candidate per capire se sono: ridondanze o discordanze reali
def judge_pairs(model, pairs, texts, ids, out_dir, on_result=None):
    path = Path(out_dir)/"judgments.jsonl"
    done = {r["key"]: r for r in read_jsonl(path)} if path.exists() else {}
    
    with open(path, "a", encoding="utf-8") as fp:
        for n, p in enumerate(pairs, 1):
            a, b = p["i"], p["j"]

            # Chiave univoca sul testo dei due chunk
            key = hashlib.sha1(f"{model}|{texts[a]}|{texts[b]}".encode()).hexdigest()
            elapsed = None
            
            if key not in done:
                started = time.perf_counter()

                # Input per LLM
                user_input = f"Testo A:\n{texts[a]}\n\nTesto B:\n{texts[b]}"
                raw = ollama_chat(model, JUDGE, user_input)
                elapsed = time.perf_counter() - started
                
                try:
                    d = json.loads(raw)
                    
                    # Estrai il valore cercando sia la chiave italiana che quella inglese
                    val_relazione = d.get("relazione", d.get("relation", "")).strip().lower()
                    
                    # Normalizzazione dei valori (se risponde in inglese o al plurale/singolare)
                    if val_relazione in ("equivalente", "equivalent"):
                        rel = "equivalente"
                    elif val_relazione in ("contraddizione", "contradiction", "contradictory"):
                        rel = "contraddizione"
                    elif val_relazione in ("complementare", "complementary"):
                        rel = "complementare"
                    elif val_relazione in ("non correlato", "unrelated"):
                        rel = "non correlato"
                    else:
                        rel = "invalid"
                        
                    # Cerca l'explanation o la spiegazione
                    expl = str(d.get("explanation", d.get("spiegazione", d.get("explanation", ""))))
                    
                except Exception as e:
                    rel, expl = "invalid", f"Errore parsing: {str(e)} | Raw: {raw[:150]}"


                # Salvataggio su file
                done[key] = {"key": key, "a": ids[a], "b": ids[b], "relation": rel, "explanation": expl}
                fp.write(json.dumps(done[key], ensure_ascii=False) + "\n"); fp.flush()
            
            p["relation"], p["explanation"] = done[key]["relation"], done[key]["explanation"]
            if on_result: on_result(n, p)
            duration = f" ({elapsed:.2f} s)" if elapsed is not None else " (cache)"
            print(f"  giudizio {n}/{len(pairs)}: {p['relation']}{duration}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="out/chunks.jsonl")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--threshold", type=float, default=0.92, help="coseno oltre il quale due chunk sono quasi-duplicati")
    ap.add_argument("--cand-threshold", type=float, default=0.85, help="coseno minimo per le coppie candidate alla discordanza")
    ap.add_argument("--min-chars", type=int, default=40, help="i chunk con meno di tanti caratteri utili (lettere e cifre) sono frammenti: esclusi dalle metriche e contati a parte")
    ap.add_argument("--judge", help="modello Ollama che verifica le coppie candidate (es. llama3.1)")
    ap.add_argument("--max-pairs", type=int, default=20, help="quante coppie candidate far giudicare")
    ap.add_argument("--show", action="store_true", help="apri il grafico 3D interattivo")
    a = ap.parse_args()
    
    run_started = time.perf_counter()
    phase_started = run_started
    
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    
    # 1. Dati estratti dai pdf
    chunks = read_jsonl(a.chunks)
    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    n = len(chunks)
    
    doc_names = sorted({c["doc_id"] for c in chunks})
    docs = np.array([doc_names.index(c["doc_id"]) for c in chunks])
    pos = np.array([position(c["chunk_id"]) for c in chunks])
    
    # Frammenti spuri (es. rumore di parsing pdf)
    ignore = np.array([content_chars(t) < a.min_chars for t in texts])
    n_eff = int((~ignore).sum())
    frag_by_doc = Counter(doc_names[d] for d in docs[ignore])

    phase_started = mark_timing("lettura e preparazione chunk", phase_started)
    
    # 2. Embedding (Sent. Transformer hf)
    X = generate_embeddings(texts)
    phase_started = mark_timing("embedding", phase_started)

    # Cosine similarity: prodotto scalare (già l2 norm.)
    sim = X @ X.T
    phase_started = mark_timing("matrice di similarita", phase_started)

    # 3. Analisi coppie nello spazio latente
    I, J = np.triu_indices(n, 1)

    # Esclusioni: chunk adiacenti/consecutivi nello stesso doc, frammenti vuoti
    adjacent = (docs[I] == docs[J]) & (np.abs(pos[I] - pos[J]) == 1)
    valid = ~adjacent & ~(ignore[I] | ignore[J])
    I, J = I[valid], J[valid]
    S = sim[I, J]

    nn = np.full(n, -1.0)
    np.maximum.at(nn, I, S)
    np.maximum.at(nn, J, S)
    phase_started = mark_timing("generazione coppie e similarita vicine", phase_started)
    
    # 4. Metriche di Ridondanza
    red, groups, edges = redundancy_at(a.threshold, I, J, S, docs, n, n_eff)
    sweep = [redundancy_at(t, I, J, S, docs, n, n_eff)[0] for t in (0.80, 0.85, 0.90, 0.92, 0.95, 0.98, 0.99)]
    phase_started = mark_timing("metriche di ridondanza", phase_started)
    
    # 5. Screen dei candidati alla discordanza
    cand = []
    for k in np.where(S >= a.cand_threshold)[0]:
        i, j = int(I[k]), int(J[k])
        if words(texts[i]) == words(texts[j]):
            continue
            
        # Filtro entità-città per evitare falsi positivi
        text_i_lower = texts[i].lower()
        text_j_lower = texts[j].lower()
    
        # Lista delle città/entità del dataset ZX Bank
        citta = ["agra", "jaipur", "kanpur", "delhi", "mumbai", "kolkata", "nagpur", "surat", "bangladesh"]
        
        # Se i due chunk contengono città esplicite diverse, salta la coppia
        citta_in_i = {c for c in citta if c in text_i_lower}
        citta_in_j = {c for c in citta if c in text_j_lower}
        
        # Se entrambi contengono città ma non c'è nessuna intersezione
        if citta_in_i and citta_in_j and not (citta_in_i & citta_in_j): continue

        ni, nj = numbers(texts[i]), numbers(texts[j])
        cand.append({
            "i": i, "j": j, "a": ids[i], "b": ids[j], "sim": float(S[k]),
            "cross_doc": bool(docs[i] != docs[j]), 
            "numeric_diff": bool(ni and nj and ni != nj)
        })

    
    # Ordinamento candidati: priorità a doc diversi con differenze numeriche
    cand.sort(key=lambda p: (-p["cross_doc"], -p["numeric_diff"], -p["sim"]))

    disc = {
        "candidate_pairs": len(cand), 
        "candidate_numeric_diff": sum(p["numeric_diff"] for p in cand),
        "candidates_cross_doc": sum(p["cross_doc"] for p in cand)}

    # 6. Validazione llm
    if a.judge and cand:
        judge_pairs(a.judge, cand[:a.max_pairs], texts, ids, od)
        rel = Counter(p["relation"] for p in cand[:a.max_pairs])
        m = sum(rel.values())
        disc.update({
            "judged": m, 
            **{f"judged_{k}": v for k, v in rel.items()},
            "contradiction_rate_judged": rel.get("contradiction", 0) / m if m else 0.0})
    
    # 7. Salvataggio e report strutturato
    fragments = {
        "min_chars": a.min_chars, "n_fragments": int(ignore.sum()),
        "fragment_share": float(ignore.mean()) if n else 0.0,
        "by_document": dict(frag_by_doc.most_common(10))
    }
    metrics = {
        "embedder": MODEL, "n_chunks": n, 
        "n_chunks_analyzed": n_eff,
        "n_documents": len(doc_names), 
        "fragments": fragments, 
        "redundancy": red, "discordance": disc
    }

    (od / "embedding_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    candidate_path = write_candidate_pairs(cand, od)
    phase_started = mark_timing("discordanze, giudizi e salvataggio metriche", phase_started)
    
    # Generazione report testuale
    snip = lambda i: texts[i][:110].replace("\n", " ")

    L = [
        json.dumps(metrics, indent=2), "", 
        "== calibrazione della soglia (ridondanza al variare del coseno) ==",
        f"{'soglia':>7} {'chunk in gruppi':>16} {'gruppi':>7} {'max':>4} {'rate':>7} {'coppie tra doc':>15}"
    ]
    L += [f"{r['threshold']:>7.2f} {r['redundant_chunks']:>16} {r['n_groups']:>7} {r['max_group_size']:>4} "
          f"{r['redundancy_rate']:>7.3f} {r['pairs_cross_doc']:>15}" for r in sweep]
    
    L += ["", f"== frammenti: chunk con meno di {a.min_chars} caratteri utili ({int(ignore.sum())} su {n}) =="]
    L += [f"{count:>4}  {name}" for name, count in frag_by_doc.most_common(15)]
    L += [f"   es. {ids[i]} | {texts[i][:60]!r}" for i in np.where(ignore)[0][:8]]
    
    L += ["", f"== gruppi di quasi-duplicati (soglia {a.threshold}) =="]
    for g in sorted(groups, key=len, reverse=True)[:25]:
        L.append(f"gruppo di {len(g)}:")
        L += [f"   {ids[i]} | {snip(i)}" for i in g]

    L += ["", "== coppie candidate alla discordanza (prime 30; prima quelle tra documenti diversi) =="]
    for p in cand[:30]:
        L.append(f"sim {p['sim']:.3f} cross_doc={p['cross_doc']} num_diff={p['numeric_diff']} {p.get('relation', '')}\n"
                 f"   A {p['a']} | {snip(p['i'])}\n   B {p['b']} | {snip(p['j'])}"
                 + (f"\n   -> {p['explanation']}" if p.get("explanation") else ""))
    
    (od / "embedding_report.txt").write_text("\n".join(L), encoding="utf-8")
    phase_started = mark_timing("generazione report", phase_started)
    
    # 8. Dimensionality reduction pca
    C, var = pca3(X)
    phase_started = mark_timing("PCA", phase_started)

    judged = [p for p in cand[:a.max_pairs]] if a.judge else cand[:60]
    plot_3d(C, var, docs, doc_names, groups, edges, judged, a.threshold, od / "latent_space_3d.png", a.show, frag_mask=ignore)
    phase_started = mark_timing("grafico 3D", phase_started)
    
    plot_hist(nn[nn > -1], a.threshold, a.cand_threshold, od / "similarity_distribution.png")
    phase_started = mark_timing("grafico distribuzione similarita", phase_started)


    print(f"chunk: {n} (analizzati {n_eff}, frammenti {int(ignore.sum())} | documenti: {len(doc_names)} | "
          f"varianza spiegata dalle prime 3 componenti: {var[:3].sum():.0%}\n"
          f"\nRidondanza (soglia {a.threshold}): {red['redundant_chunks']} chunk in {red['n_groups']} gruppi "
          f"(max {red['max_group_size']}), tasso {red['redundancy_rate']:.3f}, coppie tra documenti diversi: {red['pairs_cross_doc']}\n"
          f"Discordanza: {disc['candidate_pairs']} coppie candidate "
          f"({disc['candidate_numeric_diff']} con numeri diversi, {disc['candidates_cross_doc']} tra documenti)"
          + (f"; giudicate {disc['judged']}: contraddizioni {disc.get('judged_contradiction', 0)}" if a.judge else 
          "  [usa --judge per verificarle]") + "\n\n"
          f"scritti in {od}/: embedding_metrics.json, embedding_report.txt, {candidate_path.name}, latent_space_3d.png, similarity_distribution.png")
    mark_timing("tempo totale", run_started)


if __name__ == "__main__":
    main()