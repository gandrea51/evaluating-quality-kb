import csv
import json
import time
import random
from pathlib import Path
import numpy as np

from kb_analysis.core import prepare_chunks, build_valid_pairs, cosine_matrix, judge_pairs, judged_summary, compatible_pair, DEFAULT_ENTITIES
from kb_analysis.cleaning_area import run_contents, _embed_cache, _embedding_cache_key
from kb_analysis.utils import read_jsonl, MODEL, numbers

CHUNKS = "out/chunks_semantic.jsonl"
OUT = Path("out")
JUDGE_MODEL = "qwen2.5:14b-instruct"
CAND_THR = 0.85
MIN_CHARS = 40
BANDS = [(0.80, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.01)]
PER_BAND = 10
SEED = 42
MIN_PRECISION = 0.80      # precisione minima richiesta a una fascia per accettarla


# STEP 1: campione di coppie per fascia di similarità da etichettare a mano
def export_redundancy_sample():
    chunks = read_jsonl(CHUNKS)
    data = prepare_chunks(chunks, MIN_CHARS)
    X, _, _ = _embed_cache(data["texts"], _embedding_cache_key(CHUNKS), MODEL)
    I, J = build_valid_pairs(data)
    S = cosine_matrix(X)[I, J]

    rng = random.Random(SEED)
    rows = []
    for lo, hi in BANDS:
        idx = np.flatnonzero((S >= lo) & (S < hi))
        pick = rng.sample(list(idx), min(PER_BAND, len(idx)))
        for k in pick:
            i, j = int(I[k]), int(J[k])
            rows.append({"band": f"{lo}-{hi}", "sim": round(float(S[k]), 4),
                         "a": data["ids"][i], "b": data["ids"][j],
                         "text_a": data["texts"][i][:600], "text_b": data["texts"][j][:600],
                         "label": ""})   # 1 = stesso contenuto (duplicato/quasi), 0 = no
    with open(OUT / "annot_redundancy.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} coppie in out/annot_redundancy.csv: compila la colonna label (1/0)")


# STEP 2: precisione per fascia -> soglia di default
TAU_GRID = (0.80, 0.85, 0.90, 0.92, 0.95, 0.98)
MIN_N = 3   # numero minimo di coppie annotate per fidarsi di una soglia

def choose_tau():
    text = {c["chunk_id"]: c["text"] for c in read_jsonl(CHUNKS)}
    allrows = [r for r in csv.DictReader(open(OUT / "annot_redundancy.csv", encoding="utf-8-sig")) if r["label"] in ("0", "1")]
    ok = [r for r in allrows if compatible_pair(text[r["a"]], text[r["b"]], DEFAULT_ENTITIES)]
    dropped = [r for r in allrows if r not in ok]

    lost_pos = sum(1 for r in dropped if r["label"] == "1")
    removed_neg = sum(1 for r in dropped if r["label"] == "0")
    print(f"filtro: positivi persi = {lost_pos}, falsi positivi eliminati = {removed_neg} su {len(allrows)} coppie")

    table = {}
    for t in TAU_GRID:
        sub = [int(r["label"]) for r in ok if float(r["sim"]) >= t]
        table[t] = {"n": len(sub), "precision": (sum(sub) / len(sub)) if sub else None}
        p = table[t]["precision"]
        print(f"sim >= {t:.2f}: {len(sub)} coppie, precisione {p if p is None else round(p, 2)}")

    valid = [t for t in TAU_GRID if table[t]["n"] >= MIN_N and table[t]["precision"] >= MIN_PRECISION]
    if not valid:
        raise SystemExit("nessuna soglia accettabile: guarda le coppie compatibili con label 0")
    tau = min(valid)
    print("tau scelto:", tau)

    json.dump({"tau": tau, "table": {str(k): v for k, v in table.items()},
               "lost_pos": lost_pos, "removed_neg": removed_neg, "n_labeled": len(allrows), "n_compatible": len(ok)},
              open(OUT / "tau.json", "w"), indent=2)
    return tau

# STEP 3: run completo -> run.json
def close_run():
    tau = json.load(open(OUT / "tau.json"))["tau"]
    started = time.perf_counter()
    result, todo = run_contents(CHUNKS, str(OUT), threshold=tau, cand_threshold=CAND_THR, min_chars=MIN_CHARS,
                                judge=JUDGE_MODEL, max_pairs=10**9)
    s = result["summary"]
    n_eff = s["n_analyzed"]

    pairs, judged = [], {}
    if todo:
        pairs = judge_pairs(todo["model"], todo["pairs"], todo["texts"], todo["ids"], todo["out_dir"])
        judged = judged_summary(pairs)

    contr = [p for p in pairs if p.get("relation") == "contraddizione"]
    involved = {p["i"] for p in contr} | {p["j"] for p in contr}

    invalid = sum(1 for p in pairs if p.get("relation") == "invalid")

    n_ver = judged.get("judged", 0) - invalid
    red = result["redundancy"]
    report = {
        "run": {"dataset": "ZX Bank", "chunks_file": CHUNKS, "embedder": MODEL, "tau": tau,
                "cand_threshold": CAND_THR, "llm": JUDGE_MODEL, "min_chars": MIN_CHARS,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.perf_counter() - started, 1)},
        "coverage": {"analyzed": n_eff, "total": s["n_chunks"], "fragments": s["n_fragments"],
                     "coverage": n_eff / s["n_chunks"]},
        "redundancy": {"rate": red["redundancy_rate"], "excess_units": red["excess_units"], "groups": red["n_groups"],
                       "largest_group": red["max_group_size"], "cross_doc_pairs": red["pairs_cross_doc"]},
        "coherence": {"candidates": len(pairs), "verified": n_ver, "invalid": invalid,
                      "contradictions": len(contr),
                      "contradiction_rate": len(contr) / n_ver if n_ver else 0.0,
                      "chunks_involved": len(involved),
                      "incoherence_kb": len(involved) / n_eff if n_eff else 0.0},
        "sensitivity": [{
            "tau": r["threshold"], 
            "rate_filtered": r["redundancy_rate"], 
            "rate_raw": raw["redundancy_rate"],
            "largest_group": r["max_group_size"]
        } for r, raw in zip(result["sweep"], result["sweep_raw"])]
    }

    json.dump(report, open(OUT / "run.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # campione per validare l'LLM: 15 "contraddizione" + 15 altre
    rng = random.Random(SEED)
    pos = [p for p in pairs if p.get("relation") == "contraddizione"]
    neg = [p for p in pairs if p.get("relation") not in ("contraddizione", "invalid")]
    rows = []
    for tag, grp in (("pred_contr", pos), ("pred_other", neg)):
        for p in rng.sample(grp, min(15, len(grp))):
            rows.append({"group": tag, "a": p["a"], "b": p["b"], "llm": p["relation"],
                         "explanation": p.get("explanation", ""),
                         "text_a": todo["texts"][p["i"]][:600], "text_b": todo["texts"][p["j"]][:600], "label": ""})
    with open(OUT / "annot_coherence.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))


# STEP 4: precisione dell'LLM sul campione (label: 1 = contraddizione reale, 0 = no)
def llm_precision():
    rows = [r for r in csv.DictReader(open(OUT / "annot_coherence.csv", encoding="utf-8-sig")) if r["label"] in ("0", "1")]
    pc = [int(r["label"]) for r in rows if r["group"] == "pred_contr"]
    po = [int(r["label"]) for r in rows if r["group"] == "pred_other"]
    res = {"precision_contradiction": sum(pc) / len(pc) if pc else None, "n_pred_contr": len(pc),
           "missed_contradiction_rate": sum(po) / len(po) if po else None, "n_pred_other": len(po)}
    print(res)
    report = json.load(open(OUT / "run.json"))
    report["validation"] = {"llm": res}
    json.dump(report, open(OUT / "run.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

from kb_analysis.utils import components
from kb_analysis.core import compatible_mask

def diagnose():
    tau = json.load(open(OUT / "tau.json"))["tau"]
    data = prepare_chunks(read_jsonl(CHUNKS), MIN_CHARS)
    X, _, _ = _embed_cache(data["texts"], _embedding_cache_key(CHUNKS), MODEL)
    I, J = build_valid_pairs(data)
    S = cosine_matrix(X)[I, J]
    ok = compatible_mask(data["texts"], I, J, S, tau, DEFAULT_ENTITIES)

    for name, mask in (("grezzo", S >= tau), ("filtrato", (S >= tau) & ok)):
        edges = list(zip(I[mask].tolist(), J[mask].tolist()))
        groups = components(data["n"], edges)
        excess = sum(len(g) - 1 for g in groups)
        print(f"{name}: coppie={int(mask.sum())} gruppi={len(groups)} R={excess / data['n_effective']:.3f} "
              f"max={max(map(len, groups), default=0)}")
        if name == "filtrato" and groups:
            big = max(groups, key=len)
            for k in big[:5]:
                print("  ", data["ids"][k], "|", data["texts"][k][:100].replace("\n", " "))

    t = data["texts"]
    print("testi esattamente identici:", len(t) - len(set(t)), "su", len(t))
    dropped = (S >= tau) & ~ok
    for k in np.flatnonzero(dropped):
        i, j = int(I[k]), int(J[k])
        print("SCARTATA", round(float(S[k]), 3), data["ids"][i], "|", data["ids"][j],
              "| numeri diversi:", sorted(numbers(t[i]) ^ numbers(t[j]))[:6])

STEP = 3   # 1 = campione ridondanza | 2 = scegli tau | 3 = run completo | 4 = validazione LLM

if __name__ == "__main__":
    {1: export_redundancy_sample, 2: choose_tau, 3: close_run, 4: llm_precision, 5: diagnose}[STEP]()