from collections import defaultdict
from pathlib import Path
import urllib.request
import statistics
import hashlib
import json
import time
import re

CHUNKS = Path("out/chunks.jsonl")
FACTS = Path("out/facts.jsonl")

MODEL = "llama3.1:8b"
URL = "http://localhost:11434/api/chat"

LIMIT = 10
CTX = 4096
PREDICT = 512
TEMPERATURE = 0
ALIVE = "30m"
TIMEOUT = 1800
VERSION = "v1"

PROMPT = """
    Estrai fatti atomici dal testo per verificare la qualità della base di conoscenza di una azienda.
    
    Rispondi SOLAMENTE con JSON valido:
        {"facts": [{"entity": "...", "property": "...", "value": "...", "evidence": "..."}]}

    Regole:
        1. Un fatto contiene una entity, una property e un value.
        2. Entity deve identificare una cosa specifica nominata nel testo.
        3. Property deve essere breve, minuscola e in snake_case.
        4. Una sola informazione esplicita, senza deduzioni.
        5. Mantieni i numeri e le unità come scritti nel testo.
        6. Evidence deve essere una citazione estratta dal testo, di circa 25 parole massime.
        7. Estrai anche informazioni da elenchi, tabelle e cataloghi.
        8. Nei menù e cataloghi, usa come entity il nome della voce specifica e come property le sue caratteristiche.
        9. Restituisci {"facts": []} solo se non esistono fatti esplicitamente supportati dal testo.

    """

actions = []
totals = defaultdict(float)
counts = defaultdict(int)

# Esecuzione di una azione e registrazione della durata
def measure(phrase, action, function):
    start = time.perf_counter()
    try:
        return function()
    finally:
        seconds = time.perf_counter() - start
        totals[phrase] += seconds
        counts[phrase] += 1
        actions.append({"phrase": phrase, "action": action, "seconds": seconds})

def read_jsonl(path):
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(ln) for ln in fp if ln.strip()]

# Normalizzazione del testo
def normalize(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()

# Cache key
def cache_key(chunk):
    content = f"{chunk['text']}|{MODEL}|{VERSION}"
    return hashlib.sha1(content.encode("utf-8")).hexdigest()

# Parsing output dell'LLM
def parse_facts(raw):
    data = json.loads(raw)
    
    if not isinstance(data, dict): raise ValueError("Risposta json non è un oggetto.")

    if not isinstance(data.get("facts"), list): raise ValueError("facts assente o non valido.")

    facts = []
    for item in data["facts"]:
        if not isinstance(item, dict): continue

        fact = {
            field: str(item.get(field, "")).strip()
            for field in ("entity", "property", "value", "evidence")    
        }
        if all(fact[field] for field in ("entity", "property", "value")): facts.append(fact)
    return facts

def validate_evidence(facts, text):
    source = normalize(text)

    for fact in facts:
        evidence = fact["evidence"]
        fact["evidence_ok"] = (bool(evidence) and normalize(evidence) in source)
    return facts

def call_ollama(chunk):
    payload = measure("payload", f"Costruzione richiesta {chunk['chunk_id']}", lambda: {
        "model": MODEL, "stream": False, "format": "json", "keep_alive": ALIVE,
        "options": {"temperature": TEMPERATURE, "num_ctx": CTX, "num_predict": PREDICT},
        "messages": [
            {"role": "system", "content": PROMPT}, {"role": "user", "content": "Testo:\n"+chunk["text"]}
        ]
    })
    request = measure("request", f"Serializzazione http {chunk['chunk_id']}", lambda:
                      urllib.request.Request(URL, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), 
                                             headers={"Content-Type": "application/json"}))

    def send():
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.load(response)

    result = measure("ollama_http", f"Chiamata Ollama e lettura risposta {chunk['chunk_id']}", send)
    return result["message"]["content"], result.get("done_reason")

# Elaborazione di un chunk
def process_chunk(chunk):
    chunk_id = chunk["chunk_id"]
    raw, reason = call_ollama(chunk)

    facts = measure("parsing", f"Parsing json {chunk_id}", lambda: parse_facts(raw))
    facts = measure("evidence", f"Validazione citazioni {chunk_id}", lambda: validate_evidence(facts, chunk["text"]))

    return {
        "key": cache_key(chunk),
        "chunk_id": chunk_id,
        "doc_id": chunk["doc_id"],
        "model": MODEL,
        "prompt_version": VERSION,
        "chars": len(chunk["text"]),
        "parse_ok": True,
        "truncated": reason == "length",
        "facts": facts,
        "raw": None
    }


def main():
    run_start = time.perf_counter()
    chunks = measure("input", "Lettura chunks.jsonl", lambda: read_jsonl(CHUNKS))

    if LIMIT > 0: chunks = chunks[:LIMIT]
    
    output_path = measure("filesystem", "Creazione cartella e preparazione output", lambda: prepare_output())
    existing = measure("cache", "Lettura facts.jsonl esistente", lambda: read_existing(output_path))

    # Ultimo record per ogni keys
    latest = {r["key"]: r for r in existing if r.get("key")}
    cache = {key: record for key, record in latest.items() if record.get("parse_ok")}
    print(f"Chunk selezionati: {len(chunks)}\n"
          f"Modello: {MODEL}\n"
          f"Record validi in cache: {len(cache)}")

    for index, chunk in enumerate(chunks, start=1):
        key = measure("cache", f"Calcolo chiave {chunk['chunk_id']}", lambda c=chunk: cache_key(c))
        if key in cache:
            print(f"[{index}/{len(chunks)}] {chunk['chunk_id']}: cache hit"); continue

        print(f"[{index}/{len(chunks)}] {chunk['chunk_id']}: elaborazione")

        try:
            record = measure("chunk_total", f"Elaborazione completa {chunk['chunk_id']}", lambda c=chunk: process_chunk(c))
        except Exception as e:
            record = {
                "key": key, "chunk_id": chunk["chunk_id"], "doc_id": chunk.get("doc_id", ""),
                "model": MODEL, "prompt_version": VERSION, "chars": len(chunk["text"]), "parse_ok": False,
                "truncated": False, "facts": [], "raw": str(e)
            }
            print(f"  Errore: {e}")
        
        def save():
            with output_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                file.flush()

        measure("write", f"Scrittura facts.jsonl {chunk['chunk_id']}", save)
        latest[key] = record
        if record["parse_ok"]: cache[key] = record

        print(f"  Fatti: {len(record['facts'])}\n"
              f"  Parsing OK: {record['parse_ok']}")

    measure("report", "Stampa del riepilogo", lambda: print_summary(chunks, latest, run_start))


def prepare_output():
    FACTS.parent.mkdir(parents=True, exist_ok=True)
    return FACTS

def read_existing(path):
    return read_jsonl(path) if path.exists() else []

def print_summary(chunks, latest, run_start):
    records = []
    for chunk in chunks:
        key = cache_key(chunk)
        if key in latest: records.append(latest[key])

    facts = [fact for record in records for fact in record.get("facts", [])]
    elapsed = time.perf_counter() - run_start

    print(f"--- Riepilogo ---\n"
          f"Chunk selezionati: {len(chunks)}\n"
          f"Chunk con record: {len(records)}\n"
          f"Fatti estratti: {len(facts)}\n"
          f"Parsing falliti: {sum(not r.get('parse_ok', False) for r in records)}\n"
          f"Output troncati: {sum(r.get('truncated', False) for r in records)}")

    if facts:
        valid = sum(f.get("evidence_ok", False) for f in facts)
        print(f"Evidenze trovate nel testo: {valid}/{len(facts)} ({valid / len(facts):.1%})\n")

    print(f"Tempo totale programma: {elapsed:.3f} s\n\n"
          f"--- Tempi aggregati per fase ---\n"
          f"{'Fase':<18} {'Chiamate':>10} {'Secondi':>12}")
    for phase in sorted(totals, key=lambda p: totals[p], reverse=True):
        print(f"{phase:<18} {counts[phase]:>10} {totals[phase]:>12.3f}")

    print("\n--- Elenco cronologico delle azioni ---")

    for i, item in enumerate(actions, start=1):
        print(f"{i:03}. [{item['phrase']:<12}] {item['seconds']:>9.3f} s | {item['action']}\n\n")

    print(f"Output salvato in: {FACTS}")


if __name__ == "__main__":
    main()

