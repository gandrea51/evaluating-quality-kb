from collections import Counter, defaultdict
from pathlib import Path
import unicodedata
import difflib
import json
import time
import re

DIR = Path("out")
CHUNKS = DIR/"chunks.jsonl"
FACTS = DIR/"facts.jsonl"
ALIASES = DIR/"aliases.json"

STYLE = "generic"       # Oppure "predicate"
MODEL = None
PROMPT_VERSION = None
KEEP = None

INT_MAX = 2_147_483_647
RESERVED = {'not'}
CURRENCIES = {"$": "usd", "€": "eur", "£": "gbp"}
NUMBER = re.compile(
    r"^(?P<cur>[$€£])?\s*"
    r"(?P<num>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"\s*(?P<unit>.*)$"
)

actions = []
totals = defaultdict(float)
counts = Counter()

def measure(phase, action, function):
    start = time.perf_counter()
    error = None

    try:
        return function()
    except Exception as exc:
        error = type(exc).__name__; raise
    finally:
        seconds = time.perf_counter() - start
        totals[phase] += seconds
        counts[phase] += 1

        actions.append({"phase": phase, "action": action, "seconds": seconds, "error": error})

def read_jsonl(path):
    with path.open(encoding="utf-8") as fp:
        return [json.loads(ln) for ln in fp if ln.strip()]

# Normalizzazione identificatori ASP
def ident(text, prefix = "x"):
    text = unicodedata.normalize("NFKD", str(text))
    text = text.encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")

    if not text: return ""
    if not text[0].isalpha(): text = f"{prefix}_{text}"
    if text in RESERVED: text += "_"
    return text

def cleaning(entity):
    return re.sub(r"^(the|a|an)\s+", "", str(entity).strip(), flags=re.IGNORECASE)

# Norm. value
def parsing(value):
    value = str(value).strip().strip("\"'“”‘’")
    match = NUMBER.match(value)
    if match:
        number = match["num"].replace(",", "")
        unit_text = match["unit"].strip()
        unit = None

        if match["cur"] and not unit_text: 
            unit = CURRENCIES[match["cur"]]
        elif not match["cur"] and unit_text in ("", "."): unit = "none"
        elif not match["cur"] and unit_text == "%": unit = "percent"
        elif not re.search(r"\d", unit_text): 
            norm = ident(unit_text, "u")
            if norm and len(norm.split("_")) <= 3: unit = norm

        if unit is not None:
            if (re.fullmatch(r"[+-]?\d+", number) and abs(int(number)) <= INT_MAX):
                return int(number), unit
            return number, unit

    return " ".join(value.lower().split()).strip(" .;:,"), "none"

# Conversione in termina ASP
def term(value):
    if isinstance(value, int): return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'

# Validazione
def words(text):
    return re.findall(r"\w+", str(text).lower())

def evidence_ok(evidence, text):
    if not evidence: return False
    return (" ".join(words(evidence))) in (" ".join(words(text)))

def value_ok(value, evidence):
    value_words = set(words(value))
    return bool(value_words) and value_words <= (set(words(evidence)))


def load_aliases():
    aliases = {"entities": {}, "properties": {}}
    if not ALIASES.exists(): return aliases

    with ALIASES.open(encoding="utf-8") as fp: raw = json.load(fp)

    for category, prefix in ("entities", "e"), ("properties", "p"):
        aliases[category] = {ident(source, prefix): ident(destination, prefix)
                             for source, destination in raw.get(category, {}).items()}
    return aliases

def apply_alias(name, table):
    visited = set()
    while name in table and name not in visited:
        visited.add(name); name = table[name]
    return name

# Filtraggio record llm
def filter_records(records):
    combinations = Counter((r.get("model"), r.get("prompt_version")) for r in records)
    if MODEL is not None:
        records = [r for r in records if r.get("model") == MODEL]
    if PROMPT_VERSION is not None:
        records = [r for r in records if r.get("prompt_version") == PROMPT_VERSION]
    return records, combinations

# Ultimo record valido per chunk
def get_last_valid_records(records):
    last = {}

    for record in records:
        if record.get("parse_ok") and record.get("chunk_id"):
            last[record["chunk_id"]] = record
    return last

# Normalizzazione
def normalize_facts(chunks, records, aliases):
    text_of = {c["chunk_id"]: c["text"] for c in chunks}
    source_of = {c["chunk_id"]: f"c{i:04d}" for i, c in enumerate(chunks, start=1)}

    dropped = Counter()
    output = []
    seen = set()

    for chunk_id, record in records.items():
        if chunk_id not in text_of:
            dropped["chunk non trovato"] += len(record["facts"]); continue

        chunk_text = text_of[chunk_id]
        source = source_of[chunk_id]

        for fact in record.get("facts", []):
            evidence = fact.get("evidence", "")
            value_original = fact.get("value", "")

            valid_evidence = evidence_ok(evidence, chunk_text)
            valid_value = value_ok(value_original, evidence)

            # Scarto fatti non verificati
            if not KEEP:
                if not valid_evidence:
                    dropped["citazione non presente nel chunk"] += 1; continue
                if not valid_value:
                    dropped["valore non presente nella citazione"] += 1; continue

            # Normalizzazione property
            prop = ident(fact.get("property", ""), "p")
            prop = apply_alias(prop, aliases["properties"])

            # Normalizzazione entity
            entity = ident(fact.get("entity", ""), "e")
            entity = apply_alias(entity, aliases["entities"])

            if not prop or not entity:
                dropped["entià' o proprietà vuota"] += 1; continue

            # Normalizzazione value
            value, unit = parsing(value_original)
            if value == "":
                dropped["valore vuoto"] += 1; continue

            # Eliminazione duplicati
            key = (prop, entity, value, unit, source)
            if key in seen:
                dropped["duplicato nella stessa sorgente"] += 1; continue
            seen.add(key)

            output.append({
                "src": source_of[chunk_id], "chunk_id": chunk_id,
                "doc_id": record.get("doc_id", ""), "predicate": prop,
                "entity": entity, "value": value, "unit": unit,
                "original": {
                    "entity": fact.get("entity", ""), "property": fact.get("property", ""), "value": value_original
                },
                "evidence": evidence, "evidence_ok": valid_evidence, "value_ok": valid_value
            })
    return output, dropped, source_of

# Creazione atomo ASP
def create_atom(fact):
    arguments = [fact["entity"], term(fact["value"]), fact["unit"], fact["src"]]
    if STYLE == "generic":
        return f"fact({fact['predicate']}, {', '.join(arguments)})."

    return f"{fact['predicate']}({', '.join(arguments)})."

# Suggerimenti alias
def suggest(counter, limit=40):
    names = sorted(counter)
    tokens = {name: set(name.split("_")) for name in names}
    output = []
    
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            subset = (tokens[first] < tokens[second] or tokens[second] < tokens[first])
            similar = (len(names) <= 800 and difflib.SequenceMatcher(None, first, second).ratio() >= 0.85)
            if subset or similar:
                output.append(f"{first} ({counter[first]}) ~ {second} ({counter[second]})")
                if len(output) >= limit: return output

    return output

def shared_token_groups(counter, limit=40):
    by_token = defaultdict(list)

    for name in counter:
        for token in set(name.split("_")):
            if (len(token) >= 4 and not token.isdigit()):
                by_token[token].append(name)

    return [f"parola '{token}': " + ", ".join(f"{name} ({counter[name]})" for name in sorted(names))
            for token, names in sorted(by_token.items()) if 2 <= len(names) <= 8][:limit]

def write_outputs(output, dropped, source_of):
    sources = sorted({f["src"] for f in output})
    chunk_of = {source: chunk for chunk, source in source_of.items()}
    doc_of = {f["src"]: f["doc_id"] for f in output}

    facts_path = DIR/"facts.lp"
    with facts_path.open("w", encoding="utf-8") as fp:
        fp.write(f"% generato dal normalizzatore; {len(output)} fatti, {len(sources)} sorgenti\n")

        for source in sources:
            fp.write(f"% {source} = {chunk_of[source]}\n")

        for source in sources:
            document = re.sub(r"\.[A-Za-z0-9]+$", "", doc_of[source])
            document = ("d_" + ident(document, "d"))
            fp.write(f"chunk_doc({source}, {document}).\n")

        fp.write("\n")
        for fact in sorted(output, key=lambda f: (f["predicate"], f["entity"], f["src"])):
            fp.write(create_atom(fact) + "\n")
    
    jsonl_path = DIR/"facts_norm.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for fact in output:
            fact["atom"] = create_atom(fact)
            file.write(json.dumps(fact, ensure_ascii=False) + "\n")

    entities = Counter(f["entity"] for f in output)
    properties = Counter(f["predicate"] for f in output)
    
    lines = [
        (f"Fatti validi: {len(output)} |"
         f"Chunk con fatti: {len(sources)} |"
         f"Predicati distinti: {len(properties)} |"
         f"Entità distinte: {len(entities)}"),
        "Scartati: " + (", ".join(f"{k}: {v}" for k, v in dropped.items()) or "nessuno"),
        "", "-- Possibili alias di entità --", *suggest(entities),
        "", "-- Entità che condividono una parola --", *shared_token_groups(entities),
        "", "-- Entità (occorrenze) --", *[f"{n}: {c}" for n, c in entities.most_common()],
        "", "-- Proprietà (occorrenze) --", *[f"{n}: {c}" for n, c in properties.most_common()]
    ]
    report_path = DIR/"norm_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return facts_path, jsonl_path, report_path, lines

def timings(start):
    print(f"--- Tempi aggregati per fase ---\n"
          f"{'Fase':<24}{'Chiamate':>12}{'Secondi':>14}")

    for phase, seconds in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        print(f"{phase:<24}{counts[phase]:>12}{seconds:>14.4f}")

    print(f"--- Dettaglio cronologico delle azioni ---")
    for i, action in enumerate(actions, start=1):
        status = (f" Errore={action['error']}" if action["error"] else "")
        print(f"{i:03}. {action['seconds']:9.4f} s | {action['phase']:<16} | {action['action']}{status}")

    print(f"Tempo totale programma: {time.perf_counter() - start:.4f} s")


def main():
    start = time.perf_counter()
    DIR.mkdir(parents=True, exist_ok=True)

    chunks = measure("input", "Lettura chunks.jsonl", lambda: read_jsonl(CHUNKS))
    records = measure("input", "Lettura facts.jsonl", lambda: read_jsonl(FACTS))

    # 2. Filtri modello / prompt
    records, combinations = measure("filter", "Filtro modello e prompt", lambda: filter_records(records))
    
    print(f"Modelli/versioni presenti:")
    for combination, count in combinations.items():
        print(f"  {combination}: {count}")

    # 3. Ultimo risultato valido per chunk
    last = measure("cache", "Selezione ultimo record valido per chunk", lambda: get_last_valid_records(records))
    
    # 4. Alias
    aliases = measure("aliases", "Caricamento alias", load_aliases)

    output, dropped, sorce_of = measure("normalize", "Validazione, normalizzazione e duplicazione", lambda: normalize_facts(chunks, last, aliases))
    paths = measure("output", "Scrittura dei tre file finali", lambda: write_outputs(output, dropped, sorce_of))
    facts_path, jsonl_path, report_path, lines = paths

    print(f"\n--- Riepilogo ---\n"
          f"{lines[0]}\n{lines[1]}\n"
          f"Prime righe di facts.lp:\n")
    for fact in output[:6]:
        print("  " + create_atom(fact))

    print(f"\nScritti:\n {facts_path}\n {jsonl_path}\n {report_path}")
    timings(start)
    

if __name__ == "__main__":
    main()

