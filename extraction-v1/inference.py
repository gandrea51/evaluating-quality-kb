from collections import defaultdict, Counter
from pathlib import Path
import clingo
import json
import time
import re

HERE = Path(__file__).resolve().parent
DIR = HERE / "out"

CHUNKS = DIR / "chunks.jsonl"
FACTS = DIR / "facts.jsonl"
FACTS_NORM = DIR / "facts_norm.jsonl"
FACTS_LP = DIR / "facts.lp"

REDUNDANCY = Path("./rules/redundancy.lp")
DISCORDANCE = Path("./rules/discordance.lp")

FUNCTIONAL = DIR / "functional.lp"
METRICS = DIR / "metrics.json"
REPORT = DIR / "metrics_report.txt"

RULES = [REDUNDANCY, DISCORDANCE]

actions = []
totals = defaultdict(float)
counts = Counter()

def measure(phase, action, function):
    start = time.perf_counter()
    error = None

    try:
        return function()
    except Exception as e: 
        error = type(e).__name__; raise
    finally:
        seconds = time.perf_counter() - start 
        totals[phase] += seconds
        counts[phase] += 1
        actions.append({"phase": phase, "action": action, "seconds": seconds, "error": error})


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as fp:
        return [json.loads(ln) for ln in fp if ln.strip()]

def load_inputs():
    chunks = measure("input", "Lettura chunks.jsonl", lambda: read_jsonl(CHUNKS))
    records = measure( "input", "Lettura facts.jsonl", lambda: read_jsonl(FACTS))
    normalized = measure("input", "Lettura facts_norm.jsonl", lambda: read_jsonl(FACTS_NORM))
    return chunks, records, normalized

# Proprietà funzionali
def is_numeric(fact):
    value = fact.get("value")
    unit = fact.get("unit", "none")

    if isinstance(value, (int, float)) and not isinstance(value, bool): return True

    return re.fullmatch(r"[+-]?\d+(?:\.\d+)?", str(value).strip()) is not None and unit != "none"
    
def create_functional_file(facts):
    properties = sorted({fact["predicate"] for fact in facts if is_numeric(fact) and fact.get("predicate")})

    def write():
        with FUNCTIONAL.open("w", encoding="utf-8") as file:
            for prop in sorted(properties):
                file.write(f"functional({prop}).\n")

    measure("functional", "Scrittura functional.lp", write)
    return properties


def execute_clingo():
    def prepare():
        ctl = clingo.Control(["1"])

        for rule in RULES: ctl.load(str(rule))

        ctl.load(str(FACTS_LP))
        ctl.load(str(FUNCTIONAL))
        ctl.ground([("base", [])])
        return ctl

    ctl = measure("clingo", "Caricamento e grounding programmi ASP", prepare)
    atoms = []

    result = measure("clingo", "Risoluzione ASP", lambda: ctl.solve(on_model=lambda model: atoms.extend(
        model.symbols(shown=True)
    )))

    if not result.satisfiable:
        raise RuntimeError("Nessun modello trovato.")

    def convert(symbol):
        if symbol.type == clingo.SymbolType.Number: return symbol.number
        if symbol.type == clingo.SymbolType.String: return symbol.string
        return str(symbol)

    grouped = defaultdict(list)
    measure("clingo", "Conversione e raggruppamento atomi", lambda: [
        grouped[atom.name].append(tuple(convert(arg) for arg in atom.arguments))
        for atom in atoms
    ])
    return grouped

# Metriche
def get_metrics(grouped, chunks, records, normalized):
    def one(name):
        values = grouped.get(name, [])
        return values[0][0] if values else 0
    
    occurrences = one("n_occurrences")
    distinct = one("n_distinct")
    functional_group = one("n_functional_groups")
    conflicting_group = one("n_conflicting_groups")
    valid_records = sum(bool(record.get("parse_ok")) for record in records)
    failed_records = len(records) - valid_records

    return {
        "input": {
            "chunks": len(chunks),
            "facts_records": len(records),
            "valid_records": valid_records,
            "failed_records": failed_records,
            "normalized_facts": len(normalized)
        }, 
        "facts": {
            "occurrences": occurrences,
            "distinct_facts": distinct,
            "excess_occurrences": occurrences-distinct,
            "redundancy_rate": (occurrences-distinct)/occurrences if occurrences else 0.0,
            "redundant_facts": one("n_redundant_facts"),
            "cross_doc_redundant_facts": one("n_cross_doc_redundant"),
        },
        "discordance": {
            "functional_groups": functional_group,
            "conflicting_groups": conflicting_group,
            "discordance_rate": conflicting_group/functional_group if functional_group else 0.0,
            "conflicting_groups_cross_doc": one("n_cross_doc_conflicts")
        }        
    }

# Report
def reporting(grouped, metrics):
    lines = [
        "--- Metriche ---", json.dumps(metrics, indent=2, ensure_ascii=False),
        "", "--- Ridondanza ---"
    ]
    
    redundant = sorted(grouped.get("redundant_fact", []), key=lambda row: -row[4])
    if not redundant:
        lines.append("Nessun fatto ridondante.")
    for prop, entity, value, unit, count in redundant:
        lines.append(f"{count}x {prop}({entity}, {value!r}, {unit})")

    lines += ["", "--- Ridondanza tra documenti ---"]
    cross_redundant = sorted(grouped.get("cross_doc_redundant", []), key=lambda row: -row[4])
    if not cross_redundant:
        lines.append("Nessuna ridondanza tra documenti.")
    for prop, entity, value, unit, count in cross_redundant:
        lines.append(f"{count} documenti {prop}({entity}, {value!r}, {unit})")

    lines += ["", "--- Discordanze ---"]
    conflicts = defaultdict(list)
    for prop, entity, value, unit, source in grouped.get("conflict_value", []):
        conflicts[(prop, entity)].append((value, unit, source))
    if not conflicts:
        lines.append("Nessuna discordanza.")
    for (prop, entity), values in sorted(conflicts.items()):
        lines.append(f"{prop}({entity}):")
        for value, unit, source in sorted(values):
            lines.append(f"    {value!r} {unit} <- {source}")
    
    lines += ["", "--- Gruppi in conflitto ---"]
    for prop, entity in sorted(grouped.get("conflicting_group", [])):
        lines.append(f"{prop}({entity})")

    lines += ["", "--- Conflitti tra documenti ---"]
    cross_conflicts = sorted(set(grouped.get("cross_doc_conflict", [])))
    if not cross_conflicts:
        lines.append("Nessun conflitto tra documenti.")

    for prop, entity in cross_conflicts:
        lines.append(f"{prop}({entity})")

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return lines

def timing(start):
    elapsed = time.perf_counter() - start

    lines = ["--- Tempi per fase ---", f"{'Fase':<20} {'Chiamate':>10} {'Secondi':>12}"]

    for phase, seconds in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"{phase:<20} {counts[phase]:>10} {seconds:>12.4f}")

    lines += ["", "--- Dettaglio azioni ---"]
    for i, action in enumerate(actions, start=1):
        error = (f" | Errore: {action['error']}" if action["error"] else "")

        lines.append(f"{i:03}. {action['seconds']:9.4f} s | {action['phase']:<12} | {action['action']}{error}")
    lines.append(f"\nTempo totale programma: {elapsed:.4f} s")
    return lines

def main():
    start = time.perf_counter()
    DIR.mkdir(parents=True, exist_ok=True)

    required = [CHUNKS, FACTS, FACTS_NORM, FACTS_LP, *RULES]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")

    chunks, records, normalized = load_inputs()
    functional = create_functional_file(normalized)

    print(f"Chunk: {len(chunks)} | Record facts: {len(records)} | Fatti normalizzati: {len(normalized)}\n"
          f"Proprietà funzionali: {functional}")

    grouped = execute_clingo()
    metrics = measure("metrics", "Calcolo delle metriche", lambda: get_metrics(grouped, chunks, records, normalized))

    measure("output", "Scrittura metrics.json", lambda: METRICS.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"))
    measure("output", "Generazione metrics_report.txt", lambda: reporting(grouped, metrics))

    # Registra l'azione di riepilogo prima di stampare i tempi.
    measure("report", "Preparazione riepilogo finale", lambda: None)

    time_lines = timing(start)

    # Aggiunge le misurazioni al report testuale.
    with REPORT.open("a", encoding="utf-8") as file:
        file.write("\n\n" + "\n".join(time_lines) + "\n")

    print(f"\n--- Risultati ---\n"
          f"{json.dumps(metrics, indent=2, ensure_ascii=False)}\n"
          f"Scritti: {METRICS}, {REPORT}, {FUNCTIONAL}")
    print("\n" + "\n".join(time_lines))


if __name__ == "__main__":
    main()

