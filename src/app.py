from flask import Flask, render_template, request, jsonify
from chunker import run_chunking, config4form, DIRECTORY, OUTPUT, PREVIEW
from analysis.core import judge_pairs
from analysis.contenuto import run_contents, finalize_coherence, get_report
#from analysis.mapping_area import run_structure
from analysis.utils import get_int, get_float
from analysis.utils import LLM_JUDGE
import urllib.request
import threading
import urllib.error
import json
import time
import copy
import uuid
import os

os.environ.setdefault("MPLBACKEND", "Agg")

"""
    Flask application for interactive analysis of KB chunks. Features: 
        - Interface: Home page and results template.
        - Routing: /analyze with copertura-ridondanza-coerenza
        - LLM Evaluation: Performs judgment of candidate pairs in a separate thread.
        - Monitoring: Exposes the status and partial results of jobs via an endpoint.

    Domain analyses and logic are delegated to the kb_analysis modules
"""

app = Flask(__name__)

JOBS = {}
JOBS_LOCK = threading.Lock()
OLLAMA_LOCK = threading.Lock()
MAX_JOBS_HISTORY = 100

GROUPS_AVAILABLE = {"Contenuto"}

def _cleanup_old_jobs():
    if len(JOBS) > MAX_JOBS_HISTORY:
        for key, job in JOBS.items():
            if job["status"] != "running":
                JOBS.pop(key)
                break

def parse_topics(raw):
    # Textarea 'expected_topics': una riga per area, formato  'Area: arg1 | arg2'
    topics = {}
    for line in (raw or "").splitlines():
        if ":" not in line:
            continue

        area, items = line.split(":", 1)
        items = [t.strip() for t in items.split("|") if t.strip()]
        if area.strip() and items:
            topics[area.strip()] = items
    return topics or None

# (job llm)
def judge_worker(job_id:str, todo:dict, partial:dict):
    started = time.perf_counter()

    def on_result(index, pair):
        pair_key = f'{pair.get("a")}|{pair.get("b")}'

        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if not job:
                return
            job["items"][pair_key] = {
                "relation": pair.get("relation"), 
                "explanation": pair.get("explanation")
            }
            job["done"] = index
            job["elapsed"] = round(time.perf_counter() - started, 1)

    try:
        # Serializzazione
        with OLLAMA_LOCK:
            pairs = judge_pairs(todo["model"], todo["pairs"], todo["texts"], todo["ids"], todo["out_dir"], on_result=on_result)
        
        # C e Coerenza = 1 - C sulla copia del risultato; poi report completo in console
        finalize_coherence(partial, pairs, todo["texts"])
        get_report(partial)
 
        coerenza = partial["coerenza"]
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["coerenza"] = {
                    key: coerenza.get(key) for key in ("n_judged", "n_invalid", "n_contradictions", "n_chunks_in_contradiction", 
                                                       "C_pairs", "C_chunks", "coerenza", "coerenza_chunks", "conflicts")
                }
                JOBS[job_id]["status"] = "done"
  
    except Exception as e:
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["error"] = f"{type(e).__name__}: {str(e)}"

    finally:
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["elapsed"] = round(time.perf_counter() - started, 1)

# (create job)
def start_job(todo:dict, results:dict) -> str:
    job_id = uuid.uuid4().hex

    partial = dict(results)
    partial["coerenza"] = copy.deepcopy(results["coerenza"])

    with JOBS_LOCK:
        _cleanup_old_jobs()
        JOBS[job_id] = {
            "status": "running", "done": 0, "total": len(todo.get("pairs", [])), 
            "elapsed": 0.0, "items": {}, "error": None, "coerenza": None
        }
    thread = threading.Thread(target=judge_worker, args=(job_id, todo, partial), daemon=True, name=f"Worker-{job_id[:8]}")
    thread.start()
    return job_id

CHAT_MODEL = os.environ.get("CHAT_MODEL", LLM_JUDGE)
CHAT_SYSTEM = (
    "Sei un assistente tecnico virtuale. Rispondi alla domanda in italiano, in modo chiaro e conciso "
    "(al massimo 5-6 righe). Se non conosci la risposta, dillo."
)

def ask_llm(quertion, timeout=120):
    body = {
        "model": CHAT_MODEL,
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0.3, "num_ctx": 4096},
        "messages": [{"role": "system", "content": CHAT_SYSTEM},
                     {"role": "user", "content": quertion}]
    }

    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)["message"]["content"].strip()



# Routes

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", results=None, error=None, chunking_modes=["fixed", "recursive", "semantic"])

@app.route("/indicators", methods=["GET"])
def indicators():
    return render_template("indicators.html", results=None, error=None)

@app.route("/ask", methods=["POST"])
def ask():
    question = ((request.get_json(silent=True) or {}).get("question") or "").strip()
    if not question:
        return jsonify(error="Scrivi una domanda."), 400
    if len(question) > 2000:
        return jsonify(error="Domanda troppo lunga (max 2000 caratteri)."), 400

    try:
        return jsonify(answer=ask_llm(question))
    except urllib.error.URLError:
        return jsonify(error="Ollama non raggiungibile: è avviato su localhost:11434?"), 503
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {e}"), 500


@app.route("/chunking", methods=["POST"])
def chunking():
    try:
        config = config4form(request.form)
        directory = request.form.get("directory", DIRECTORY)

        save = request.form.get("save") == "on"
        output = OUTPUT if save else None

        result = run_chunking(directory, config, output, save)
        preview = result["chunks"][:PREVIEW]

        return jsonify({
            "ok": True,
            "report": result["report"],
            "config": result["config"],
            "preview": [{
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "pages": c["pages"],
                "length": len(c["text"]),
                "text": c["text"][:500]
            } for c in preview]
        })

    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    except FileNotFoundError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 404

    except Exception as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500


@app.route("/analyze", methods=["POST"])
def running():
    form = request.form
    group = form.get("group", "Contenuto")

    if group not in GROUPS_AVAILABLE:
        return render_template("index.html", results=None, error=f"Inferenza sul gruppo {group} non ancora disponibile.")

    chunks_path = form.get("chunks", "out/chunks.jsonl").strip()
    if not chunks_path:
        return render_template("index.html", results=None, error=f"Seleziona una strategia di chunking.")
    min_share = get_float(form, "min_share", 0.0)

    try:
        results, todo = run_contents(
            chunks_path=chunks_path,
            threshold=get_float(form, "threshold", 0.92),
            cand_threshold=get_float(form, "cand_threshold", 0.85),
            min_chars=max(0, get_int(form, "min_chars", 40)),
            k=max(0, get_int(form, "k", 0)),
            k_max=max(3, get_int(form, "k_max", 12)),
            min_share=min_share if min_share > 0 else None,
            expected_topics=parse_topics(form.get("expected_topics")),
            topic_threshold=get_float(form, "topic_threshold", 0.55),
            judge=(form.get("judge") or "").strip() or None,
            max_pairs=max(1, get_int(form, "max_pairs", 10)),
            max_show=get_int(form, "max_show", 50)
        )
    except Exception as e:
        return render_template("index.html", results=None, error=f"{type(e).__name__}: {e}")

    job_id = start_job(todo, results) if todo else None

    return render_template("contents.html", results=results, job_id=job_id, error=None)

@app.route("/status/<job_id>", methods=["GET"])
def status(job_id:str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job: 
            return jsonify(status="unknown", error="Job non trovato"), 404

        return jsonify({
            "status": job["status"],
            "done": job["done"],
            "total": job["total"],
            "elapsed": job["elapsed"],
            "error": job["error"],
            "items": dict(job["items"]),
            "coerenza": job["coerenza"]
        })


if __name__ == "__main__":
    app.run(debug=True)

