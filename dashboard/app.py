from flask import Flask, render_template, request, jsonify
from kb_analysis.core import judge_pairs
from kb_analysis.cleaning_area import run_contents
from kb_analysis.mapping_area import run_structure
from kb_analysis.utils import get_int, get_float
import threading
import time
import uuid
import os

"""
    Flask application for interactive analysis of KB chunks. Features: 
        - Interface: Home page and results template.
        - Routing: Send the request to content or structure analysis (now)
        - LLM Evaluation: Performs judgment of candidate pairs in a separate thread.
        - Monitoring: Exposes the status and partial results of jobs via an endpoint.

    Domain analyses and logic are delegated to the kb_analysis modules
"""

app = Flask(__name__)

JOBS = {}
OLLAMA_LOCK = threading.Lock()

os.environ.setdefault("MPLBACKEND", "Agg")

# (job llm)
def judge_worker(job_id, todo):
    job = JOBS[job_id]
    started = time.perf_counter()

    def on_result(index, pair):
        pair_key = f'{pair["a"]}|{pair["b"]}'

        job["items"][pair_key] = {
            "relation": pair["relation"], 
            "explanation": pair["explanation"]
        }
        job["done"] = index
        job["elapsed"] = round(time.perf_counter() - started, 1)

    try:
        with OLLAMA_LOCK:
            judge_pairs(todo["model"], todo["pairs"], todo["texts"], todo["ids"], todo["out_dir"], on_result=on_result)
        job["status"] = "done"
    
    except Exception as e:
        job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"

    finally:
        job["elapsed"] = round(time.perf_counter() - started, 1)

# (create job)
def start_job(todo):
    job_id = uuid.uuid4().hex

    JOBS[job_id] = {
        "status": "running", "done": 0, "total": len(todo["pairs"]), 
        "elapsed": 0, "items": {}, "error": None
    }
    thread = threading.Thread(target=judge_worker, args=(job_id, todo), daemon=True)
    thread.start()

    return job_id


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", results=None, error=None)


@app.route("/analyze", methods=["POST"])
def running():
    form = request.form
    group = form.get("group", "Contenuto")

    chunks_path = form.get("chunks", "out/chunks.jsonl")
    min_chars = get_int(form, "min_chars", 40)

    # Group Two
    if group == "Struttura":
        try:
            results = run_structure(
                chunks_path=chunks_path,
                k=get_int(form, "k", 0),
                k_max=get_int(form, "k_max", 12),
                min_chars=min_chars,
                short_chars=get_int(form, "short_chars", 100)
            )
        except Exception as e:
            return render_template("index.html", results=None, error=f"{type(e).__name__}: {e}")
        return render_template("mapping.html", results=results, error=None)

    # Group One
    try:
        results, todo = run_contents(
            chunks_path=chunks_path,
            threshold=get_float(form, "threshold", 0.92),
            cand_threshold=get_float(form, "cand_threshold", 0.85),
            min_chars=min_chars,
            judge=(form.get("judge") or "").strip() or None,
            max_pairs=get_int(form, "max_pairs", 10),
            max_show=get_int(form, "max_show", 50)
        )
    except Exception as e:
        return render_template("index.html", results=None, error=f"{type(e).__name__}: {e}")

    job_id = start_job(todo) if todo else None

    return render_template("cleaning.html", results=results, job_id=job_id, error=None)

@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job: 
        return jsonify(status="unknown"), 404

    return jsonify({
        **job, 
        "items": dict(job["items"])
    })


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5001)

