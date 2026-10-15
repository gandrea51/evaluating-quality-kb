from flask import Flask, render_template, request, jsonify
from embedding import judge_pairs
from content import run_contents
import threading
import time
import uuid
import os

JOBS = {}
OLLAMA_LOCK = threading.Lock()

os.environ.setdefault("MPLBACKEND", "Agg")
app = Flask(__name__)

def judge_worker(job_id, todo):
    job = JOBS[job_id]
    t0 = time.perf_counter()

    def on_result(n, p):
        job["items"][f'{p["a"]}|{p["b"]}'] = {"relation": p["relation"], "explanation": p["explanation"]}
        job["done"] = n
        job["elapsed"] = round(time.perf_counter()-t0, 1)

    try:
        with OLLAMA_LOCK:
            judge_pairs(todo["model"], todo["pairs"], todo["texts"], todo["ids"], todo["out_dir"], on_result=on_result)
        job["status"] = "done"
    except Exception as e:
        job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
    job["elapsed"] = round(time.perf_counter()-t0, 1)
    

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", results=None, error=None)


@app.route("/analyze", methods=["POST"])
def running():
    f = request.form
    try:
        results, todo = run_contents(
            chunks_path=f.get("chunks", "out/chunks.jsonl"),
            threshold=float(f.get("threshold", 0.92)),
            cand_threshold=float(f.get("cand_threshold", 0.85)),
            min_chars=int(f.get("min_chars", 40)),
            judge=(f.get("judge") or "").strip() or None,
            max_pairs=int(f.get("max_pairs", 10)),
            max_show=int(f.get("max_show", 50))
        )
    except Exception as e:
        return render_template("index.html", results=None, error=f"{type(e).__name__}: {e}")

    job_id = None
    if todo:
        job_id = uuid.uuid4().hex
        JOBS[job_id] = {"status": "running", "done": 0, "total": len(todo["pairs"]), "items": {}, "error": None}
        threading.Thread(target=judge_worker, args=(job_id, todo), daemon=True).start()

    
    return render_template("contenuto.html", results=results, job_id=job_id, error=None)

@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job: return jsonify(status="unknown"), 404

    return jsonify({**job, "items": dict(job["items"])})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

