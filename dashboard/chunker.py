from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import numpy as np
import statistics
import logging
import pypdf
import json
import time
import re

DIRECTORY = "../RAG-Multi-Corpus/datasets/ZX Bank"
EMBEDDER = "BAAI/bge-m3"
FORMAT = "pdf"
MODE = "fixed"                  # Alternative: fixed / recursive / semantic
SIZE = 500                      # Dimensione massima
OVERLAP = 50                    # Fixed
MIN_SIZE = 150                  # Minimo prima del taglio semantico
SEM_PERCENTILE = 25
PREVIEW = 3                     # Numero chunk da mostrare 
OUTPUT = f"out/chunks_{MODE}.jsonl"

"""
    Extracting text from PDFs splits into chunks with three strategies:
        - fixed: size-controlled segments, with configurable overlay
        - recursive: progressive cuts that favor separators and textual boundaries
        - semantic: grouping sentences based on the similarity of their embeddings

    For each chunk: source identifier and document, range in text, source pages and s method parameters.
    Output: jsonl file
"""

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ChunkConfig:
    mode: str = MODE
    size: int = SIZE
    overlap: int = OVERLAP
    min_size: int = MIN_SIZE
    sem_percentile: float = SEM_PERCENTILE
    embed_model: str = EMBEDDER
    batch_size: int = 32

    def __post_init__(self):
        if self.mode not in {"fixed", "recursive", "semantic"}:
            raise ValueError("mode: values are fixed / recursive / semantic")

        for name in ("size", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name}: positive integer")

        if self.mode == "fixed":
            if isinstance(self.overlap, bool) or not isinstance(self.overlap, int) or not 0 <= self.overlap < self.size:
                raise ValueError("overlap: integer in [0, size-1]")

        if self.mode == "semantic":
            if isinstance(self.min_size, bool) or not isinstance(self.min_size, int) or not 0 <= self.min_size <= self.size:
                raise ValueError("min_size: integer in [0, size]")

            if not np.isfinite(self.sem_percentile) or not 0 <= self.sem_percentile <= 100:
                raise ValueError("sem_percentile: number in [0, 100]")

            if not isinstance(self.embed_model, str) or not self.embed_model.strip():
                raise ValueError("embed_model: not empty")

# Utility

# spazi ai bordi
def trim_span(text, start, end):
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1

    return start, end


# (method 1) Fixed
def fixed_spans(text:str, size=SIZE, overlap=OVERLAP):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Request: size > 0 & 0 <= overlap < size")
    
    spans = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))

        if end < len(text): 
            search_start = start + size // 2
            split = max(
                text.rfind(" ", search_start, end),
                text.rfind("\n", search_start, end)
            )

            if split > start: 
                end = split

        spans.append((start, end))
        if end == len(text): 
            break

        # Avanzamento
        start = max(end - overlap, start + 1)
    return spans


# (method 2) Recursive
def recursive_spans(text: str, size=SIZE, seps=("\n\n", "\n", ". ", " ")):
    if size <= 0:
        raise ValueError("size: positive number")
    if any(not separator for separator in seps):
        raise ValueError("separators: not empty")
    

    def split(start, end, level):
        if start >= end:
            return []
        
        if end - start <= size: 
            return [(start, end)]

        if level >= len(seps): 
            return [(i, min(i + size, end)) for i in range(start, end, size)]

        separator = seps[level]
        pieces = []
        position = start

        # Divide il segmento usando il separatore corrente.
        while position < end:
            found = text.find(separator, position, end)

            if found == -1:
                pieces.append((position, end))
                break

            piece_end = found + len(separator)
            pieces.append((position, piece_end))
            position = piece_end

        # Prova ad aggregare pezzi consecutivi senza superare la dimensione target.
        output = []
        current = None

        for piece_start, piece_end in pieces:
            if piece_end - piece_start > size:
                if current is not None:
                    output.append(current)
                    current = None

                output.extend(split(piece_start, piece_end, level + 1))
                continue

            if current is None:
                current = (piece_start, piece_end)
            elif piece_end - current[0] <= size:
                current = (current[0], piece_end)
            else:
                output.append(current)
                current = (piece_start, piece_end)

        if current is not None:
            output.append(current)
        return output
    
    return split(0, len(text), 0)


# (method 3) Semantic

@lru_cache(maxsize=2)
def get_embedder(model_name):
    return SentenceTransformer(model_name)

def embed(texts, model_name=EMBEDDER, batch_size=32):
    if not texts:
        raise ValueError("Any texts")

    model = get_embedder(model_name)
    X = np.asarray(model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("embedding: not valid")
    
    return X

def sentence_units(text:str, size=SIZE):
    units, start = [], 0
    for match in re.finditer(r"(?<=[.!?])\s+|\n\s*\n", text):
        if match.start() > start:
            units.append((start, match.start()))
        start = match.end()

    if start < len(text):
        units.append((start, len(text)))

    output = []
    for start, end in units:
        if end - start > size:
            output.extend((start + a, start + b) for a, b in recursive_spans(text[start:end], size=size))
        else:
            output.append((start, end))

    return [(start, end) for start, end in output if text[start:end].strip()]

def semantic_spans(text, config):
    units = sentence_units(text, config.size)
    if len(units) < 2:
        return units

    X = embed([text[start:end] for start, end in units], config.embed_model, config.batch_size)
    similarities = np.clip(np.sum(X[:-1] * X[1:], axis=1), -1.0, 1.0)
    cutoff = float(np.percentile(similarities, config.sem_percentile))

    spans = []
    current_start, current_end = units[0]
    for index, (start, end) in enumerate(units[1:], start=1):
        too_big = end - current_start > config.size
        topic_shift = similarities[index - 1] < cutoff and current_end - current_start >= config.min_size

        if too_big or topic_shift:
            spans.append((current_start, current_end))
            current_start = start

        current_end = end

    spans.append((current_start, current_end))
    return spans


# Seleziono il chunker
def get_spans(text:str, config):
    if not text.strip():
        return []

    if config.mode == "fixed":
        return fixed_spans(text, config.size, config.overlap)
    if config.mode == "recursive":
        return recursive_spans(text, config.size)
    return semantic_spans(text, config)


# Ricerca dei documenti 
def find_documents(directory):
    root = Path(directory).expanduser().resolve()

    if not root.is_dir():
        raise NotADirectoryError(f"directory: not valid {root}")

    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")
    if not files:
        raise FileNotFoundError(f"any file in {root}")
    return root, files

def extract_pdf(file):
    pieces, pages = [], []
    offset = 0

    with Path(file).open("rb") as fp:
        reader = pypdf.PdfReader(fp)
        for number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""

            if not page_text.strip():
                continue

            if pieces:
                offset += 2

            start = offset
            pieces.append(page_text)
            offset += len(page_text)
            pages.append({"page": number, "start": start, "end": offset})

    return "\n\n".join(pieces), pages


# Creazione chunks
def make_chunks(root, files, config):
    chunks = []
    report = {
        "documents_found": len(files), "documents_processed": 0, "documents_empty": 0,
        "documents_failed": 0, "errors": []
    }

    for file in files:
        doc_id = file.relative_to(root).as_posix()

        try:
            text, pages = extract_pdf(file)
        except Exception as exc:
            report["documents_failed"] += 1
            report["errors"].append({
                "doc_id": doc_id, "error": f"{type(exc).__name__}: {exc}"
            })
            logger.warning("Reading error %s: %s", doc_id, exc)
            continue

        if not text.strip():
            report["documents_empty"] += 1
            logger.warning("PDF without text: %s", doc_id)
            continue

        spans = get_spans(text, config)
        chunk_number = 0
        for start, end in spans:
            start, end = trim_span(text, start, end)

            if start == end:
                continue

            source_pages = [page["page"] for page in pages if page["start"] < end and page["end"] > start]
            chunk = {
                "chunk_id": f"{doc_id}::{chunk_number:04d}",
                "doc_id": doc_id,
                "start": start,
                "end": end,
                "text": text[start:end],
                "pages": source_pages,
                "chunker": config.mode,
                "chunk_size": config.size,
                "source_format": "pdf"
            }

            if config.mode == "fixed":
                chunk["overlap"] = config.overlap
            elif config.mode == "semantic":
                chunk.update({
                    "min_size": config.min_size,
                    "sem_percentile": config.sem_percentile,
                    "embed_model": config.embed_model
                })

            chunks.append(chunk)
            chunk_number += 1

        report["documents_processed"] += 1
    return chunks, report

# Output

def save_chunks(chunks, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fp:
        for chunk in chunks:
            fp.write(json.dumps(chunk, ensure_ascii=True)+"\n")
    return path

def run_chunking(directory=DIRECTORY, config=None, output=None):
    config = config if config is not None else ChunkConfig()
    started = time.perf_counter()

    root, files = find_documents(directory)
    chunks, report = make_chunks(root, files, config)

    lengths = [len(chunk["text"]) for chunk in chunks]

    report.update({
        "mode": config.mode,
        "chunk_size": config.size,
        "n_chunks": len(chunks),
        "lengths": {
            "min": min(lengths, default=0),
            "median": float(statistics.median(lengths)) if lengths else 0.0,
            "max": max(lengths, default=0)
        }
    })

    output_path = save_chunks(chunks, output) if output is not None else None

    report["output"] = str(output_path) if output_path is not None else None
    report["seconds"] = round(time.perf_counter() - started, 2)
    return {"chunks": chunks, "report": report}

def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    config = ChunkConfig()
    output = OUTPUT or f"out/chunks_{config.mode}.jsonl"

    result = run_chunking(DIRECTORY, config, output)

    print(f"\n--- Statistics ---\n"
          f"{json.dumps(result["report"], indent=2, ensure_ascii=False)}\n")

    if PREVIEW > 0:
        print(f"--- Preview ---")
        for chunk in result["chunks"][:PREVIEW]:
            print(f"\n-- {chunk['chunk_id']} ({len(chunk['text'])} chars; pages {chunk['pages']})\n"
                  f"{chunk['text'][:300]}")

if __name__ == "__main__":
    main()


