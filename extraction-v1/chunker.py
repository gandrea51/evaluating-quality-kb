from pathlib import Path
import statistics
import pypdf
import json

DIRECTORY = "../RAG-Multi-Corpus/datasets/ZX Bank"
FORMAT = "pdf"
CHUNKER = "recursive"
SIZE = 500                      # Parametri recursive
OVERLAP = 50                    # Parametro usato dal fixed chunker
OUTPUT = "out/chunks.jsonl"
PREVIEW = 3                     # Numero chunk da mostrare 

# Chunkers

def fixed_spans(text: str, size: int = 500, overlap: int = 50):
    '''Divisione in chunk di size dim, con overlap'''
    n, spans, start = len(text), [], 0

    while start < n:
        end = min(start + size, n)
        if end < n:     # Non è la fine ? cerco un separatore 
            space = text.rfind(" ", start + size // 2, end)
            newline = text.rfind("\n", start + size // 2, end)
            splits = max(space, newline)
            if splits != -1: end = splits

        spans.append((start, end))
        if end >= n: break
        start = max(end - overlap, start + 1)
    return spans

def recursive_spans(text: str, size: int = 500, seps=("\n\n", "\n", ". ", " ")):
    '''Divisione ricorsiva del testo con separatori più granulari'''
    def recursion(start, end, level):
        # Il segmento è già abbastanza piccolo
        if end - start <= size: return [(start, end)]

        # Abbiamo esaurito i separatori: fallback a chunk di dimensione fissa.
        if level >= len(seps): return [(i, min(i + size, end)) for i in range(start, end, size)]

        separator = seps[level]
        pieces = []
        position = start

        # Divide il segmento usando il separatore corrente.
        while position < end:
            sep_pos = text.find(separator, position, end)

            if sep_pos == -1:
                # Non ci sono altri separatori: aggiunge tutto ciò che rimane.
                pieces.append((position, end)); break

            piece_end = sep_pos + len(separator)
            pieces.append((position, piece_end))
            position = piece_end

        # Prova ad aggregare pezzi consecutivi senza superare la dimensione target.
        output = []
        current = None

        for piece_start, piece_end in pieces:
            piece_size = piece_end - piece_start
            # Se il singolo pezzo è troppo grande, prova ricorsivamente con il separatore successivo.
            if piece_size > size:
                if current is not None:
                    output.append(current)
                    current = None

                output.extend(recursion(piece_start, piece_end, level + 1))
                continue

            # Primo pezzo nel nuovo
            if current is None:
                current = (piece_start, piece_end)
            # Aggiunta al chunk corrente ?
            elif piece_end - current[0] <= size:
                current = (current[0], piece_end)
            # Supererebbe size, chiudo il current
            else:
                output.append(current)
                current = (piece_start, piece_end)

        if current is not None:
            output.append(current)
        return output
    return recursion(0, len(text), 0)


# Seleziono il chunker
def get_spans(text: str):
    if CHUNKER == "fixed": return fixed_spans(text, SIZE, OVERLAP)
    if CHUNKER == "recursive": return recursive_spans(text, SIZE)

    raise ValueError(f"Chunker {CHUNKER} non riconosciuto. Usa: fixed o recursive.")

# Ricerca dei documenti 
def find_document():
    '''Cerca tutti documenti del formato specificato'''
    root = Path(DIRECTORY)
    if not root.exists(): raise ValueError(f"Directory inesistente: {root}.")

    files = sorted(root.rglob(f"*.{FORMAT}"))
    if not files:
        raise FileNotFoundError(f"Nessun file .{FORMAT}.")
    return root, files

# Creazione chunks
def make_chunks(root: Path, files: list):
    chunks = []
    for file in files:
        doc_id = file.relative_to(root).as_posix()
        pdf_pieces = []
        try:
            with open(file, "rb") as fp:
                reader = pypdf.PdfReader(fp)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text: pdf_pieces.append(page_text)

            text = "\n\n".join(pdf_pieces)
        except Exception as e:
            print(f"Errore lettura PDF {doc_id}: {e}"); continue

        chunk_number = 0
        for start, end in get_spans(text):
            piece = text[start:end].strip()
            if not piece: continue

            chunk = {
                "chunk_id": f"{doc_id}::{chunk_number:04d}",
                "doc_id": doc_id, "start": start, "end": end, "text": piece,
                "chunker": CHUNKER, "chunk_size": SIZE, "source_format": FORMAT
            }
            chunks.append(chunk)
            chunk_number += 1
    return chunks


# Salvataggio
def saving(chunks):
    path = Path(OUTPUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for chunk in chunks:
            fp.write(json.dumps(chunk, ensure_ascii=True)+"\n")
    return path

# Statistiche
def stats(files, chunks, output):
    if not chunks:
        print("Nessun chunk prodotto."); return

    lengths = [len(chunk["text"]) for chunk in chunks]
    print(f"\n--- Statistiche ---\n"
          f"Documenti letti: {len(files)}\n"
          f"Chunk prodotti: {len(chunks)}\n"
          f"Chunker: {CHUNKER}\n"
          f"Chunk size: {SIZE}")
    if CHUNKER == "fixed": print(f"Overlap: {OVERLAP}")
    print(f"Lunghezza chunk: min {min(lengths)} - mediana {int(statistics.median(lengths))} - max {max(lengths)} chars\n"
          f"Output in: {output}")

def preview(chunks):
    if PREVIEW <= 0: return
    print(f"\n--- Preview ---")
    for chunk in chunks[:PREVIEW]:
        print(f"\n-- {chunk['chunk_id']} ({len(chunk['text'])} char) --")
        print(chunk["text"][:300])


def main() -> None:
    root, files = find_document()
    chunks = make_chunks(root, files)
    output = saving(chunks)
    stats(files, chunks, output)
    preview(chunks)

if __name__ == "__main__":
    main()
