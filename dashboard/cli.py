from pathlib import Path
from collections import Counter
from kb_analysis.core import build_metrics, build_valid_pairs, candidate_summary, cosine_matrix, find_contradiction_candidates, generate_embeddings, judged_summary
from kb_analysis.core import judge_pairs, mark_timing, nearest_neighbor_sim, prepare_chunks, redundancy_at, redundancy_sweep, write_candidate_pairs
from kb_analysis.utils import read_jsonl, pca3, plot_3d, plot_hist
import numpy as np
import argparse
import json
import time

THRESHOLDS = [0.8, 0.85, 0.9, 0.92, 0.95, 0.98, 0.99]

"""
    Analysis: Redundancies, possible contradictions in chunks.
    Calculation: Similarity metrics; Identification: Candidate pairs; Reports and Charts; (bonus) Ollama to evaluate selected pairs.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Analysis of redundancy and inconsistency for a knowledge base extracted from a PDF")
    parser.add_argument("--chunks", default="out/chunks.jsonl")
    parser.add_argument("--out-dir", default="out")
    parser.add_argument("--threshold", type=float, default=0.92, help="cosine beyond which two chunks are near-duplicates")
    parser.add_argument("--cand-threshold", type=float, default=0.85, help="minimum cosine for candidate pairs with discordance")
    parser.add_argument("--min-chars", type=int, default=40, help="Chunks containing fewer than a certain number of meaningful characters (letters and numbers) are fragments: they are excluded from the metrics and counted separately")
    parser.add_argument("--judge", help="Ollama model that evaluates candidate pairs (e.g. llama3.1)")
    parser.add_argument("--max-pairs", type=int, default=20, help="how many candidate couples to put forward for judging")
    parser.add_argument("--show", action="store_true", help="Open the interactive 3D graph")
    return parser.parse_args()

# (report)

def build_report(metrics, sweep, data, groups, candidates, threshold,):
    texts = data["texts"]
    ids = data["ids"]
    fragments = data["fragments"]
    n = data["n"]

    def snippet(index):
        return texts[index][:110].replace("\n", " ")

    lines = [
        json.dumps(metrics, indent=2, ensure_ascii=False), "", "-- threshold calibration (redundancy as the cosine varies) --",
        (f"{'soglia':>7} {'chunk in gruppi':>16} {'gruppi':>7} {'max':>4} {'rate':>7} {'coppie tra doc':>15}")
    ]

    lines.extend((f"{item['threshold']:>7.2f} {item['redundant_chunks']:>16} {item['n_groups']:>7} "
                  f"{item['max_group_size']:>4} {item['redundancy_rate']:>7.3f} {item['pairs_cross_doc']:>15}") for item in sweep
    )

    lines.extend(["",(f"-- fragments: chunks with less than {data['min_chars']} chars useful ({int(fragments.sum())} su {n}) --"),])

    fragment_counter = Counter(data["doc_names"][doc_index] for doc_index in data["docs"][fragments])
    lines.extend(f"{count:>4}  {name}" for name, count in fragment_counter.most_common(15))
    lines.extend(f"   eg. {ids[index]} | {texts[index][:60]!r}" for index in np.flatnonzero(fragments)[:8])

    lines.extend(["", f"-- groups of near-duplicates (soglia {threshold}) --",])
    for group in sorted(groups, key=len, reverse=True)[:25]:
        lines.append(f"gruppo di {len(group)}:")
        lines.extend(f"   {ids[index]} | {snippet(index)}" for index in group)

    lines.extend(["", "-- pairs at risk of incompatibility (first 30; those across different documents come first) --",])
    for pair in candidates[:30]:
        explanation = pair.get("explanation")
        relation = pair.get("relation", "")

        lines.append((f"sim {pair['sim']:.3f} cross_doc={pair['cross_doc']} num_diff={pair['numeric_diff']} {relation}\n"
                      f"   A {pair['a']} | {snippet(pair['i'])}\n"
                      f"   B {pair['b']} | {snippet(pair['j'])}" + (f"\n   -> {explanation}" if explanation else "")))
    return "\n".join(lines)

# (main)
def run_analysis(args):
    run_started = time.perf_counter()
    phase_started = run_started

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # (chunk) Read-Preparation
    chunks = read_jsonl(args.chunks)
    data = prepare_chunks(chunks, min_chars=args.min_chars)
    data["min_chars"] = args.min_chars
    phase_started = mark_timing("read & prepation chunk", phase_started)

    # (embedding)
    X = generate_embeddings(data["texts"])
    phase_started = mark_timing("embedding", phase_started)

    similarity = cosine_matrix(X)
    phase_started = mark_timing("similarity matrix", phase_started)

    # (pairs e knn)
    I, J = build_valid_pairs(data)
    S = similarity[I, J]

    nn = nearest_neighbor_sim(I, J, S, data["n"])
    phase_started = mark_timing("pairs e knn", phase_started)

    # (redundancy)
    redundancy_metrics, groups, edges = redundancy_at(threshold=args.threshold, I=I, J=J, S=S, docs=data["docs"], n=data["n"], n_eff=data["n_effective"])
    sweep = redundancy_sweep(thresholds=THRESHOLDS, I=I, J=J, S=S, docs=data["docs"], n=data["n"], n_eff=data["n_effective"])
    phase_started = mark_timing("redundancy metrics", phase_started)

    # (discordance)
    candidates = find_contradiction_candidates( texts=data["texts"], ids=data["ids"], docs=data["docs"], I=I, J=J, S=S, cand_threshold=args.cand_threshold)
    discordance_metrics = candidate_summary(candidates)

    # (llm)
    if args.judge and candidates:
        selected = candidates[: args.max_pairs]
        judge_pairs(model=args.judge, pairs=selected, texts=data["texts"], ids=data["ids"], out_dir=out_dir)
        discordance_metrics.update(judged_summary(selected))

    # Metriche, candidati e report
    metrics = build_metrics(model_name="sentence-transformers", data=data, redundancy_metrics=redundancy_metrics, discordance_metrics=discordance_metrics)

    metrics_path = out_dir / "embedding_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    candidate_path = write_candidate_pairs(candidates, out_dir)

    phase_started = mark_timing("discordance, judges, metrics", phase_started)

    report = build_report(metrics=metrics, sweep=sweep, data=data, groups=groups, candidates=candidates, threshold=args.threshold)

    report_path = out_dir / "embedding_report.txt"
    report_path.write_text(report, encoding="utf-8")
    phase_started = mark_timing("report", phase_started)

    # (graphs)
    C, explained_variance = pca3(X)
    phase_started = mark_timing("PCA", phase_started)

    plotted_candidates = (candidates[: args.max_pairs] if args.judge else candidates[:60])

    plot_3d(C=C, var=explained_variance, docs=data["docs"], doc_names=data["doc_names"], red_groups=groups, red_edges=edges,
        cand_pairs=plotted_candidates, thr=args.threshold, path=out_dir / "latent_space_3d.png", show=args.show, frag_mask=data["fragments"])
    phase_started = mark_timing("3d graph", phase_started)

    plot_hist(nn=nn[nn > -1], thr=args.threshold, cand_thr=args.cand_threshold, path=out_dir / "similarity_distribution.png", show=args.show)
    phase_started = mark_timing("distribution graph", phase_started)

    print(
        f"chunk: {data['n']} (analyze {data['n_effective']}, "
        f"fragments {int(data['fragments'].sum())}) | documents: {len(data['doc_names'])} | variance explained by the first 3 components: {explained_variance[:3].sum():.0%}\n\n"
        f"Redundancy (threshold {args.threshold}): {redundancy_metrics['redundant_chunks']} chunk in {redundancy_metrics['n_groups']} groups "
        f"(max {redundancy_metrics['max_group_size']}), rate {redundancy_metrics['redundancy_rate']:.3f}, "
        " pairs across different documents: {redundancy_metrics['pairs_cross_doc']}\n"
        f"Discordance: {discordance_metrics['candidate_pairs']} candidate pairs ({discordance_metrics['candidate_numeric_diff']} with different numbers, "
        f"{discordance_metrics['candidates_cross_doc']} among documents)"
        + (f"; judges {discordance_metrics['judged']}: contraddictions {discordance_metrics.get('judged_contraddizione', 0)}" if args.judge else "  [use --judge to check them]")
        + "\n\n"
        f"write {out_dir}/: embedding_metrics.json, embedding_report.txt, {candidate_path.name}, latent_space_3d.png, similarity_distribution.png"
    )

    mark_timing("total time", run_started)

    return {
        "metrics": metrics,
        "groups": groups,
        "candidates": candidates,
        "outputs": {
            "metrics": metrics_path,
            "report": report_path,
            "candidates": candidate_path,
        },
    }


def main():
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
