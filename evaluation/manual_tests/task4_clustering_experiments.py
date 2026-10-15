"""Task 4: Systematic comparison of clustering and representation alternatives on LDC2021S10."""

import json
from pathlib import Path
import numpy as np
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from Project.ssl_pipeline.rttm import parse_rttm, RTTMSpeakerTurn
from Project.ssl_pipeline.spectral_clustering import build_affinity_matrix
from Project.ssl_pipeline.timeline_reconstruction import reconstruct_speaker_timeline
from Project.ssl_pipeline.dataset_evaluation import DiarizationScorer, EvaluationProtocol

def main():
    root = Path(__file__).resolve().parents[2]
    eval_dir = root / "evaluation"
    res_dir = eval_dir / "manual_tests" / "results" / "test_C_two_speakers"
    pipe_dir = res_dir / "pipeline"

    ref_rttm_path = eval_dir / "LDC2021S10.rttm"
    ref_turns = parse_rttm(ref_rttm_path, allow_overlap=True)
    audio_dur = 120.0

    # Load cached normalized embeddings from WavLM Base+
    emb_data = np.load(pipe_dir / "normalized_embeddings.npz")
    vectors = emb_data["vectors"]
    starts = emb_data["starts"]
    ends = emb_data["ends"]
    n_segments = len(vectors)

    # 1. Identify unambiguous reference segments
    clipped_ref = [t for t in ref_turns if t.start < audio_dur]
    unambiguous_indices = []
    unambiguous_labels = []

    for idx, (s, e) in enumerate(zip(starts, ends)):
        mid = (s + e) / 2.0
        active = [t.speaker_id for t in clipped_ref if t.start <= mid < t.end]
        if len(active) == 1:
            unambiguous_indices.append(idx)
            unambiguous_labels.append(0 if active[0] == "speaker117" else 1)

    unambiguous_indices = np.array(unambiguous_indices)
    unambiguous_labels = np.array(unambiguous_labels)
    n_unambig = len(unambiguous_indices)
    print(f"Total Segments: {n_segments}, Unambiguous Segments: {n_unambig}")
    print(f"  speaker117: {np.sum(unambiguous_labels == 0)}, speaker118: {np.sum(unambiguous_labels == 1)}")

    # Define Clustering Methods to test:
    # 1. Baseline: Spectral Clustering (assign_labels='kmeans', threshold=0.0)
    # 2. Alternative 1: Spectral Clustering (assign_labels='discretize')
    # 3. Alternative 2: Agglomerative Hierarchical Clustering (cosine distance, average linkage)
    # 4. Alternative 3: Agglomerative Hierarchical Clustering (cosine distance, complete linkage)
    # 5. Alternative 4: Spherical / Cosine KMeans on L2-normalized embeddings

    v_norm = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    sim = v_norm @ v_norm.T

    methods = {
        "Baseline: Spectral (kmeans)": lambda: SpectralClustering(
            n_clusters=2, affinity="precomputed", assign_labels="kmeans", random_state=42
        ).fit_predict(build_affinity_matrix(sim, threshold=0.0)),
        
        "Alt 1: Spectral (discretize)": lambda: SpectralClustering(
            n_clusters=2, affinity="precomputed", assign_labels="discretize", random_state=42
        ).fit_predict(build_affinity_matrix(sim, threshold=0.0)),
        
        "Alt 2: AHC (cosine, average)": lambda: AgglomerativeClustering(
            n_clusters=2, metric="cosine", linkage="average"
        ).fit_predict(v_norm),
        
        "Alt 3: AHC (cosine, complete)": lambda: AgglomerativeClustering(
            n_clusters=2, metric="cosine", linkage="complete"
        ).fit_predict(v_norm),
        
        "Alt 4: KMeans (L2 normalized)": lambda: KMeans(
            n_clusters=2, random_state=42, n_init=10
        ).fit_predict(v_norm),
    }

    scorer = DiarizationScorer(EvaluationProtocol(collar=0.0, skip_overlap=False, uem_mode="full_file"))

    results = []

    for name, cluster_fn in methods.items():
        try:
            preds = cluster_fn()
        except Exception as exc:
            print(f"Failed {name}: {exc}")
            continue

        # (A) Diagnostic comparison on unambiguous segments
        preds_unambig = preds[unambiguous_indices]
        acc_p0 = np.mean(preds_unambig == unambiguous_labels)
        acc_p1 = np.mean((1 - preds_unambig) == unambiguous_labels)
        best_purity = max(acc_p0, acc_p1)
        c0_count = int(np.sum(preds == 0))
        c1_count = int(np.sum(preds == 1))

        # (B) Build RTTM turns through timeline reconstruction
        from Project.ssl_pipeline.spectral_clustering import ClusteredEmbedding
        clustered_embeds = [
            ClusteredEmbedding(
                segment_id=i,
                start=float(starts[i]),
                end=float(ends[i]),
                cluster_id=int(preds[i]),
                vector=vectors[i],
            )
            for i in range(n_segments)
        ]

        timeline = reconstruct_speaker_timeline(clustered_embeds)
        hyp_turns = [
            RTTMSpeakerTurn(
                recording_id="DH_0052",
                channel_id="1",
                start=round(t.start, 3),
                duration=round(t.duration, 3),
                speaker_id=t.speaker_id,
            )
            for t in timeline
        ]

        eval_res = scorer.score(ref_turns, hyp_turns, uri="DH_0052", uem_end=audio_dur)

        results.append({
            "method": name,
            "cluster_sizes": [c0_count, c1_count],
            "unambiguous_accuracy_percent": round(best_purity * 100.0, 2),
            "der_percent": round(eval_res.der * 100.0, 2),
            "jer_percent": round(eval_res.jer * 100.0, 2) if eval_res.jer is not None else None,
            "missed_seconds": round(eval_res.missed_speech, 3),
            "false_alarm_seconds": round(eval_res.false_alarm, 3),
            "confusion_seconds": round(eval_res.speaker_confusion, 3),
            "turn_count": len(hyp_turns),
        })

    print("\n=== SYSTEMATIC CLUSTERING COMPARISON ON LDC2021S10 ===")
    print(f"{'Method':<30} | {'Clusters':<10} | {'Purity':<8} | {'DER %':<8} | {'Miss (s)':<9} | {'FA (s)':<8} | {'Conf (s)':<9}")
    print("-" * 95)
    for r in results:
        print(f"{r['method']:<30} | {str(r['cluster_sizes']):<10} | {r['unambiguous_accuracy_percent']:>6.1f}% | {r['der_percent']:>6.2f}% | {r['missed_seconds']:>7.3f}s | {r['false_alarm_seconds']:>6.3f}s | {r['confusion_seconds']:>7.3f}s")

    out_file = res_dir / "clustering_comparison.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved clustering experiment results to {out_file}")

if __name__ == "__main__":
    main()
