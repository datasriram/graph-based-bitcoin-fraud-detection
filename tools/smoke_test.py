from __future__ import annotations

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from core.pipeline import DEFAULT_THRESHOLD, build_graph_from_csv_bytes, detect_rings, run_inference
from model.graphsage import load_model


def main() -> None:
    base = BASE
    graph_mode = sys.argv[1] if len(sys.argv) > 1 else "auto"
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_THRESHOLD
    graph = build_graph_from_csv_bytes(
        (base / "data" / "sample_nodes.csv").read_bytes(),
        (base / "data" / "sample_edges.csv").read_bytes(),
        base / "artifacts",
        graph_mode=graph_mode,
    )
    model = load_model(base / "artifacts" / "graphsage_model.pt")
    probs, preds = run_inference(model, graph, threshold=threshold)
    rings, communities = detect_rings(graph, probs, preds, threshold=threshold)
    print("x.shape", tuple(graph.x.shape))
    print("edge_index.shape", tuple(graph.edge_index.shape))
    print("graph_method", graph.graph_method)
    print("threshold", threshold)
    print("nodes", graph.num_nodes)
    print("edges", graph.num_edges)
    print("predicted_illicit", int(preds.sum()))
    print("rings", len(rings))
    print("communities", len(communities))


if __name__ == "__main__":
    main()
