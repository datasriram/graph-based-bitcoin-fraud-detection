from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

LOGGER = logging.getLogger(__name__)

FEATURE_COUNT = 166
FEATURE_NAMES = ["time_step"] + [f"f{i}" for i in range(165)]
DEFAULT_THRESHOLD = 0.942
DEFAULT_K_NEIGHBORS = 6
LABEL_COLUMNS = {"class", "label", "y", "target", "ground_truth"}


class PipelineInputError(ValueError):
    """Raised when uploaded data cannot be converted into a valid PyG graph."""


@dataclass
class BuiltGraph:
    nodes_df: pd.DataFrame
    tx_ids: List[str]
    tx_to_idx: Dict[str, int]
    x: torch.Tensor
    edge_index: torch.Tensor
    src: List[int]
    dst: List[int]
    graph_method: str
    warnings: List[str]
    debug: Dict[str, Any]

    @property
    def num_nodes(self) -> int:
        return len(self.tx_ids)

    @property
    def num_edges(self) -> int:
        return len(self.src)


def _canonical_txid(value: Any) -> str:
    if pd.isna(value):
        raise PipelineInputError("txId contains missing values.")
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        if float(value).is_integer():
            return str(int(value))
    text = str(value).strip()
    if not text:
        raise PipelineInputError("txId contains empty values.")
    try:
        as_float = float(text)
        if as_float.is_integer():
            return str(int(as_float))
    except ValueError:
        pass
    return text


def _read_csv(data: bytes, *, header: Optional[int] = 0) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(data), header=header)
    except pd.errors.EmptyDataError as exc:
        raise PipelineInputError("Uploaded CSV is empty.") from exc
    except Exception as exc:
        raise PipelineInputError(f"Could not parse CSV: {exc}") from exc


def _rename_common_aliases(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "txid": "txId",
        "tx_id": "txId",
        "transaction_id": "txId",
        "transaction": "txId",
        "time": "time_step",
        "timestep": "time_step",
        "time step": "time_step",
    }
    rename_map = {}
    for col in df.columns:
        text = str(col).strip()
        low = text.lower()
        rename_map[col] = aliases.get(low, text)
    return df.rename(columns=rename_map)


def _standardize_positional_nodes(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[1] not in (FEATURE_COUNT + 1, FEATURE_COUNT + 2):
        raise PipelineInputError(
            "nodes CSV must have txId plus 166 feature columns "
            "(txId,time_step,f0..f164), with an optional label/class column."
        )
    out = df.copy()
    columns = ["txId", *FEATURE_NAMES]
    if out.shape[1] == FEATURE_COUNT + 2:
        columns.append("class")
    out.columns = columns
    return out


def standardize_nodes_dataframe(df: pd.DataFrame, *, allow_positional: bool = True) -> pd.DataFrame:
    """Return columns txId,time_step,f0..f164 plus any optional label columns."""
    if df.empty:
        raise PipelineInputError("nodes CSV has no rows.")

    df = _rename_common_aliases(df.copy())
    has_required = "txId" in df.columns and all(col in df.columns for col in FEATURE_NAMES)

    if not has_required and "txId" in df.columns:
        non_feature_cols = {
            col for col in df.columns if str(col).strip().lower() in LABEL_COLUMNS
        }
        positional_cols = [col for col in df.columns if col not in non_feature_cols and col != "txId"]
        if len(positional_cols) >= FEATURE_COUNT:
            rename_map = {positional_cols[0]: "time_step"}
            rename_map.update({positional_cols[i + 1]: f"f{i}" for i in range(165)})
            df = df.rename(columns=rename_map)
            has_required = all(col in df.columns for col in ["txId", *FEATURE_NAMES])

    if not has_required and allow_positional:
        df = _standardize_positional_nodes(df)
        has_required = True

    if not has_required:
        missing = [col for col in ["txId", *FEATURE_NAMES] if col not in df.columns]
        raise PipelineInputError(f"nodes CSV is missing required columns: {missing[:8]}")

    keep_cols = ["txId", *FEATURE_NAMES]
    for col in df.columns:
        if str(col).strip().lower() in LABEL_COLUMNS and col not in keep_cols:
            keep_cols.append(col)
    out = df.loc[:, keep_cols].copy()

    out["txId"] = out["txId"].map(_canonical_txid)
    duplicate_mask = out["txId"].duplicated(keep="first")
    if duplicate_mask.any():
        dropped = int(duplicate_mask.sum())
        LOGGER.warning("Dropping %s duplicate txId rows.", dropped)
        out = out.loc[~duplicate_mask].copy()

    for col in FEATURE_NAMES:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if out.empty:
        raise PipelineInputError("nodes CSV has no usable txId rows after cleaning.")
    return out.reset_index(drop=True)


def read_nodes_csv(data: bytes) -> pd.DataFrame:
    first_pass = _read_csv(data, header=0)
    try:
        return standardize_nodes_dataframe(first_pass, allow_positional=False)
    except PipelineInputError:
        second_pass = _read_csv(data, header=None)
        return standardize_nodes_dataframe(second_pass, allow_positional=True)


def read_edges_csv(data: Optional[bytes]) -> Optional[pd.DataFrame]:
    if not data:
        return None
    df = _read_csv(data, header=0)
    if df.empty:
        return None
    lower_cols = {str(col).strip().lower() for col in df.columns}
    known_cols = {"txid1", "txid2", "src", "dst", "source", "target"}
    if not (lower_cols & known_cols):
        try:
            float(str(df.columns[0]))
            float(str(df.columns[1]))
            df = _read_csv(data, header=None)
        except (ValueError, TypeError):
            pass
    if df.shape[1] < 2:
        headerless = _read_csv(data, header=None)
        if headerless.shape[1] < 2:
            raise PipelineInputError("edges CSV must have at least two columns.")
        df = headerless
    df.columns = [str(col).strip() for col in df.columns]
    return df


def load_normalizer(artifact_dir: Path) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    candidates = [artifact_dir, artifact_dir.parent]
    col_min_path = next((p / "col_min.npy" for p in candidates if (p / "col_min.npy").exists()), None)
    col_rng_path = next((p / "col_rng.npy" for p in candidates if (p / "col_rng.npy").exists()), None)
    if col_min_path is None or col_rng_path is None:
        raise PipelineInputError(
            "Missing col_min.npy or col_rng.npy. Copy the training normalization artifacts "
            "into artifacts/ before starting the API."
        )

    col_min = np.load(col_min_path).astype(np.float32)
    col_rng = np.load(col_rng_path).astype(np.float32)
    if col_min.shape != (FEATURE_COUNT,) or col_rng.shape != (FEATURE_COUNT,):
        raise PipelineInputError(
            f"Normalizer shape mismatch: col_min={col_min.shape}, col_rng={col_rng.shape}, "
            f"expected ({FEATURE_COUNT},)."
        )
    col_rng[~np.isfinite(col_rng) | (col_rng == 0)] = 1.0

    median_path = next(
        (p / "col_median.npy" for p in candidates if (p / "col_median.npy").exists()),
        None,
    )
    col_median = None
    if median_path is not None:
        col_median = np.load(median_path).astype(np.float32)
        if col_median.shape != (FEATURE_COUNT,):
            col_median = None
    return col_min, col_rng, col_median


def normalize_features(nodes_df: pd.DataFrame, artifact_dir: Path) -> Tuple[np.ndarray, torch.Tensor, List[str]]:
    warnings: List[str] = []
    col_min, col_rng, col_median = load_normalizer(artifact_dir)
    feat = nodes_df.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float32)
    feat[~np.isfinite(feat)] = np.nan

    if np.isnan(feat).any():
        if col_median is not None:
            fill_values = col_median
            warnings.append("NaN values filled with saved training medians.")
        else:
            fill_values = np.nanmedian(feat, axis=0)
            fill_values = np.where(np.isfinite(fill_values), fill_values, col_min)
            warnings.append("NaN values filled with upload medians; save col_median.npy for exact training median fill.")
        rows, cols = np.where(np.isnan(feat))
        feat[rows, cols] = fill_values[cols]

    feat_norm = (feat - col_min) / col_rng
    feat_norm = np.nan_to_num(feat_norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x = torch.tensor(feat_norm, dtype=torch.float32)
    if x.ndim != 2 or x.shape[1] != FEATURE_COUNT:
        raise PipelineInputError(f"x has shape {tuple(x.shape)}, expected (N, {FEATURE_COUNT}).")
    return feat_norm, x, warnings


def _edge_columns(edges_df: pd.DataFrame) -> Tuple[str, str, str]:
    cols = list(edges_df.columns)
    lower = {str(col).lower(): col for col in cols}
    tx_pairs = [("txid1", "txid2"), ("src", "dst"), ("source", "target")]
    for left, right in tx_pairs:
        if left in lower and right in lower:
            mode = "index" if (left, right) in {("src", "dst"), ("source", "target")} else "txid"
            return lower[left], lower[right], mode
    return cols[0], cols[1], "auto"


def _looks_like_index_edges(left: pd.Series, right: pd.Series, num_nodes: int) -> bool:
    lnum = pd.to_numeric(left, errors="coerce")
    rnum = pd.to_numeric(right, errors="coerce")
    if lnum.isna().any() or rnum.isna().any():
        return False
    if not ((lnum % 1 == 0).all() and (rnum % 1 == 0).all()):
        return False
    return bool((lnum.between(0, num_nodes - 1).all()) and (rnum.between(0, num_nodes - 1).all()))


def _dedupe_edges(edge_pairs: Iterable[Tuple[int, int]]) -> Tuple[List[int], List[int]]:
    clean = sorted({(int(s), int(d)) for s, d in edge_pairs if int(s) != int(d)})
    if not clean:
        return [], []
    src, dst = zip(*clean)
    return list(src), list(dst)


def build_edges_from_upload(
    edges_df: Optional[pd.DataFrame],
    tx_to_idx: Dict[str, int],
    num_nodes: int,
    warnings: List[str],
) -> Tuple[List[int], List[int], str]:
    if edges_df is None or edges_df.empty:
        return [], [], "none"

    left_col, right_col, mode = _edge_columns(edges_df)
    left = edges_df[left_col]
    right = edges_df[right_col]

    if mode == "auto":
        mode = "index" if _looks_like_index_edges(left, right, num_nodes) else "txid"

    pairs: List[Tuple[int, int]] = []
    dropped = 0
    if mode == "index":
        left_num = pd.to_numeric(left, errors="coerce")
        right_num = pd.to_numeric(right, errors="coerce")
        for s, d in zip(left_num, right_num):
            if pd.isna(s) or pd.isna(d):
                dropped += 1
                continue
            si, di = int(s), int(d)
            if 0 <= si < num_nodes and 0 <= di < num_nodes:
                pairs.append((si, di))
            else:
                dropped += 1
        method = "uploaded_index_edges"
    else:
        for s, d in zip(left, right):
            try:
                stx = _canonical_txid(s)
                dtx = _canonical_txid(d)
            except PipelineInputError:
                dropped += 1
                continue
            if stx in tx_to_idx and dtx in tx_to_idx:
                pairs.append((tx_to_idx[stx], tx_to_idx[dtx]))
            else:
                dropped += 1
        method = "uploaded_txid_edges"

    if dropped:
        warnings.append(f"Dropped {dropped} uploaded edges whose endpoints were not in nodes CSV.")
    src, dst = _dedupe_edges(pairs)
    return src, dst, method


def build_knn_edges_same_timestep(
    nodes_df: pd.DataFrame,
    feat_norm: np.ndarray,
    *,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
) -> Tuple[List[int], List[int]]:
    if len(nodes_df) <= 1:
        return [], []

    k = max(1, int(k_neighbors))
    groups: Dict[str, List[int]] = {}
    for idx, value in enumerate(nodes_df["time_step"].to_numpy()):
        key = str(int(value)) if np.isfinite(value) and float(value).is_integer() else str(value)
        groups.setdefault(key, []).append(idx)

    edge_set: set[Tuple[int, int]] = set()
    for indices in groups.values():
        if len(indices) <= 1:
            continue
        local = np.asarray(indices, dtype=np.int64)
        features = feat_norm[local, 1:]
        neighbor_count = min(k, len(local) - 1)
        for row_idx, node_idx in enumerate(local):
            diff = features - features[row_idx]
            dist = np.einsum("ij,ij->i", diff, diff)
            dist[row_idx] = np.inf
            nearest = np.argpartition(dist, neighbor_count)[:neighbor_count]
            for neighbor in nearest:
                other = int(local[int(neighbor)])
                edge_set.add((int(node_idx), other))
                edge_set.add((other, int(node_idx)))

    if not edge_set:
        order = sorted(
            range(len(nodes_df)),
            key=lambda i: (float(nodes_df.at[i, "time_step"]), nodes_df.at[i, "txId"]),
        )
        for s, d in zip(order, order[1:]):
            edge_set.add((s, d))
            edge_set.add((d, s))

    return _dedupe_edges(edge_set)


def build_graph_from_csv_bytes(
    nodes_csv: bytes,
    edges_csv: Optional[bytes],
    artifact_dir: Path,
    *,
    graph_mode: str = "auto",
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
) -> BuiltGraph:
    graph_mode = graph_mode.lower().strip()
    if graph_mode not in {"auto", "uploaded", "knn", "hybrid"}:
        raise PipelineInputError("graph_mode must be one of: auto, uploaded, knn, hybrid.")

    nodes_df = read_nodes_csv(nodes_csv)
    tx_ids = nodes_df["txId"].astype(str).tolist()
    tx_to_idx = {tx_id: idx for idx, tx_id in enumerate(tx_ids)}
    feat_norm, x, warnings = normalize_features(nodes_df, artifact_dir)

    src: List[int] = []
    dst: List[int] = []
    method = "none"
    uploaded_edges_df = read_edges_csv(edges_csv)

    if graph_mode in {"auto", "uploaded", "hybrid"}:
        src, dst, method = build_edges_from_upload(uploaded_edges_df, tx_to_idx, len(tx_ids), warnings)
        if graph_mode == "uploaded" and not src:
            raise PipelineInputError("No valid uploaded edges could be mapped to the uploaded nodes.")

    if graph_mode == "knn" or (graph_mode == "auto" and not src) or graph_mode == "hybrid":
        knn_src, knn_dst = build_knn_edges_same_timestep(
            nodes_df,
            feat_norm,
            k_neighbors=k_neighbors,
        )
        if graph_mode == "hybrid" and src:
            src, dst = _dedupe_edges(list(zip(src, dst)) + list(zip(knn_src, knn_dst)))
            method = f"{method}+knn_same_timestep"
        else:
            src, dst = knn_src, knn_dst
            method = "knn_same_timestep"
        if not src and len(tx_ids) > 1:
            raise PipelineInputError("Could not construct any graph edges from the uploaded nodes.")

    edge_index = (
        torch.tensor([src, dst], dtype=torch.long)
        if src
        else torch.zeros((2, 0), dtype=torch.long)
    )
    if edge_index.numel() > 0:
        min_idx = int(edge_index.min().item())
        max_idx = int(edge_index.max().item())
        if min_idx < 0 or max_idx >= len(tx_ids):
            raise PipelineInputError(
                f"edge_index contains invalid node ids: min={min_idx}, max={max_idx}, nodes={len(tx_ids)}."
            )

    debug = {
        "x_shape": list(x.shape),
        "edge_index_shape": list(edge_index.shape),
        "num_nodes": len(tx_ids),
        "num_edges": len(src),
        "feature_count": FEATURE_COUNT,
        "graph_method": method,
        "warnings": warnings,
    }
    LOGGER.info(
        "Built graph: x.shape=%s edge_index.shape=%s nodes=%s edges=%s method=%s",
        tuple(x.shape),
        tuple(edge_index.shape),
        len(tx_ids),
        len(src),
        method,
    )
    return BuiltGraph(nodes_df, tx_ids, tx_to_idx, x, edge_index, src, dst, method, warnings, debug)


def build_pyg_data(graph: BuiltGraph) -> Data:
    if graph.x.shape[1] != FEATURE_COUNT:
        raise PipelineInputError(f"x.shape[1]={graph.x.shape[1]}, expected {FEATURE_COUNT}.")
    return Data(x=graph.x, edge_index=graph.edge_index)


def run_inference(model: torch.nn.Module, graph: BuiltGraph, *, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    data = build_pyg_data(graph)
    model.eval()
    with torch.no_grad():
        logits = model(data)
        if logits.ndim != 2 or logits.shape[0] != graph.num_nodes or logits.shape[1] != 2:
            raise PipelineInputError(
                f"Model output shape {tuple(logits.shape)} is invalid; expected ({graph.num_nodes}, 2)."
            )
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    preds = (probs >= float(threshold)).astype(np.int64)
    return probs, preds


def make_predictions(graph: BuiltGraph, probs: np.ndarray, preds: np.ndarray) -> List[Dict[str, Any]]:
    predictions = []
    for idx, tx_id in enumerate(graph.tx_ids):
        probability = float(probs[idx])
        predictions.append(
            {
                "node_index": idx,
                "txId": tx_id,
                "fraud_probability": round(probability, 6),
                "prediction": "illicit" if int(preds[idx]) == 1 else "licit",
                "is_illicit": bool(preds[idx] == 1),
                "time_step": int(graph.nodes_df.at[idx, "time_step"])
                if np.isfinite(graph.nodes_df.at[idx, "time_step"])
                else None,
            }
        )
    return predictions


def _ring_record(
    ring_id: int,
    method: str,
    nodes: Sequence[int],
    graph_nx: nx.Graph,
    graph: BuiltGraph,
    probs: np.ndarray,
    preds: np.ndarray,
) -> Dict[str, Any]:
    ordered_nodes = sorted(int(n) for n in nodes)
    subgraph = graph_nx.subgraph(ordered_nodes)
    pred_illicit = int(sum(int(preds[n]) for n in ordered_nodes))
    avg_prob = float(np.mean([probs[n] for n in ordered_nodes])) if ordered_nodes else 0.0
    max_prob = float(np.max([probs[n] for n in ordered_nodes])) if ordered_nodes else 0.0
    return {
        "ring_id": ring_id,
        "method": method,
        "size": len(ordered_nodes),
        "edge_count": int(subgraph.number_of_edges()),
        "pred_illicit": pred_illicit,
        "illicit_ratio": round(pred_illicit / max(1, len(ordered_nodes)), 4),
        "avg_fraud_probability": round(avg_prob, 6),
        "max_fraud_probability": round(max_prob, 6),
        "node_indices": ordered_nodes,
        "txIds": [graph.tx_ids[n] for n in ordered_nodes],
    }


def detect_rings(
    graph: BuiltGraph,
    probs: np.ndarray,
    preds: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_size: int = 3,
    community_illicit_ratio: float = 0.4,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    graph_nx = nx.Graph()
    graph_nx.add_nodes_from(range(graph.num_nodes))
    graph_nx.add_edges_from(zip(graph.src, graph.dst))

    flagged_nodes = [idx for idx, prob in enumerate(probs) if float(prob) >= float(threshold)]
    flagged_graph = graph_nx.subgraph(flagged_nodes).copy()

    rings: List[Dict[str, Any]] = []
    for comp in nx.connected_components(flagged_graph):
        if len(comp) >= min_size:
            rings.append(_ring_record(len(rings), "fraud_connected_component", list(comp), graph_nx, graph, probs, preds))
    rings.sort(key=lambda r: (r["avg_fraud_probability"], r["size"]), reverse=True)
    for idx, ring in enumerate(rings):
        ring["ring_id"] = idx

    community_rings: List[Dict[str, Any]] = []
    if graph.num_edges > 0 and graph.num_nodes >= min_size:
        try:
            from networkx.algorithms.community import label_propagation_communities

            for community in label_propagation_communities(graph_nx):
                nodes = list(community)
                if len(nodes) < min_size:
                    continue
                illicit_ratio = float(np.mean([preds[n] for n in nodes]))
                avg_prob = float(np.mean([probs[n] for n in nodes]))
                if illicit_ratio >= community_illicit_ratio and avg_prob >= threshold * 0.6:
                    community_rings.append(
                        _ring_record(
                            len(community_rings),
                            "community_detection",
                            nodes,
                            graph_nx,
                            graph,
                            probs,
                            preds,
                        )
                    )
        except Exception as exc:  # Community detection should never break inference.
            LOGGER.warning("Community detection skipped: %s", exc)

    community_rings.sort(key=lambda r: (r["illicit_ratio"], r["avg_fraud_probability"]), reverse=True)
    for idx, ring in enumerate(community_rings):
        ring["ring_id"] = idx
    return rings, community_rings


def build_graph_preview(
    graph: BuiltGraph,
    probs: np.ndarray,
    rings: Sequence[Dict[str, Any]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    max_nodes: int = 600,
    max_edges: int = 1400,
) -> Dict[str, Any]:
    ring_by_node: Dict[int, int] = {}
    for ring in rings:
        for node in ring["node_indices"]:
            ring_by_node[int(node)] = int(ring["ring_id"])

    selected = set(ring_by_node.keys())
    for node in np.argsort(-probs).tolist():
        if len(selected) >= max_nodes:
            break
        selected.add(int(node))
    if not selected:
        selected = set(range(min(graph.num_nodes, max_nodes)))

    edge_items = []
    for s, d in zip(graph.src, graph.dst):
        if s in selected and d in selected:
            edge_items.append({"source": int(s), "target": int(d)})
        if len(edge_items) >= max_edges:
            break

    node_items = []
    for node in sorted(selected):
        node_items.append(
            {
                "id": int(node),
                "txId": graph.tx_ids[node],
                "probability": round(float(probs[node]), 6),
                "ring_id": ring_by_node.get(node),
                "is_illicit": bool(node in ring_by_node or probs[node] >= float(threshold)),
                "time_step": int(graph.nodes_df.at[node, "time_step"])
                if np.isfinite(graph.nodes_df.at[node, "time_step"])
                else None,
            }
        )
    return {"nodes": node_items, "edges": edge_items}
