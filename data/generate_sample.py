from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path(r"C:\Users\tumma\Downloads\fraud_detection_api_v2\fraud_detection_api")
DOWNLOADS = Path(r"C:\Users\tumma\Downloads")
FEATURE_ZIP = DOWNLOADS / "elliptic_txs_features.csv.zip"
EDGE_ZIP = DOWNLOADS / "elliptic_txs_edgelist.csv.zip"
CLASS_ZIP = DOWNLOADS / "elliptic_txs_classes.csv.zip"


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def _standardize_source_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    nodes = nodes.copy()
    if "txId" not in nodes.columns:
        nodes = pd.read_csv(DEFAULT_SOURCE / "real_sample_nodes.csv", header=None)
        cols = ["txId", "time_step"] + [f"f{i}" for i in range(165)]
        if nodes.shape[1] == 168:
            cols.append("class")
        nodes.columns = cols
        return nodes

    feature_cols = [col for col in nodes.columns if col not in {"txId", "class", "label"}]
    rename = {feature_cols[0]: "time_step"}
    rename.update({feature_cols[i + 1]: f"f{i}" for i in range(165)})
    return nodes.rename(columns=rename)


def _generate_from_full_elliptic(max_nodes: int) -> bool:
    if not (FEATURE_ZIP.exists() and EDGE_ZIP.exists() and CLASS_ZIP.exists()):
        return False

    classes = pd.read_csv(CLASS_ZIP, compression="zip")
    edges = pd.read_csv(EDGE_ZIP, compression="zip")
    illicit_ids = set(classes.loc[classes["class"].astype(str) == "1", "txId"].astype(str))

    illicit_edges = edges[
        edges["txId1"].astype(str).isin(illicit_ids)
        & edges["txId2"].astype(str).isin(illicit_ids)
    ].copy()

    uf = UnionFind()
    for left, right in zip(illicit_edges["txId1"].astype(str), illicit_edges["txId2"].astype(str)):
        uf.union(left, right)
    for tx_id in illicit_ids:
        uf.find(tx_id)

    components = {}
    for tx_id in illicit_ids:
        components.setdefault(uf.find(tx_id), []).append(tx_id)
    ranked = sorted(components.values(), key=len, reverse=True)

    selected_ids = []
    for component in ranked:
        if len(component) < 5:
            continue
        selected_ids.extend(component)
        if len(selected_ids) >= max_nodes:
            break
    selected_ids = selected_ids[:max_nodes]
    selected_set = set(selected_ids)

    sample_edges = edges[
        edges["txId1"].astype(str).isin(selected_set)
        & edges["txId2"].astype(str).isin(selected_set)
    ].copy()

    rows = []
    for chunk in pd.read_csv(FEATURE_ZIP, compression="zip", header=None, chunksize=50000):
        chunk_ids = chunk.iloc[:, 0].astype(str)
        picked = chunk.loc[chunk_ids.isin(selected_set)].copy()
        if not picked.empty:
            rows.append(picked)
    if not rows:
        return False

    nodes = pd.concat(rows, ignore_index=True)
    columns = ["txId", "time_step"] + [f"f{i}" for i in range(165)]
    nodes.columns = columns
    nodes["txId"] = nodes["txId"].astype(str)
    order = {tx_id: idx for idx, tx_id in enumerate(selected_ids)}
    nodes["_order"] = nodes["txId"].map(order)
    nodes = nodes.sort_values("_order").drop(columns=["_order"])

    nodes.to_csv(HERE / "sample_nodes.csv", index=False)
    sample_edges.to_csv(HERE / "sample_edges.csv", index=False)

    print("Generated from full Elliptic illicit connected components")
    print(f"sample_nodes.csv: {len(nodes)} rows")
    print(f"sample_edges.csv: {len(sample_edges)} rows")
    return True


def generate(max_nodes: int = 600) -> None:
    if _generate_from_full_elliptic(max_nodes):
        return

    source_nodes = DEFAULT_SOURCE / "real_sample_nodes.csv"
    source_edges = DEFAULT_SOURCE / "real_sample_edges.csv"
    if not source_nodes.exists() or not source_edges.exists():
        raise FileNotFoundError(
            "Could not find real_sample_nodes.csv and real_sample_edges.csv in the existing project."
        )

    nodes = _standardize_source_nodes(pd.read_csv(source_nodes))
    edges = pd.read_csv(source_edges)
    edges.columns = ["txId1", "txId2"]

    ids_in_edges = pd.unique(edges[["txId1", "txId2"]].values.ravel("K"))
    selected_ids = set(str(int(tx)) for tx in ids_in_edges[:max_nodes])

    if "class" in nodes.columns:
        illicit = nodes.loc[nodes["class"].astype(str).isin(["1", "illicit"]), "txId"].astype(str).head(80)
        selected_ids.update(illicit.tolist())

    selected = nodes.loc[nodes["txId"].astype(str).isin(selected_ids)].copy()
    if len(selected) > max_nodes:
        selected = selected.head(max_nodes)
    selected_ids = set(selected["txId"].astype(str))

    sample_edges = edges[
        edges["txId1"].astype(str).isin(selected_ids) & edges["txId2"].astype(str).isin(selected_ids)
    ].copy()

    required_cols = ["txId", "time_step"] + [f"f{i}" for i in range(165)]
    selected.loc[:, required_cols].to_csv(HERE / "sample_nodes.csv", index=False)
    sample_edges.to_csv(HERE / "sample_edges.csv", index=False)

    print(f"sample_nodes.csv: {len(selected)} rows")
    print(f"sample_edges.csv: {len(sample_edges)} rows")


if __name__ == "__main__":
    generate()
