# %% [markdown]
# # Bitcoin Fraud Ring Detection with GraphSAGE
#
# End-to-end Colab cells:
#
# 1. Upload Elliptic CSV zip files.
# 2. Preprocess features exactly once and save normalization artifacts.
# 3. Train GraphSAGE without changing the deployment architecture.
# 4. Test and tune threshold.
# 5. Detect fraud rings.
# 6. Save all outputs, including one draggable HTML for all rings and one draggable HTML per ring.

# %%
!pip -q install torch-geometric networkx scikit-learn pandas numpy matplotlib

# %%
import json
import os
import shutil
import warnings
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from google.colab import files
from networkx.algorithms.community import label_propagation_communities
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

OUT_DIR = Path("outputs")
RING_DIR = OUT_DIR / "rings"
ARTIFACT_DIR = Path("artifacts")
OUT_DIR.mkdir(exist_ok=True)
RING_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

# %% [markdown]
# ## Upload data
#
# Upload:
#
# - `elliptic_txs_features.csv.zip`
# - `elliptic_txs_edgelist.csv.zip`
# - `elliptic_txs_classes.csv.zip`

# %%
uploaded = files.upload()
for name in uploaded:
    print("uploaded:", name)

# %%
FEATURE_ZIP = Path("elliptic_txs_features.csv.zip")
EDGE_ZIP = Path("elliptic_txs_edgelist.csv.zip")
CLASS_ZIP = Path("elliptic_txs_classes.csv.zip")

assert FEATURE_ZIP.exists(), "Upload elliptic_txs_features.csv.zip"
assert EDGE_ZIP.exists(), "Upload elliptic_txs_edgelist.csv.zip"
assert CLASS_ZIP.exists(), "Upload elliptic_txs_classes.csv.zip"

raw_features = pd.read_csv(FEATURE_ZIP, compression="zip", header=None)
raw_edges = pd.read_csv(EDGE_ZIP, compression="zip")
raw_classes = pd.read_csv(CLASS_ZIP, compression="zip")

print("features:", raw_features.shape)
print("edges:", raw_edges.shape)
print("classes:", raw_classes.shape)
display(raw_features.head(2))
display(raw_edges.head(2))
display(raw_classes.head(2))

# %% [markdown]
# ## Feature preprocessing
#
# Important:
#
# - `txId` is used only for mapping.
# - Model input is exactly 166 columns: `time_step + 165 features`.
# - We save `col_min.npy`, `col_rng.npy`, and `col_median.npy` for deployment.

# %%
tx_ids = raw_features.iloc[:, 0].astype(np.int64).values
id_map = {int(tx): idx for idx, tx in enumerate(tx_ids)}
time_steps = raw_features.iloc[:, 1].astype(np.int64).values

feat_raw = raw_features.iloc[:, 1:].values.astype(np.float32)
col_median = np.nanmedian(feat_raw, axis=0).astype(np.float32)
nan_rows, nan_cols = np.where(np.isnan(feat_raw))
feat_raw[nan_rows, nan_cols] = col_median[nan_cols]

col_min = feat_raw.min(axis=0).astype(np.float32)
col_rng = (feat_raw.max(axis=0) - col_min).astype(np.float32)
col_rng[col_rng == 0] = 1.0
feat_norm = ((feat_raw - col_min) / col_rng).astype(np.float32)

np.save(ARTIFACT_DIR / "col_min.npy", col_min)
np.save(ARTIFACT_DIR / "col_rng.npy", col_rng)
np.save(ARTIFACT_DIR / "col_median.npy", col_median)

x = torch.tensor(feat_norm, dtype=torch.float32)
assert x.shape[1] == 166
assert not torch.isnan(x).any()

print("x.shape:", tuple(x.shape))
print("saved normalizer artifacts:", sorted(p.name for p in ARTIFACT_DIR.glob("*.npy")))

# %% [markdown]
# ## Build original Elliptic graph

# %%
edge_df = raw_edges.copy()
edge_df.columns = ["txId1", "txId2"]
edge_df["src"] = edge_df["txId1"].astype(np.int64).map(id_map)
edge_df["dst"] = edge_df["txId2"].astype(np.int64).map(id_map)
edge_df = edge_df.dropna(subset=["src", "dst"])

src_idx = edge_df["src"].astype(np.int64).values
dst_idx = edge_df["dst"].astype(np.int64).values
edge_index = torch.tensor(np.stack([src_idx, dst_idx]), dtype=torch.long)

assert edge_index.shape[0] == 2
assert edge_index.max().item() < x.shape[0]

print("edge_index.shape:", tuple(edge_index.shape))

# %% [markdown]
# ## Labels and train/test split
#
# Elliptic labels:
#
# - `1` means illicit.
# - `2` means licit.
# - `unknown` is ignored during supervised training/testing.

# %%
labels_df = raw_classes.copy()
labels_df.columns = ["txId", "class"]
labels_df["node_idx"] = labels_df["txId"].astype(np.int64).map(id_map)
labels_df = labels_df.dropna(subset=["node_idx"]).copy()
labels_df["node_idx"] = labels_df["node_idx"].astype(np.int64)
labels_df["y"] = labels_df["class"].astype(str).map({"unknown": -1, "2": 0, "1": 1})

y_arr = np.full(x.shape[0], -1, dtype=np.int64)
y_arr[labels_df["node_idx"].values] = labels_df["y"].values
y = torch.tensor(y_arr, dtype=torch.long)

train_mask = torch.tensor((y_arr >= 0) & (time_steps <= 34), dtype=torch.bool)
test_mask = torch.tensor((y_arr >= 0) & (time_steps >= 35), dtype=torch.bool)

data = Data(x=x, edge_index=edge_index, y=y)
print(data)
print("train nodes:", int(train_mask.sum()))
print("test nodes:", int(test_mask.sum()))
print("licit:", int((y == 0).sum()), "illicit:", int((y == 1).sum()), "unknown:", int((y == -1).sum()))

# %% [markdown]
# ## GraphSAGE model

# %%
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.c1 = SAGEConv(in_channels, 256)
        self.c2 = SAGEConv(256, 128)
        self.c3 = SAGEConv(128, 64)
        self.c4 = SAGEConv(64, 2)
        self.b1 = torch.nn.BatchNorm1d(256)
        self.b2 = torch.nn.BatchNorm1d(128)
        self.b3 = torch.nn.BatchNorm1d(64)
        self.drop = torch.nn.Dropout(0.4)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.b1(self.c1(x, edge_index)))
        x = self.drop(x)
        x = F.relu(self.b2(self.c2(x, edge_index)))
        x = self.drop(x)
        x = F.relu(self.b3(self.c3(x, edge_index)))
        x = self.drop(x)
        return self.c4(x, edge_index)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=y_arr[train_mask.cpu().numpy()],
)
alpha = torch.tensor(weights, dtype=torch.float32, device=device)
criterion = FocalLoss(gamma=2.0, alpha=alpha)

model = GraphSAGE(data.num_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

print("device:", device)
print("class weights:", weights)
print("parameters:", sum(p.numel() for p in model.parameters()))

# %% [markdown]
# ## Train

# %%
best_pr_auc = 0.0
best_epoch = 0
history = []

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    logits = model(data)
    loss = criterion(logits[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            probs = F.softmax(model(data), dim=1)[:, 1]
            test_probs = probs[test_mask].detach().cpu().numpy()
            test_y = data.y[test_mask].detach().cpu().numpy()
            pr_auc = average_precision_score(test_y, test_probs)
            roc_auc = roc_auc_score(test_y, test_probs)
        history.append({"epoch": epoch, "loss": float(loss.item()), "pr_auc": pr_auc, "roc_auc": roc_auc})
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_epoch = epoch
            torch.save(model.state_dict(), ARTIFACT_DIR / "graphsage_model.pt")
        print(f"epoch {epoch:03d} | loss {loss.item():.4f} | PR-AUC {pr_auc:.4f} | ROC-AUC {roc_auc:.4f}")

model.load_state_dict(torch.load(ARTIFACT_DIR / "graphsage_model.pt", map_location=device))
pd.DataFrame(history).to_csv(OUT_DIR / "training_history.csv", index=False)
print("best epoch:", best_epoch, "best PR-AUC:", round(best_pr_auc, 4))

# %% [markdown]
# ## Evaluate and tune threshold

# %%
model.eval()
with torch.no_grad():
    all_logits = model(data)
    all_probs = F.softmax(all_logits, dim=1)[:, 1].detach().cpu().numpy()

test_y = data.y[test_mask].detach().cpu().numpy()
test_probs = all_probs[test_mask.detach().cpu().numpy()]

precision, recall, thresholds = precision_recall_curve(test_y, test_probs)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
best_idx = int(np.argmax(f1))
THRESHOLD = float(thresholds[min(best_idx, len(thresholds) - 1)])
test_pred = (test_probs >= THRESHOLD).astype(int)

print("optimal threshold:", round(THRESHOLD, 4))
print(classification_report(test_y, test_pred, target_names=["licit", "illicit"], digits=4))
print("ROC-AUC:", round(roc_auc_score(test_y, test_probs), 4))
print("PR-AUC:", round(average_precision_score(test_y, test_probs), 4))

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, color="#2563eb", linewidth=2)
plt.scatter([recall[best_idx]], [precision[best_idx]], color="#dc2626", s=90)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_DIR / "precision_recall_curve.png", dpi=160)
plt.show()

# %% [markdown]
# ## Save node predictions

# %%
preds = (all_probs >= THRESHOLD).astype(np.int64)
node_predictions = pd.DataFrame({
    "node_index": np.arange(len(tx_ids)),
    "txId": tx_ids,
    "time_step": time_steps,
    "true_label": y_arr,
    "fraud_probability": all_probs,
    "prediction": preds,
})
node_predictions.to_csv(OUT_DIR / "node_predictions.csv", index=False)
display(node_predictions.sort_values("fraud_probability", ascending=False).head(10))

# %% [markdown]
# ## Detect fraud rings
#
# Ring logic:
#
# - First create a graph from the original transaction edges.
# - Keep predicted-illicit nodes.
# - Connected components of those nodes are direct fraud rings.
# - Also run label propagation communities and keep suspicious communities.

# %%
G = nx.Graph()
G.add_nodes_from(range(len(tx_ids)))
G.add_edges_from(zip(src_idx.tolist(), dst_idx.tolist()))

flagged_nodes = np.where(preds == 1)[0].tolist()
flagged_graph = G.subgraph(flagged_nodes).copy()

fraud_rings = []
for comp in nx.connected_components(flagged_graph):
    if len(comp) < 3:
        continue
    nodes = sorted(int(n) for n in comp)
    sub = G.subgraph(nodes)
    fraud_rings.append({
        "ring_id": len(fraud_rings),
        "method": "predicted_illicit_connected_component",
        "size": len(nodes),
        "edge_count": int(sub.number_of_edges()),
        "avg_fraud_probability": float(np.mean(all_probs[nodes])),
        "max_fraud_probability": float(np.max(all_probs[nodes])),
        "node_indices": nodes,
        "txIds": [int(tx_ids[n]) for n in nodes],
    })
fraud_rings = sorted(fraud_rings, key=lambda r: (r["avg_fraud_probability"], r["size"]), reverse=True)
for i, ring in enumerate(fraud_rings):
    ring["ring_id"] = i

community_rings = []
for comm in label_propagation_communities(G):
    nodes = sorted(int(n) for n in comm)
    if len(nodes) < 3:
        continue
    pred_ratio = float(np.mean(preds[nodes]))
    avg_prob = float(np.mean(all_probs[nodes]))
    if pred_ratio >= 0.4 and avg_prob >= THRESHOLD * 0.6:
        sub = G.subgraph(nodes)
        community_rings.append({
            "ring_id": len(community_rings),
            "method": "label_propagation_community",
            "size": len(nodes),
            "edge_count": int(sub.number_of_edges()),
            "pred_illicit_ratio": pred_ratio,
            "avg_fraud_probability": avg_prob,
            "max_fraud_probability": float(np.max(all_probs[nodes])),
            "node_indices": nodes,
            "txIds": [int(tx_ids[n]) for n in nodes],
        })
community_rings = sorted(community_rings, key=lambda r: (r["pred_illicit_ratio"], r["avg_fraud_probability"]), reverse=True)
for i, ring in enumerate(community_rings):
    ring["ring_id"] = i

print("fraud rings:", len(fraud_rings))
print("suspicious communities:", len(community_rings))

# %%
summary_rows = []
for ring in fraud_rings:
    summary_rows.append({
        "ring_id": ring["ring_id"],
        "method": ring["method"],
        "size": ring["size"],
        "edge_count": ring["edge_count"],
        "avg_fraud_probability": ring["avg_fraud_probability"],
        "max_fraud_probability": ring["max_fraud_probability"],
    })

pd.DataFrame(summary_rows).to_csv(OUT_DIR / "rings_summary.csv", index=False)
with open(OUT_DIR / "fraud_rings.json", "w", encoding="utf-8") as f:
    json.dump(fraud_rings, f, indent=2)
with open(OUT_DIR / "community_rings.json", "w", encoding="utf-8") as f:
    json.dump(community_rings, f, indent=2)

display(pd.DataFrame(summary_rows).head(20))

# %% [markdown]
# ## Draggable HTML visualizations
#
# These files work as standalone browser files. You can drag any node; connected nodes move through the force layout.

# %%
def make_visual_payload(rings, focus_ring_id=None, max_nodes=900, max_edges=2500):
    ring_by_node = {}
    for ring in rings:
        for node in ring["node_indices"]:
            ring_by_node[int(node)] = int(ring["ring_id"])

    if focus_ring_id is None:
        selected = set(ring_by_node.keys())
    else:
        ring = rings[int(focus_ring_id)]
        selected = set(map(int, ring["node_indices"]))
        for node in list(selected):
            selected.update(G.neighbors(node))

    if len(selected) > max_nodes:
        ranked = sorted(selected, key=lambda n: all_probs[n], reverse=True)
        selected = set(ranked[:max_nodes])

    vis_edges = []
    for s, d in zip(src_idx.tolist(), dst_idx.tolist()):
        if s in selected and d in selected:
            vis_edges.append({"source": int(s), "target": int(d)})
            if len(vis_edges) >= max_edges:
                break

    vis_nodes = []
    for node in sorted(selected):
        vis_nodes.append({
            "id": int(node),
            "txId": int(tx_ids[node]),
            "probability": float(all_probs[node]),
            "ring_id": ring_by_node.get(int(node)),
            "is_illicit": bool(preds[node] == 1),
            "time_step": int(time_steps[node]),
        })

    return {
        "nodes": vis_nodes,
        "edges": vis_edges,
        "rings": rings,
        "focus_ring_id": focus_ring_id,
    }


def write_force_graph_html(path, payload, title):
    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
html, body {{ margin: 0; height: 100%; background: #0b1017; color: #e8edf5; font-family: Inter, Segoe UI, Arial, sans-serif; }}
#top {{ height: 54px; display: flex; align-items: center; gap: 12px; padding: 0 16px; background: #111821; border-bottom: 1px solid #253142; }}
#badge {{ background: #dc2626; color: white; border-radius: 999px; padding: 4px 12px; font-weight: 800; font-size: 12px; }}
#title {{ font-weight: 800; font-size: 16px; }}
#meta {{ color: #9fb0c7; font-size: 13px; }}
#main {{ display: grid; grid-template-columns: 1fr 300px; height: calc(100% - 54px); }}
#graph {{ width: 100%; height: 100%; cursor: grab; }}
#graph.dragging {{ cursor: grabbing; }}
#side {{ border-left: 1px solid #253142; background: #111821; padding: 14px; overflow: auto; }}
.ring {{ padding: 9px 0; border-bottom: 1px solid #1f2a38; font-size: 12px; cursor: pointer; }}
.name {{ font-weight: 800; color: #f8fafc; }}
.dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 99px; margin-right: 8px; }}
#hover {{ margin-top: 16px; color: #d7e3f5; font-size: 12px; line-height: 1.45; }}
button {{ background: #172234; border: 1px solid #34445c; color: #dce8fa; border-radius: 6px; height: 30px; padding: 0 12px; cursor: pointer; }}
</style>
</head>
<body>
<div id="top">
  <span id="badge">FRAUD RINGS</span>
  <span id="title">{title}</span>
  <span id="meta"></span>
  <button id="all">All Rings</button>
  <button id="fit">Fit</button>
  <button id="pause">Pause</button>
</div>
<div id="main">
  <canvas id="graph"></canvas>
  <aside id="side"><div id="rings"></div><div id="hover"></div></aside>
</div>
<script>
const graph = {json.dumps(payload)};
const canvas = document.getElementById("graph"), ctx = canvas.getContext("2d");
const colors = ["#ef4444","#f97316","#eab308","#84cc16","#22c55e","#14b8a6","#06b6d4","#3b82f6","#8b5cf6","#d946ef","#ec4899","#f43f5e"];
const nodes = graph.nodes || [], edges = graph.edges || [], rings = graph.rings || [];
const byId = new Map(nodes.map(n => [n.id, n]));
const ringMap = new Map();
rings.forEach(r => (r.node_indices || []).forEach(n => ringMap.set(Number(n), r.ring_id)));
nodes.forEach(n => {{ if (ringMap.has(n.id)) n.ring_id = ringMap.get(n.id); n.vx = 0; n.vy = 0; }});
let focusRing = graph.focus_ring_id, scale = 1, ox = 0, oy = 0, paused = false, drag = null, pan = null, hover = null;
function resize() {{ const r = canvas.getBoundingClientRect(); canvas.width = r.width; canvas.height = r.height; }}
function sx(x) {{ return x * scale + ox; }} function sy(y) {{ return y * scale + oy; }}
function wx(x) {{ return (x - ox) / scale; }} function wy(y) {{ return (y - oy) / scale; }}
function keepSet() {{ if (focusRing === null || focusRing === undefined || focusRing === "all") return new Set(nodes.map(n => n.id));
  const ring = rings.find(r => r.ring_id === Number(focusRing)); return new Set((ring ? ring.node_indices : []).map(Number)); }}
function ns() {{ const k = keepSet(); return nodes.filter(n => k.has(n.id)); }}
function es() {{ const k = keepSet(); return edges.filter(e => k.has(e.source) && k.has(e.target)); }}
function layout() {{ resize(); const show = ns(); const cols = Math.ceil(Math.sqrt(show.length || 1));
  show.forEach((n, i) => {{ const a = Math.PI * 2 * i / Math.max(1, show.length); const r = 80 + 5 * Math.sqrt(show.length);
    n.x = canvas.width / 2 + r * Math.cos(a) + (i % cols) * 4; n.y = canvas.height / 2 + r * Math.sin(a) + Math.floor(i / cols) * 4; }});
  fit(); }}
function fit() {{ const show = ns(); if (!show.length) return; const minX = Math.min(...show.map(n => n.x)), maxX = Math.max(...show.map(n => n.x));
  const minY = Math.min(...show.map(n => n.y)), maxY = Math.max(...show.map(n => n.y)); const pad = 80;
  scale = Math.min(canvas.width / Math.max(1, maxX - minX + pad * 2), canvas.height / Math.max(1, maxY - minY + pad * 2), 2);
  ox = canvas.width / 2 - scale * ((minX + maxX) / 2); oy = canvas.height / 2 - scale * ((minY + maxY) / 2); }}
function tick() {{ if (paused || drag) return; const show = ns(), links = es(); const cx = wx(canvas.width * .47), cy = wy(canvas.height * .52);
  show.forEach(n => {{ n.vx += (cx - n.x) * .001; n.vy += (cy - n.y) * .001; }});
  for (let i=0;i<show.length;i++) for (let j=i+1;j<show.length;j++) {{ const a=show[i], b=show[j], dx=a.x-b.x, dy=a.y-b.y, d2=Math.max(80, dx*dx+dy*dy), f=140/d2; a.vx+=dx*f; a.vy+=dy*f; b.vx-=dx*f; b.vy-=dy*f; }}
  links.forEach(e => {{ const s=byId.get(e.source), d=byId.get(e.target); if(!s||!d) return; const dx=d.x-s.x, dy=d.y-s.y, dist=Math.max(1, Math.hypot(dx,dy)), f=(dist-45)*.015; s.vx+=dx/dist*f; s.vy+=dy/dist*f; d.vx-=dx/dist*f; d.vy-=dy/dist*f; }});
  show.forEach(n => {{ n.vx*=.86; n.vy*=.86; n.x+=n.vx; n.y+=n.vy; }}); }}
function draw() {{ ctx.fillStyle="#0b1017"; ctx.fillRect(0,0,canvas.width,canvas.height); es().forEach(e => {{ const s=byId.get(e.source), d=byId.get(e.target); if(!s||!d) return; ctx.strokeStyle=s.ring_id===d.ring_id?"rgba(250,204,21,.30)":"rgba(148,163,184,.16)"; ctx.beginPath(); ctx.moveTo(sx(s.x),sy(s.y)); ctx.lineTo(sx(d.x),sy(d.y)); ctx.stroke(); }});
  ns().forEach(n => {{ const c=n.ring_id!==null&&n.ring_id!==undefined?colors[n.ring_id%colors.length]:null; ctx.fillStyle=c||(n.is_illicit?"#facc15":"#38bdf8"); ctx.strokeStyle=hover&&hover.id===n.id?"#fff":(n.is_illicit?"#fde047":"#0b1017"); ctx.lineWidth=hover&&hover.id===n.id?3:1.5; ctx.beginPath(); ctx.arc(sx(n.x),sy(n.y),n.is_illicit?6:4,0,Math.PI*2); ctx.fill(); ctx.stroke(); }}); }}
function hit(mx,my) {{ let best=null, bd=999; ns().forEach(n => {{ const d=Math.hypot(sx(n.x)-mx, sy(n.y)-my); if(d<bd) {{ best=n; bd=d; }} }}); return bd<=16?best:null; }}
function renderRings() {{ document.getElementById("meta").textContent = `${{nodes.length}} nodes | ${{edges.length}} edges | ${{rings.length}} rings`; const box=document.getElementById("rings"); box.innerHTML=""; rings.slice(0,80).forEach(r => {{ const div=document.createElement("div"); div.className="ring"; div.innerHTML=`<span class="dot" style="background:${{colors[r.ring_id%colors.length]}}"></span><span class="name">Ring ${{r.ring_id}}</span><br>${{r.size}} nodes | avg p=${{Number(r.avg_fraud_probability).toFixed(3)}}`; div.onclick=()=>{{focusRing=r.ring_id; layout(); renderRings();}}; box.appendChild(div); }}); }}
canvas.onmousemove = ev => {{ const r=canvas.getBoundingClientRect(), mx=ev.clientX-r.left, my=ev.clientY-r.top; if(drag) {{ drag.x=wx(mx); drag.y=wy(my); drag.vx=0; drag.vy=0; return; }} if(pan) {{ ox=pan.ox+ev.clientX-pan.x; oy=pan.oy+ev.clientY-pan.y; return; }} hover=hit(mx,my); document.getElementById("hover").innerHTML=hover?`<b>txId ${{hover.txId}}</b><br>node ${{hover.id}}<br>p=${{Number(hover.probability).toFixed(4)}}`:""; }};
canvas.onmousedown = ev => {{ const r=canvas.getBoundingClientRect(); drag=hit(ev.clientX-r.left, ev.clientY-r.top); canvas.classList.add("dragging"); if(!drag) pan={{x:ev.clientX,y:ev.clientY,ox:ox,oy:oy}}; }};
window.onmouseup = () => {{ drag=null; pan=null; canvas.classList.remove("dragging"); }};
canvas.onwheel = ev => {{ ev.preventDefault(); const r=canvas.getBoundingClientRect(), mx=ev.clientX-r.left, my=ev.clientY-r.top, bx=wx(mx), by=wy(my); scale*=ev.deltaY<0?1.1:.9; scale=Math.max(.2,Math.min(5,scale)); ox=mx-bx*scale; oy=my-by*scale; }};
document.getElementById("all").onclick=()=>{{focusRing="all"; layout(); renderRings();}};
document.getElementById("fit").onclick=()=>{{fit();}};
document.getElementById("pause").onclick=ev=>{{paused=!paused; ev.target.textContent=paused?"Play":"Pause";}};
function frame() {{ tick(); draw(); requestAnimationFrame(frame); }}
layout(); renderRings(); frame();
</script>
</body>
</html>
"""
    Path(path).write_text(html, encoding="utf-8")


all_payload = make_visual_payload(fraud_rings, focus_ring_id=None)
write_force_graph_html(OUT_DIR / "fraud_rings_all_interactive.html", all_payload, f"All Fraud Rings - {len(fraud_rings)} rings")

for ring in fraud_rings:
    payload = make_visual_payload(fraud_rings, focus_ring_id=ring["ring_id"], max_nodes=500, max_edges=1200)
    write_force_graph_html(RING_DIR / f"ring_{ring['ring_id']:03d}.html", payload, f"Fraud Ring {ring['ring_id']}")

print("saved:", OUT_DIR / "fraud_rings_all_interactive.html")
print("saved per-ring HTML files:", len(list(RING_DIR.glob("ring_*.html"))))

# %% [markdown]
# ## Package and download outputs

# %%
shutil.make_archive("fraud_detection_outputs", "zip", OUT_DIR)
shutil.make_archive("graphsage_artifacts", "zip", ARTIFACT_DIR)

files.download("fraud_detection_outputs.zip")
files.download("graphsage_artifacts.zip")

print("Done.")
