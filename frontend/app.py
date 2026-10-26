from __future__ import annotations



import json
import time
import sys

from pathlib import Path

from typing import Optional



import pandas as pd


import streamlit as st

import streamlit.components.v1 as components



st.set_page_config(page_title="Bitcoin Fraud Detection & Graph Analytics", layout="wide")



BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

SAMPLE_NODES = DATA_DIR / "sample_nodes.csv"

SAMPLE_EDGES = DATA_DIR / "sample_edges.csv"
ARTIFACT_DIR = BASE_DIR / "validated_graphsage_export" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "graphsage_model.pt"

# Streamlit Cloud launches frontend/app.py: make project packages importable.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from core.pipeline import (
    FEATURE_COUNT, PipelineInputError, build_graph_from_csv_bytes,
    run_inference, detect_rings, build_graph_preview, make_predictions,
)
from model.graphsage import load_model




st.markdown(

    """

<style>

.stApp { background: #0f141b; color: #e8edf5; }

section[data-testid="stSidebar"] { background: #141a22; border-right: 1px solid #263142; }

div[data-testid="metric-container"] {

  background: #151d27; border: 1px solid #2a3647; border-radius: 8px; padding: 14px;

}

.block-container { padding-top: 1.3rem; }

.stDataFrame { border: 1px solid #2a3647; border-radius: 8px; }

</style>

""",

    unsafe_allow_html=True,

)





@st.cache_resource(show_spinner="Loading validated GraphSAGE model...")
def get_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {MODEL_PATH}")
    return load_model(MODEL_PATH, in_channels=FEATURE_COUNT)


def get_validated_threshold() -> float:
    path = ARTIFACT_DIR / "threshold.npy"
    return float(np.load(path).item()) if path.exists() else 0.802532


def read_sample_file(path: Path) -> Optional[bytes]:
    return path.read_bytes() if path.exists() else None


def submit_pipeline(
    nodes_bytes: bytes,
    edges_bytes: Optional[bytes],
    *,
    threshold: float,
    graph_mode: str,
    k_neighbors: int,
    min_ring_size: int,
) -> dict:
    model = get_model()
    started = time.perf_counter()
    graph = build_graph_from_csv_bytes(
        nodes_bytes, edges_bytes, ARTIFACT_DIR,
        graph_mode=graph_mode, k_neighbors=k_neighbors,
    )
    infer_started = time.perf_counter()
    probs, preds = run_inference(model, graph, threshold=threshold)
    inference_ms = round((time.perf_counter() - infer_started) * 1000, 2)
    component_rings, community_rings = detect_rings(
        graph, probs, preds, threshold=threshold, min_size=min_ring_size,
    )
    primary_rings = component_rings if component_rings else community_rings
    preview = build_graph_preview(graph, probs, primary_rings, threshold=threshold)
    flagged_nodes = {idx for ring in primary_rings for idx in ring["node_indices"]}
    return {
        "summary": {
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "x_shape": graph.debug["x_shape"],
            "edge_index_shape": graph.debug["edge_index_shape"],
            "graph_method": graph.graph_method,
            "threshold": float(threshold),
            "predicted_illicit": int(preds.sum()),
            "illicit_pct": round(float(preds.mean()) * 100, 2) if graph.num_nodes else 0.0,
            "total_rings": len(primary_rings),
            "total_flagged_ring_nodes": len(flagged_nodes),
            "inference_ms": inference_ms,
            "total_pipeline_ms": round((time.perf_counter() - started) * 1000, 2),
        },
        "debug": graph.debug,
        "predictions": make_predictions(graph, probs, preds),
        "rings": primary_rings,
        "component_rings": component_rings,
        "community_rings": community_rings,
        "graph": preview,
    }


def ring_graph_html(result: dict, focus_ring_id: Optional[int] = None) -> str:

    """Interactive force graph: drag nodes, pan blank space, scroll to zoom."""

    payload = {

        "nodes": result.get("graph", {}).get("nodes", []),

        "edges": result.get("graph", {}).get("edges", []),

        "rings": result.get("rings", []),

        "focus_ring_id": focus_ring_id,

    }

    template = """

<div id="wrap">

  <div id="topbar">

    <button id="allBtn" class="toolBtn">All Rings</button>

    <button id="fitBtn" class="toolBtn">Fit</button>

    <button id="resetBtn" class="toolBtn">Reset</button>

    <button id="pauseBtn" class="toolBtn">Pause</button>

  </div>

  <canvas id="graphCanvas"></canvas>

  <div id="side">

    <div class="sideTitle">Clusters (__RING_COUNT__)</div>

    <div id="ringList"></div>

    <div id="hoverBox"></div>

  </div>

</div>

<style>

  #wrap { height: 640px; width: 100%; display: grid; grid-template-columns: 1fr 280px;

          grid-template-rows: 44px 1fr; background: #0b1017; border: 1px solid #253142;

          border-radius: 8px; overflow: hidden; font-family: Inter, Segoe UI, Arial, sans-serif; }

  #topbar { grid-column: 1 / 2; display: flex; align-items: center; gap: 8px; padding: 7px 10px;

            border-bottom: 1px solid #253142; background: #101722; }

  .toolBtn { background: #172234; border: 1px solid #34445c; color: #dce8fa; border-radius: 6px;

             height: 28px; padding: 0 11px; font-size: 12px; cursor: pointer; }

  .toolBtn:hover { background: #20314b; border-color: #4f76b9; }

  #graphCanvas { grid-column: 1 / 2; width: 100%; height: 596px; display: block; cursor: grab; }

  #graphCanvas.dragging { cursor: grabbing; }

  #side { grid-column: 2 / 3; grid-row: 1 / 3; border-left: 1px solid #253142; background: #111821;

          color: #e8edf5; padding: 14px; overflow: auto; }

  .sideTitle { color: #9fb0c7; font-weight: 700; font-size: 13px; margin-bottom: 12px; }

  .ringRow { padding: 8px 0; border-bottom: 1px solid #1f2a38; font-size: 12px; line-height: 1.35; cursor: pointer; }

  .ringRow:hover { background: rgba(59,130,246,.08); }

  .ringRow.active { background: rgba(250,204,21,.10); }

  .ringName { font-weight: 700; color: #f8fafc; }

  .dot { display: inline-block; width: 10px; height: 10px; border-radius: 99px; margin-right: 7px; }

  #hoverBox { margin-top: 14px; color: #d7e3f5; font-size: 12px; line-height: 1.45; min-height: 56px; }

</style>

<script>

const graph = __GRAPH_JSON__;

const canvas = document.getElementById("graphCanvas");

const ctx = canvas.getContext("2d");

const side = document.getElementById("ringList");

const hover = document.getElementById("hoverBox");

const allBtn = document.getElementById("allBtn");

const fitBtn = document.getElementById("fitBtn");

const resetBtn = document.getElementById("resetBtn");

const pauseBtn = document.getElementById("pauseBtn");

const colors = ["#ef4444","#f97316","#eab308","#84cc16","#22c55e","#14b8a6","#06b6d4",

                "#3b82f6","#8b5cf6","#d946ef","#ec4899","#f43f5e"];



const nodes = graph.nodes || [];

const edges = graph.edges || [];

const rings = graph.rings || [];

const byId = new Map(nodes.map(n => [n.id, n]));

const ringMap = new Map();

rings.forEach(r => (r.node_indices || []).forEach(n => ringMap.set(Number(n), r.ring_id)));

nodes.forEach(n => { if (ringMap.has(n.id)) n.ring_id = ringMap.get(n.id); });

const adjacency = new Map();

edges.forEach(e => {

  if (!adjacency.has(e.source)) adjacency.set(e.source, new Set());

  if (!adjacency.has(e.target)) adjacency.set(e.target, new Set());

  adjacency.get(e.source).add(e.target);

  adjacency.get(e.target).add(e.source);

});



let focusRing = graph.focus_ring_id;

let paused = false, dragNode = null, hoverNode = null, panDrag = null;

let offsetX = 0, offsetY = 0, scale = 1;



function resize() {

  const rect = canvas.getBoundingClientRect();

  canvas.width = Math.floor(rect.width);

  canvas.height = Math.floor(rect.height);

}

function sx(x) { return x * scale + offsetX; }

function sy(y) { return y * scale + offsetY; }

function wx(x) { return (x - offsetX) / scale; }

function wy(y) { return (y - offsetY) / scale; }



function visibleNodeSet() {

  if (focusRing === null || focusRing === undefined || focusRing === "all") {

    return new Set(nodes.map(n => n.id));

  }

  const ring = rings.find(r => r.ring_id === Number(focusRing));

  const keep = new Set((ring && ring.node_indices ? ring.node_indices : []).map(Number));

  keep.forEach(id => (adjacency.get(id) || []).forEach(n => keep.add(n)));

  return keep;

}

function shownNodes() {

  const keep = visibleNodeSet();

  return nodes.filter(n => keep.has(n.id));

}

function shownEdges() {

  const keep = visibleNodeSet();

  return edges.filter(e => keep.has(e.source) && keep.has(e.target));

}



function layout() {

  resize();

  const w = canvas.width, h = canvas.height;

  const byRing = new Map(), loose = [];

  nodes.forEach(n => {

    n.vx = 0; n.vy = 0;

    if (n.ring_id !== null && n.ring_id !== undefined) {

      if (!byRing.has(n.ring_id)) byRing.set(n.ring_id, []);

      byRing.get(n.ring_id).push(n);

    } else {

      loose.push(n);

    }

  });

  const ringIds = Array.from(byRing.keys()).sort((a,b) => a-b);

  const cx = w * 0.46, cy = h * 0.52;

  const rx = Math.max(130, w * 0.30), ry = Math.max(95, h * 0.32);

  ringIds.forEach((rid, i) => {

    const members = byRing.get(rid);

    const theta = Math.PI * 2 * i / Math.max(1, ringIds.length) - Math.PI / 2;

    const groupX = cx + rx * Math.cos(theta);

    const groupY = cy + ry * Math.sin(theta);

    const r = Math.max(20, Math.min(62, 10 + members.length * 1.35));

    members.forEach((n, j) => {

      const a = Math.PI * 2 * j / Math.max(1, members.length);

      n.x = groupX + r * Math.cos(a);

      n.y = groupY + r * Math.sin(a);

    });

  });

  loose.sort((a,b) => b.probability - a.probability);

  loose.forEach((n, i) => {

    n.x = w * 0.08 + (i % 5) * 38;

    n.y = 48 + (Math.floor(i / 5) % 13) * 38;

  });

  fitView();

}



function fitView() {

  const show = shownNodes();

  if (!show.length) return;

  const minX = Math.min(...show.map(n => n.x)), maxX = Math.max(...show.map(n => n.x));

  const minY = Math.min(...show.map(n => n.y)), maxY = Math.max(...show.map(n => n.y));

  const pad = 80, bw = Math.max(1, maxX - minX + pad * 2), bh = Math.max(1, maxY - minY + pad * 2);

  scale = Math.min(canvas.width / bw, canvas.height / bh, 1.8);

  offsetX = canvas.width / 2 - scale * ((minX + maxX) / 2);

  offsetY = canvas.height / 2 - scale * ((minY + maxY) / 2);

}



function tick() {

  if (paused || dragNode) return;

  const ns = shownNodes(), es = shownEdges();

  const centerX = wx(canvas.width * 0.48), centerY = wy(canvas.height * 0.50);

  ns.forEach(n => {

    n.vx += (centerX - n.x) * 0.0009;

    n.vy += (centerY - n.y) * 0.0009;

  });

  for (let i = 0; i < ns.length; i++) {

    for (let j = i + 1; j < ns.length; j++) {

      const a = ns[i], b = ns[j];

      const dx = a.x - b.x, dy = a.y - b.y;

      const d2 = Math.max(70, dx * dx + dy * dy);

      const f = (a.ring_id === b.ring_id ? 95 : 155) / d2;

      a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;

    }

  }

  es.forEach(e => {

    const s = byId.get(e.source), d = byId.get(e.target);

    if (!s || !d) return;

    const dx = d.x - s.x, dy = d.y - s.y, dist = Math.max(1, Math.sqrt(dx * dx + dy * dy));

    const target = s.ring_id === d.ring_id ? 34 : 62;

    const f = (dist - target) * 0.012;

    const fx = dx / dist * f, fy = dy / dist * f;

    s.vx += fx; s.vy += fy; d.vx -= fx; d.vy -= fy;

  });

  ns.forEach(n => {

    n.vx = Math.max(-8, Math.min(8, n.vx * 0.86));

    n.vy = Math.max(-8, Math.min(8, n.vy * 0.86));

    n.x += n.vx; n.y += n.vy;

  });

}



function draw() {

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = "#0b1017"; ctx.fillRect(0, 0, canvas.width, canvas.height);

  shownEdges().forEach(e => {

    const s = byId.get(e.source), d = byId.get(e.target);

    if (!s || !d) return;

    ctx.strokeStyle = s.ring_id !== undefined && s.ring_id === d.ring_id ? "rgba(250,204,21,.28)" : "rgba(148,163,184,.14)";

    ctx.lineWidth = 1;

    ctx.beginPath(); ctx.moveTo(sx(s.x), sy(s.y)); ctx.lineTo(sx(d.x), sy(d.y)); ctx.stroke();

  });

  shownNodes().forEach(n => {

    const c = n.ring_id !== null && n.ring_id !== undefined ? colors[n.ring_id % colors.length] : null;

    const radius = n.is_illicit ? 6 : 4;

    ctx.fillStyle = c || (n.is_illicit ? "#facc15" : "#38bdf8");

    ctx.strokeStyle = hoverNode && hoverNode.id === n.id ? "#ffffff" : (n.is_illicit ? "#fde047" : "#0b1017");

    ctx.lineWidth = hoverNode && hoverNode.id === n.id ? 3 : (n.is_illicit ? 2 : 1);

    ctx.beginPath(); ctx.arc(sx(n.x), sy(n.y), radius, 0, Math.PI * 2); ctx.fill(); ctx.stroke();

  });

}



function renderList() {

  side.innerHTML = "";

  rings.slice(0, 50).forEach(r => {

    const div = document.createElement("div");

    div.className = "ringRow";

    if (Number(focusRing) === r.ring_id) div.classList.add("active");

    const c = colors[r.ring_id % colors.length];

    div.innerHTML = `<span class="dot" style="background:${c}"></span><span class="ringName">Ring ${r.ring_id}</span><br>` +

      `${r.size} nodes | avg p=${Number(r.avg_fraud_probability).toFixed(3)}`;

    div.addEventListener("click", () => { focusRing = r.ring_id; fitView(); renderList(); draw(); });

    side.appendChild(div);

  });

}



function hitNode(mx, my) {

  let best = null, bestDist = 9999;

  shownNodes().forEach(n => {

    const dx = sx(n.x) - mx, dy = sy(n.y) - my;

    const d = Math.sqrt(dx * dx + dy * dy);

    if (d < bestDist) { best = n; bestDist = d; }

  });

  return bestDist <= 16 ? best : null;

}



canvas.addEventListener("mousemove", ev => {

  const rect = canvas.getBoundingClientRect();

  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;

  if (dragNode) {

    dragNode.x = wx(mx); dragNode.y = wy(my); dragNode.vx = 0; dragNode.vy = 0; draw(); return;

  }

  if (panDrag) {

    offsetX = panDrag.ox + ev.clientX - panDrag.x;

    offsetY = panDrag.oy + ev.clientY - panDrag.y;

    draw(); return;

  }

  hoverNode = hitNode(mx, my);

  if (!hoverNode) { hover.textContent = ""; draw(); return; }

  hover.innerHTML = `<b>txId ${hoverNode.txId}</b><br>node ${hoverNode.id}<br>p=${Number(hoverNode.probability).toFixed(4)}` +

    (hoverNode.ring_id !== null && hoverNode.ring_id !== undefined ? `<br>ring ${hoverNode.ring_id}` : "");

  draw();

});

canvas.addEventListener("mousedown", ev => {

  const rect = canvas.getBoundingClientRect();

  dragNode = hitNode(ev.clientX - rect.left, ev.clientY - rect.top);

  canvas.classList.add("dragging");

  if (!dragNode) panDrag = { x: ev.clientX, y: ev.clientY, ox: offsetX, oy: offsetY };

});

window.addEventListener("mouseup", () => { dragNode = null; panDrag = null; canvas.classList.remove("dragging"); });

canvas.addEventListener("wheel", ev => {

  ev.preventDefault();

  const rect = canvas.getBoundingClientRect();

  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;

  const bx = wx(mx), by = wy(my);

  scale *= ev.deltaY < 0 ? 1.1 : 0.9;

  scale = Math.max(0.25, Math.min(4, scale));

  offsetX = mx - bx * scale; offsetY = my - by * scale;

  draw();

}, { passive: false });

allBtn.addEventListener("click", () => { focusRing = "all"; fitView(); renderList(); draw(); });

fitBtn.addEventListener("click", () => { fitView(); draw(); });

resetBtn.addEventListener("click", () => { layout(); renderList(); draw(); });

pauseBtn.addEventListener("click", () => { paused = !paused; pauseBtn.textContent = paused ? "Play" : "Pause"; });

window.addEventListener("resize", () => { resize(); fitView(); draw(); });



function frame() { tick(); draw(); requestAnimationFrame(frame); }

layout(); renderList(); frame();

</script>

"""

    return (

        template.replace("__GRAPH_JSON__", json.dumps(payload))

        .replace("__RING_COUNT__", str(len(payload["rings"])))

    )





try:
    _model = get_model()
    model_ready = True
except Exception as exc:
    model_ready = False
    st.error(f"Model failed to load: {exc}")

with st.sidebar:
    st.markdown("### Model")
    if model_ready:
        st.success("Validated GraphSAGE loaded locally")
    else:
        st.error("Model unavailable")

    st.markdown("### Inference")

    threshold = st.number_input("Fraud classification threshold", min_value=0.0, max_value=1.0, value=get_validated_threshold(), step=0.001, format="%.6f")

    graph_mode = st.selectbox("Graph mode", ["uploaded", "auto", "knn", "hybrid"], index=0, help="Use actual transaction edges for the validated workflow. KNN/hybrid results were not independently evaluated.")

    k_neighbors = st.slider("KNN neighbors", 1, 15, 6)

    min_ring_size = st.slider("Minimum ring size", 2, 20, 3)



    st.markdown("### Data")

    sample_ready = SAMPLE_NODES.exists()

    use_sample = st.toggle("Use sample CSVs", value=sample_ready, disabled=not sample_ready)

    if sample_ready:

        st.download_button("Download sample_nodes.csv", SAMPLE_NODES.read_bytes(), "sample_nodes.csv")

    if SAMPLE_EDGES.exists():

        st.download_button("Download sample_edges.csv", SAMPLE_EDGES.read_bytes(), "sample_edges.csv")



st.title("Bitcoin Fraud Detection & Graph Analytics")
st.caption("GraphSAGE transaction-risk scoring using the Elliptic Bitcoin dataset.")
st.info("Graph clusters indicate suspected high-risk transaction groups, not confirmed coordinated fraud rings.")



nodes_bytes = None

edges_bytes = None

nodes_name = "sample_nodes.csv"

edges_name = "sample_edges.csv"



if use_sample:

    nodes_bytes = read_sample_file(SAMPLE_NODES)

    edges_bytes = read_sample_file(SAMPLE_EDGES)

    st.caption(f"Loaded sample files from {DATA_DIR}")

else:

    left, right = st.columns(2)

    with left:

        nodes_upload = st.file_uploader("Nodes CSV", type="csv")

    with right:

        edges_upload = st.file_uploader("Edges CSV", type="csv")

    if nodes_upload is not None:

        nodes_name = nodes_upload.name

        nodes_bytes = nodes_upload.getvalue()

    if edges_upload is not None:

        edges_name = edges_upload.name

        edges_bytes = edges_upload.getvalue()



run = st.button("Run analysis", type="primary", disabled=(not model_ready or not nodes_bytes or (graph_mode == "uploaded" and not edges_bytes)))



if run:

    try:

        with st.spinner("Running GraphSAGE inference"):

            result = submit_pipeline(
                nodes_bytes,
                edges_bytes,

                threshold=threshold,

                graph_mode=graph_mode,

                k_neighbors=k_neighbors,

                min_ring_size=min_ring_size,

            )

    except Exception as exc:

        st.error(str(exc))

        st.stop()



    summary = result["summary"]

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    m1.metric("Nodes", f"{summary['num_nodes']:,}")

    m2.metric("Edges", f"{summary['num_edges']:,}")

    m3.metric("Illicit", f"{summary['predicted_illicit']:,}", f"{summary['illicit_pct']}%")

    m4.metric("Clusters", summary["total_rings"])

    m5.metric("Inference", f"{summary['inference_ms']} ms")

    m6.metric("Graph", summary["graph_method"])



    st.markdown("#### Suspected High-Risk Clusters")

    ring_options = {"All rings": None}

    ring_options.update({f"Ring {r['ring_id']} ({r['size']} nodes)": r["ring_id"] for r in result.get("rings", [])})

    selected_ring_label = st.selectbox("Graph view", list(ring_options.keys()))

    components.html(ring_graph_html(result, focus_ring_id=ring_options[selected_ring_label]), height=660)



    tab_rings, tab_predictions, tab_debug = st.tabs(["Clusters", "Predictions", "Debug"])



    with tab_rings:

        rings = result.get("rings", [])

        if not rings:

            st.info("No suspected high-risk clusters detected at this threshold.")

        for ring in rings:

            with st.expander(

                f"Ring {ring['ring_id']} | {ring['size']} nodes | "

                f"avg p={ring['avg_fraud_probability']:.3f}"

            ):

                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Size", ring["size"])

                c2.metric("Edges", ring["edge_count"])

                c3.metric("Illicit ratio", f"{ring['illicit_ratio']:.0%}")

                c4.metric("Max p", f"{ring['max_fraud_probability']:.3f}")

                st.dataframe(

                    pd.DataFrame(

                        {

                            "node_index": ring["node_indices"],

                            "txId": ring["txIds"],

                        }

                    ),

                    use_container_width=True,

                    hide_index=True,

                )



    with tab_predictions:

        pred_df = pd.DataFrame(result["predictions"])

        pred_df = pred_df.sort_values("fraud_probability", ascending=False)

        st.dataframe(pred_df, use_container_width=True, hide_index=True, height=520)



    with tab_debug:

        st.json(result["debug"])

else:

    st.info("Upload nodes and edges CSVs or use the included sample, then run analysis.")
