from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline import (  # noqa: E402
    DEFAULT_K_NEIGHBORS,
    DEFAULT_THRESHOLD,
    FEATURE_COUNT,
    PipelineInputError,
    build_graph_from_csv_bytes,
    build_graph_preview,
    detect_rings,
    make_predictions,
    run_inference,
)
from model.graphsage import load_model  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
LOGGER = logging.getLogger("fraud_detection_api")

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "graphsage_model.pt"

MODEL = None
STARTUP_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, STARTUP_TIME
    STARTUP_TIME = time.time()
    if MODEL_PATH.exists():
        LOGGER.info("Loading GraphSAGE model from %s", MODEL_PATH)
        MODEL = load_model(MODEL_PATH, in_channels=FEATURE_COUNT)
        LOGGER.info("Model loaded with %s parameters.", f"{sum(p.numel() for p in MODEL.parameters()):,}")
    else:
        LOGGER.warning("Model not found at %s. Inference endpoints will return 503.", MODEL_PATH)
    yield


app = FastAPI(
    title="Bitcoin Fraud Ring Detection API",
    version="2.0.0",
    description="CSV-first GraphSAGE inference API for Elliptic Bitcoin fraud ring detection.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_model():
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Put graphsage_model.pt in artifacts/ and restart the API.",
        )


def _validate_runtime_options(threshold: float, k_neighbors: int, min_ring_size: int):
    if not 0.0 <= float(threshold) <= 1.0:
        raise HTTPException(status_code=422, detail="threshold must be between 0 and 1.")
    if int(k_neighbors) < 1:
        raise HTTPException(status_code=422, detail="k_neighbors must be >= 1.")
    if int(min_ring_size) < 2:
        raise HTTPException(status_code=422, detail="min_ring_size must be >= 2.")


async def _read_upload(upload: Optional[UploadFile]) -> Optional[bytes]:
    if upload is None:
        return None
    data = await upload.read()
    if not data:
        return None
    return data


async def _run_full_pipeline(
    nodes_csv: UploadFile,
    edges_csv: Optional[UploadFile],
    *,
    threshold: float,
    graph_mode: str,
    k_neighbors: int,
    min_ring_size: int,
):
    _require_model()
    _validate_runtime_options(threshold, k_neighbors, min_ring_size)

    nodes_bytes = await _read_upload(nodes_csv)
    edges_bytes = await _read_upload(edges_csv)
    if not nodes_bytes:
        raise HTTPException(status_code=422, detail="nodes_csv is required and cannot be empty.")

    started = time.perf_counter()
    try:
        graph = build_graph_from_csv_bytes(
            nodes_bytes,
            edges_bytes,
            ARTIFACT_DIR,
            graph_mode=graph_mode,
            k_neighbors=k_neighbors,
        )

        infer_started = time.perf_counter()
        probs, preds = run_inference(MODEL, graph, threshold=threshold)
        inference_ms = round((time.perf_counter() - infer_started) * 1000, 2)

        component_rings, community_rings = detect_rings(
            graph,
            probs,
            preds,
            threshold=threshold,
            min_size=min_ring_size,
        )
        primary_rings = component_rings if component_rings else community_rings
        preview = build_graph_preview(graph, probs, primary_rings, threshold=threshold)
        predictions = make_predictions(graph, probs, preds)

    except PipelineInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Pipeline failed.")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}") from exc

    total_ms = round((time.perf_counter() - started) * 1000, 2)
    flagged_nodes = sorted({node for ring in primary_rings for node in ring["node_indices"]})
    response = {
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
            "total_pipeline_ms": total_ms,
        },
        "debug": graph.debug,
        "predictions": predictions,
        "rings": primary_rings,
        "component_rings": component_rings,
        "community_rings": community_rings,
        "graph": preview,
    }
    LOGGER.info(
        "Pipeline complete: nodes=%s edges=%s x.shape=%s edge_index.shape=%s rings=%s illicit=%s",
        graph.num_nodes,
        graph.num_edges,
        graph.debug["x_shape"],
        graph.debug["edge_index_shape"],
        len(primary_rings),
        int(preds.sum()),
    )
    return response


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "uptime_s": round(time.time() - STARTUP_TIME, 1),
        "artifacts": {
            "model": MODEL_PATH.exists(),
            "col_min": (ARTIFACT_DIR / "col_min.npy").exists() or (BASE_DIR / "col_min.npy").exists(),
            "col_rng": (ARTIFACT_DIR / "col_rng.npy").exists() or (BASE_DIR / "col_rng.npy").exists(),
        },
    }


@app.get("/model-info", tags=["System"])
def model_info():
    _require_model()
    return {
        "architecture": "GraphSAGE",
        "input_features": FEATURE_COUNT,
        "layers": [256, 128, 64, 2],
        "default_threshold": DEFAULT_THRESHOLD,
        "normalization": "saved col_min.npy and col_rng.npy from training",
        "total_params": int(sum(p.numel() for p in MODEL.parameters())),
    }


@app.post("/full-pipeline", tags=["Inference"])
async def full_pipeline(
    nodes_csv: UploadFile = File(..., description="CSV with columns txId,time_step,f0..f164."),
    edges_csv: Optional[UploadFile] = File(None, description="Optional CSV with txId1,txId2 or source,target."),
    threshold: float = Form(DEFAULT_THRESHOLD),
    graph_mode: str = Form("auto"),
    k_neighbors: int = Form(DEFAULT_K_NEIGHBORS),
    min_ring_size: int = Form(3),
):
    return await _run_full_pipeline(
        nodes_csv,
        edges_csv,
        threshold=threshold,
        graph_mode=graph_mode,
        k_neighbors=k_neighbors,
        min_ring_size=min_ring_size,
    )


@app.post("/predict", tags=["Inference"])
async def predict(
    nodes_csv: UploadFile = File(..., description="CSV with columns txId,time_step,f0..f164."),
    edges_csv: Optional[UploadFile] = File(None, description="Optional CSV with txId1,txId2 or source,target."),
    threshold: float = Form(DEFAULT_THRESHOLD),
    graph_mode: str = Form("auto"),
    k_neighbors: int = Form(DEFAULT_K_NEIGHBORS),
    min_ring_size: int = Form(3),
):
    result = await _run_full_pipeline(
        nodes_csv,
        edges_csv,
        threshold=threshold,
        graph_mode=graph_mode,
        k_neighbors=k_neighbors,
        min_ring_size=min_ring_size,
    )
    return {
        "summary": result["summary"],
        "debug": result["debug"],
        "predictions": result["predictions"],
    }


@app.post("/detect-rings", tags=["Inference"])
async def detect_fraud_rings(
    nodes_csv: UploadFile = File(..., description="CSV with columns txId,time_step,f0..f164."),
    edges_csv: Optional[UploadFile] = File(None, description="Optional CSV with txId1,txId2 or source,target."),
    threshold: float = Form(DEFAULT_THRESHOLD),
    graph_mode: str = Form("auto"),
    k_neighbors: int = Form(DEFAULT_K_NEIGHBORS),
    min_ring_size: int = Form(3),
):
    result = await _run_full_pipeline(
        nodes_csv,
        edges_csv,
        threshold=threshold,
        graph_mode=graph_mode,
        k_neighbors=k_neighbors,
        min_ring_size=min_ring_size,
    )
    return {
        "summary": result["summary"],
        "debug": result["debug"],
        "rings": result["rings"],
        "component_rings": result["component_rings"],
        "community_rings": result["community_rings"],
        "graph": result["graph"],
    }
