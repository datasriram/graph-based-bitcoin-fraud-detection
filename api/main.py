
from __future__ import annotations

import logging
import math
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline import (
    DEFAULT_K_NEIGHBORS,
    FEATURE_COUNT,
    PipelineInputError,
    build_graph_from_csv_bytes,
    build_graph_preview,
    detect_rings,
    make_predictions,
    run_inference,
)
from model.graphsage import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

LOGGER = logging.getLogger("fraud_detection_api")

BASE_DIR = Path(__file__).resolve().parent.parent

# Validated model artifacts
ARTIFACT_DIR = (
    BASE_DIR / "validated_graphsage_export" / "artifacts"
)

MODEL_PATH = ARTIFACT_DIR / "graphsage_model.pt"
THRESHOLD_PATH = ARTIFACT_DIR / "threshold.npy"

FALLBACK_THRESHOLD = 0.802532

# Limit uploaded CSV size to 20 MB per file
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

MODEL = None
VALIDATED_THRESHOLD = FALLBACK_THRESHOLD
STARTUP_TIME = time.time()


def load_validated_threshold() -> float:
    if not THRESHOLD_PATH.exists():
        raise RuntimeError(
            f"Validated threshold artifact missing: {THRESHOLD_PATH}"
        )

    value = np.load(THRESHOLD_PATH, allow_pickle=False)

    if value.size != 1:
        raise RuntimeError(
            "Threshold artifact must contain exactly one number."
        )

    threshold = float(value.item())

    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise RuntimeError(
            "Validated threshold must be between 0 and 1."
        )

    return threshold


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, VALIDATED_THRESHOLD, STARTUP_TIME

    STARTUP_TIME = time.time()
    MODEL = None

    required_artifacts = [
        MODEL_PATH,
        THRESHOLD_PATH,
        ARTIFACT_DIR / "col_min.npy",
        ARTIFACT_DIR / "col_rng.npy",
        ARTIFACT_DIR / "col_median.npy",
    ]

    missing = [
        str(path)
        for path in required_artifacts
        if not path.is_file()
    ]

    if missing:
        LOGGER.error(
            "Missing validated artifacts: %s",
            missing,
        )
        raise RuntimeError(
            "Validated model artifacts are incomplete."
        )

    LOGGER.info(
        "Loading validated GraphSAGE model from %s",
        MODEL_PATH,
    )

    MODEL = load_model(
        MODEL_PATH,
        in_channels=FEATURE_COUNT,
    )

    MODEL.eval()

    VALIDATED_THRESHOLD = load_validated_threshold()

    LOGGER.info(
        "Model loaded successfully. Parameters: %s",
        f"{sum(p.numel() for p in MODEL.parameters()):,}",
    )

    LOGGER.info(
        "Validated classification threshold: %.6f",
        VALIDATED_THRESHOLD,
    )

    try:
        yield
    finally:
        MODEL = None
        LOGGER.info("API shutdown complete.")


app = FastAPI(
    title="Bitcoin Fraud Detection API",
    version="3.0.0",
    description=(
        "GraphSAGE-based Bitcoin transaction risk scoring "
        "and suspected fraud-cluster analysis using the "
        "Elliptic dataset."
    ),
    lifespan=lifespan,
)

# Streamlit communicates with FastAPI through Python requests.
# Restrict permissive browser access for public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _require_model():
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Validated GraphSAGE model is not loaded.",
        )


def _validate_runtime_options(
    threshold: float,
    graph_mode: str,
    k_neighbors: int,
    min_ring_size: int,
):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise HTTPException(
            status_code=422,
            detail="threshold must be between 0 and 1.",
        )

    if graph_mode not in {
        "uploaded",
        "auto",
        "knn",
        "hybrid",
    }:
        raise HTTPException(
            status_code=422,
            detail="Invalid graph construction mode.",
        )

    if not 1 <= k_neighbors <= 50:
        raise HTTPException(
            status_code=422,
            detail="k_neighbors must be between 1 and 50.",
        )

    if not 2 <= min_ring_size <= 100:
        raise HTTPException(
            status_code=422,
            detail="min_ring_size must be between 2 and 100.",
        )


async def _read_upload(
    upload: Optional[UploadFile],
) -> Optional[bytes]:

    if upload is None:
        return None

    chunks = []
    total_size = 0

    while True:
        chunk = await upload.read(1024 * 1024)

        if not chunk:
            break

        total_size += len(chunk)

        if total_size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="CSV file exceeds the 20 MB upload limit.",
            )

        chunks.append(chunk)

    return b"".join(chunks) or None


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

    _validate_runtime_options(
        threshold,
        graph_mode,
        k_neighbors,
        min_ring_size,
    )

    nodes_bytes = await _read_upload(nodes_csv)
    edges_bytes = await _read_upload(edges_csv)

    if not nodes_bytes:
        raise HTTPException(
            status_code=422,
            detail="nodes_csv is required and cannot be empty.",
        )

    # Default validated workflow uses real transaction edges.
    if graph_mode == "uploaded" and not edges_bytes:
        raise HTTPException(
            status_code=422,
            detail=(
                "An edges CSV is required for uploaded graph mode. "
                "The validated GraphSAGE model was trained "
                "using transaction relationships."
            ),
        )

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

        probs, preds = run_inference(
            MODEL,
            graph,
            threshold=threshold,
        )

        inference_ms = round(
            (time.perf_counter() - infer_started) * 1000,
            2,
        )

        component_rings, community_rings = detect_rings(
            graph,
            probs,
            preds,
            threshold=threshold,
            min_size=min_ring_size,
        )

        primary_rings = (
            component_rings
            if component_rings
            else community_rings
        )

        preview = build_graph_preview(
            graph,
            probs,
            primary_rings,
            threshold=threshold,
        )

        predictions = make_predictions(
            graph,
            probs,
            preds,
        )

    except PipelineInputError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        LOGGER.exception("Fraud detection pipeline failed.")

        raise HTTPException(
            status_code=500,
            detail=(
                "An internal error occurred during inference. "
                "Please verify your CSV files."
            ),
        ) from exc

    total_ms = round(
        (time.perf_counter() - started) * 1000,
        2,
    )

    flagged_nodes = sorted({
        node
        for ring in primary_rings
        for node in ring["node_indices"]
    })

    warnings = list(graph.warnings)

    if graph_mode in {"auto", "knn", "hybrid"}:
        warnings.append(
            "KNN or automatically constructed graph relationships "
            "have not been validated against the reported "
            "held-out test performance."
        )

    warnings.append(
        "Detected groups are suspected risk clusters, "
        "not confirmed coordinated fraud rings."
    )

    response = {
        "summary": {
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
            "x_shape": graph.debug["x_shape"],
            "edge_index_shape": graph.debug[
                "edge_index_shape"
            ],
            "graph_method": graph.graph_method,
            "threshold": float(threshold),
            "predicted_illicit": int(preds.sum()),
            "illicit_pct": (
                round(float(preds.mean()) * 100, 2)
                if graph.num_nodes
                else 0.0
            ),
            "total_rings": len(primary_rings),
            "total_flagged_ring_nodes": len(flagged_nodes),
            "inference_ms": inference_ms,
            "total_pipeline_ms": total_ms,
        },
        "debug": {
            **graph.debug,
            "warnings": warnings,
        },
        "predictions": predictions,
        "rings": primary_rings,
        "component_rings": component_rings,
        "community_rings": community_rings,
        "graph": preview,
    }

    LOGGER.info(
        "Pipeline complete: nodes=%s edges=%s "
        "method=%s illicit=%s clusters=%s",
        graph.num_nodes,
        graph.num_edges,
        graph.graph_method,
        int(preds.sum()),
        len(primary_rings),
    )

    return response


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "ok" if MODEL is not None else "degraded",
        "model_loaded": MODEL is not None,
        "uptime_s": round(time.time() - STARTUP_TIME, 1),
        "artifacts": {
            "model": MODEL_PATH.exists(),
            "col_min": (
                ARTIFACT_DIR / "col_min.npy"
            ).exists(),
            "col_rng": (
                ARTIFACT_DIR / "col_rng.npy"
            ).exists(),
            "col_median": (
                ARTIFACT_DIR / "col_median.npy"
            ).exists(),
            "threshold": THRESHOLD_PATH.exists(),
        },
    }


@app.get("/model-info", tags=["System"])
def model_info():
    _require_model()

    return {
        "architecture": "GraphSAGE",
        "input_features": FEATURE_COUNT,
        "layers": [256, 128, 64, 2],
        "default_threshold": VALIDATED_THRESHOLD,
        "default_graph_mode": "uploaded",
        "normalization": (
            "Training-period-only saved median, "
            "minimum and range statistics"
        ),
        "evaluation": {
            "dataset": "Elliptic Bitcoin Dataset",
            "train_time_steps": "1-30",
            "validation_time_steps": "31-35",
            "test_time_steps": "36-49",
            "test_roc_auc": 0.9042,
            "test_pr_auc": 0.6070,
            "test_precision": 0.5740,
            "test_recall": 0.6115,
            "test_f1": 0.5922,
        },
        "total_params": int(
            sum(p.numel() for p in MODEL.parameters())
        ),
        "note": (
            "Reported metrics were measured on chronological "
            "graph snapshots with historical graph context. "
            "They are not guaranteed for arbitrary uploaded "
            "subgraphs or synthetic KNN graphs."
        ),
    }


@app.post("/full-pipeline", tags=["Inference"])
async def full_pipeline(
    nodes_csv: UploadFile = File(
        ...,
        description=(
            "CSV containing txId,time_step,f0..f164."
        ),
    ),
    edges_csv: Optional[UploadFile] = File(
        None,
        description="CSV containing txId1,txId2.",
    ),
    threshold: Optional[float] = Form(None),
    graph_mode: str = Form("uploaded"),
    k_neighbors: int = Form(DEFAULT_K_NEIGHBORS),
    min_ring_size: int = Form(3),
):
    return await _run_full_pipeline(
        nodes_csv,
        edges_csv,
        threshold=(
            VALIDATED_THRESHOLD
            if threshold is None
            else threshold
        ),
        graph_mode=graph_mode,
        k_neighbors=k_neighbors,
        min_ring_size=min_ring_size,
    )


@app.post("/predict", tags=["Inference"])
async def predict(
    nodes_csv: UploadFile = File(
        ...,
        description="CSV containing txId,time_step,f0..f164.",
    ),
    edges_csv: Optional[UploadFile] = File(
        None,
        description="CSV containing txId1,txId2.",
    ),
    threshold: Optional[float] = Form(None),
    graph_mode: str = Form("uploaded"),
    k_neighbors: int = Form(DEFAULT_K_NEIGHBORS),
    min_ring_size: int = Form(3),
):
    result = await _run_full_pipeline(
        nodes_csv,
        edges_csv,
        threshold=(
            VALIDATED_THRESHOLD
            if threshold is None
            else threshold
        ),
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
    nodes_csv: UploadFile = File(
        ...,
        description="CSV containing txId,time_step,f0..f164.",
    ),
    edges_csv: Optional[UploadFile] = File(
        None,
        description="CSV containing txId1,txId2.",
    ),
    threshold: Optional[float] = Form(None),
    graph_mode: str = Form("uploaded"),
    k_neighbors: int = Form(DEFAULT_K_NEIGHBORS),
    min_ring_size: int = Form(3),
):
    result = await _run_full_pipeline(
        nodes_csv,
        edges_csv,
        threshold=(
            VALIDATED_THRESHOLD
            if threshold is None
            else threshold
        ),
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
