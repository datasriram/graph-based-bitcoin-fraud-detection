# Bitcoin Fraud Detection & Graph Analytics

**GraphSAGE · PyTorch Geometric · FastAPI · Streamlit**

An end-to-end machine learning application for scoring the risk of Bitcoin transactions and exploring *suspected* high-risk transaction clusters. It uses a four-layer GraphSAGE graph neural network (GNN) trained on the Elliptic Bitcoin transaction dataset, served through a FastAPI backend and an interactive Streamlit dashboard.

> **Important:** A high-risk prediction or connected component is not proof of fraud or coordinated criminal activity. This is a research/portfolio application, not a production AML decision system.

## Features

- Transaction-level illicit-risk scores and binary predictions.
- Interactive transaction graph with cluster exploration, zoom, pan, and node details.
- Suspected high-risk cluster discovery using connected components and community detection.
- CSV-based inference through FastAPI and Streamlit.
- Bundled sample node and edge CSVs for a quick demo.
- Adjustable decision threshold and minimum cluster size.
- Optional experimental KNN and hybrid graph-construction modes (not covered by the reported test metrics).

## Architecture

```text
Nodes CSV + Edges CSV
          |
          v
Streamlit interface ---> FastAPI (/full-pipeline)
                              |
                              v
                 Validate data and map IDs
                              |
                              v
                 Apply saved training-only
                 imputation and normalization
                              |
                              v
                 Build transaction graph
                              |
                              v
                 GraphSAGE inference
                              |
                              v
                 Scores + suspected clusters
                              |
                              v
                 Interactive visualization
```

The model consists of four GraphSAGE convolutions with output dimensions **256 → 128 → 64 → 2**, with batch normalization, ReLU, and dropout (`0.4`) between layers. Training uses class-weighted focal loss to address label imbalance. Total trainable parameters: **168,514**.

## Dataset

[Elliptic Bitcoin Dataset on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

| Property | Value |
|---|---:|
| Transactions | 203,769 |
| Transaction edges | 234,355 |
| Input features per transaction | 166 |
| Time steps | 49 |
| Labeled illicit | 4,545 |
| Labeled licit | 42,019 |
| Unlabeled | 157,205 |

The 166 model inputs comprise the time-step feature and 165 transaction features. Labels are interpreted as `1 = illicit`, `2 = licit`, and `unknown = unlabeled`; unlabeled nodes can contribute graph context but are excluded from supervised loss and labeled evaluation.

## Evaluation methodology

To avoid selecting a model using test-set results, the updated experiment uses a **chronological split**:

| Split | Time steps | Labeled transactions | Illicit | Licit |
|---|---|---:|---:|---:|
| Train | 1–30 | 26,905 | 2,954 | 23,951 |
| Validation | 31–35 | 4,330 | 690 | 3,640 |
| Test | 36–49 | 15,329 | 901 | 14,428 |

- Missing-value imputation and min/range normalization are fitted **only on training-period nodes**, then reused in validation, testing, and application inference.
- The training graph contains nodes and edges available through time step 30; validation adds nodes and edges through step 35; testing uses the full graph through step 49. This is **chronological snapshot evaluation**, not a per-transaction online replay.
- The checkpoint is selected by **validation average precision (PR-AUC)**, evaluated every 10 epochs.
- The classification threshold is selected by maximizing **validation F1** and then held fixed for testing.
- The held-out test period is used for final reported metrics, not for checkpoint or threshold selection.

### Results: held-out test period

| Metric | Result |
|---|---:|
| ROC-AUC | **0.9042** |
| Average precision / PR-AUC | **0.6070** |
| Illicit-class precision | **0.5740** |
| Illicit-class recall | **0.6115** |
| Illicit-class F1 | **0.5922** |
| Accuracy | 0.9505 |
| Validation-selected threshold | **0.802532** |
| Selected epoch | **150** |

Test confusion matrix (rows = actual, columns = predicted):

| | Predicted licit | Predicted illicit |
|---|---:|---:|
| Actual licit | 14,019 | 409 |
| Actual illicit | 350 | 551 |

The model detected **551 of 901** labeled illicit transactions in the held-out period and missed **350**. In this imbalanced setting, PR-AUC, precision, and recall are more informative than accuracy alone.

Validation results were higher (PR-AUC **0.9153**, F1 **0.8743**). That gap highlights limitations in generalizing to later time periods; these test results should not be interpreted as live operational performance.

The checkpoint, saved preprocessing values, and selected threshold are under `validated_graphsage_export/artifacts/`; corresponding results are under `validated_graphsage_export/evaluation/`.

## Run locally

Use **Python 3.13**, which was used to test the application locally. Run the following commands from the repository root.

```powershell
# Clone
git clone https://github.com/datasriram/graph-based-bitcoin-fraud-detection.git
cd graph-based-bitcoin-fraud-detection

# Create and activate a virtual environment (Windows PowerShell)
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, the commands can instead be run with `.\.venv\Scripts\python.exe` without activating.

Start the backend in the first terminal:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal (with the same environment activated):

```powershell
python -m streamlit run frontend/app.py
```

Open **http://localhost:8501**. The API health endpoint is **http://127.0.0.1:8000/health**, and interactive API documentation is at **http://127.0.0.1:8000/docs**.

### Try the included sample

1. Leave **Use sample CSVs** enabled in the Streamlit sidebar.
2. Select **uploaded** graph mode (the default).
3. Keep the validation-selected threshold (`0.802532`) initially.
4. Click **Run analysis** to view scores, clusters, and the interactive graph.

For custom data, upload both a nodes CSV and the corresponding transaction edges CSV.

## CSV format

**Nodes:** `txId,time_step,f0,f1,...,f164` (166 features after the transaction ID). An optional class/label column can be present, but it is not used to generate predictions.

**Edges:** `txId1,txId2`, identifying linked transactions present in the nodes file. Uploaded edges are the **recommended and validated graph construction family**.

The default mode requires an edges CSV. Other modes (`auto`, `knn`, `hybrid`) are available for experimentation, but their predictions **must not be represented as having the same validated test performance**. Predictions on small uploaded subgraphs can also differ from predictions made with full chronological historical graph context.

## API endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Backend and model-loading status |
| `/model-info` | GET | Architecture, threshold, and evaluation summary |
| `/full-pipeline` | POST | Predictions, graph preview, and suspected clusters |
| `/predict` | POST | Transaction-level predictions |
| `/detect-rings` | POST | Suspected high-risk clusters and graph preview |

Inference endpoints accept multipart form uploads (`nodes_csv`, `edges_csv`) and optional `threshold`, `graph_mode`, `k_neighbors`, and `min_ring_size` parameters.

## Repository layout

```text
api/                            FastAPI backend
core/                           CSV processing, graph building, inference
frontend/                       Streamlit interface and graph visualization
model/                          GraphSAGE network and checkpoint loader
data/                           Sample nodes and edges
notebooks/                      Training notebook(s)
validated_graphsage_export/
  artifacts/                    Validated checkpoint, normalizers, threshold
  evaluation/                   Held-out metrics and training history
requirements.txt                Application dependencies
README.md
```

Legacy experiments and generated outputs may also be retained in the repository for reference; the application uses the **validated_graphsage_export** artifacts. **Note:** the legacy notebook in `notebooks/` may still contain the earlier evaluation procedure; do not rerun it as a reproduction of these validated metrics until the corrected Colab notebook is saved and committed.

## Limitations and future work

- The held-out benchmark uses chronological graph snapshots with historical context, **not true real-time transaction arrival**.
- A cluster of high-risk transactions does not establish a fraud ring; analyst verification would be needed.
- Threshold performance can shift with fraud prevalence and transaction behavior.
- Results for arbitrary uploaded subgraphs and KNN/hybrid edges have not been independently benchmarked.
- A simple feature-only baseline and deeper analysis of temporal shift would be worthwhile future comparisons.

## Author

Sriram Tummala

## Usage

Academic and portfolio demonstration. Dataset usage is subject to the source dataset's terms. No open-source license is asserted here unless a separate license file is added.
