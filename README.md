# Bitcoin Fraud Detection & Graph Analytics

**GraphSAGE · PyTorch Geometric · Streamlit**

An interactive machine learning application that scores Bitcoin transactions for illicit-activity risk and explores **suspected high-risk transaction clusters**. A four-layer GraphSAGE graph neural network (GNN) is trained on the [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set). The Streamlit application runs model inference directly—**no separate API server is required**.

> **Important:** A high-risk prediction or connected component is not proof of fraud or coordinated criminal activity. This is a research and portfolio demonstration, not a production anti-money-laundering decision system.

## Features

- Transaction-level illicit-risk scores and binary predictions.
- Interactive graph exploration with zoom, pan, node inspection, and cluster selection.
- Suspected high-risk cluster exploration using connected components and community detection.
- Upload transaction-node and transaction-edge CSVs, or use the bundled sample data.
- Adjustable classification threshold and minimum cluster size.
- Optional experimental KNN and hybrid graph modes. **Reported test metrics do not validate these alternative modes.**
- Local, in-process GraphSAGE inference within Streamlit.

## How it works

```text
Nodes CSV + Edges CSV
        |
        v
Streamlit interface
        |
        v
Validate inputs and map transaction IDs
        |
        v
Apply saved training-period imputation and normalization
        |
        v
Construct transaction graph
        |
        v
GraphSAGE model inference
        |
        v
Risk scores + suspected high-risk clusters
        |
        v
Interactive graph and prediction tables
```

### Model architecture

| Layer | Input features | Output features |
|---|---:|---:|
| GraphSAGE convolution 1 | 166 | 256 |
| GraphSAGE convolution 2 | 256 | 128 |
| GraphSAGE convolution 3 | 128 | 64 |
| GraphSAGE convolution 4 | 64 | 2 |

The model uses batch normalization, ReLU, and dropout (`0.4`) between the convolution layers. Training uses class-weighted focal loss to address class imbalance. **Trainable parameters: 168,514.**

**Stack:** Python, PyTorch, PyTorch Geometric, Streamlit, NetworkX, NumPy, and Pandas.

## Dataset

The [Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) represents Bitcoin transactions as graph nodes, with directed edges between related transactions.

| Property | Value |
|---|---:|
| Transactions | 203,769 |
| Transaction edges | 234,355 |
| Model input features | 166 |
| Time steps | 49 |
| Labeled illicit | 4,545 |
| Labeled licit | 42,019 |
| Unlabeled | 157,205 |

The 166 inputs consist of the time-step feature and 165 transaction features. Original labels use `1 = illicit`, `2 = licit`, and `unknown = unlabeled`. Unlabeled nodes can supply graph context but are excluded from supervised training loss and labeled evaluation.

## Evaluation methodology

The corrected experiment uses a **chronological split** rather than choosing the checkpoint or threshold using test results.

| Split | Time steps | Labeled transactions | Illicit | Licit |
|---|---|---:|---:|---:|
| Train | 1–30 | 26,905 | 2,954 | 23,951 |
| Validation | 31–35 | 4,330 | 690 | 3,640 |
| Test | 36–49 | 15,329 | 901 | 14,428 |

- Imputation and min/range normalization are fitted using **training-period nodes only**; the same saved parameters are used for validation, test, and application inference.
- Training uses graph data through time step 30; validation uses a graph snapshot through step 35; the held-out test uses a snapshot through step 49. This is **chronological snapshot evaluation**, not a per-transaction streaming replay.
- The best checkpoint is selected using **validation average precision (PR-AUC)**.
- The decision threshold is selected by maximizing **validation F1** and kept fixed during final test evaluation.
- The held-out test period is not used to select the checkpoint or threshold.

### Held-out test results

| Metric | Score |
|---|---:|
| ROC-AUC | **0.9042** |
| Average precision / PR-AUC | **0.6070** |
| Illicit precision | **0.5740** |
| Illicit recall | **0.6115** |
| Illicit F1 | **0.5922** |
| Accuracy | 0.9505 |
| Validation-selected threshold | **0.802532** |
| Selected epoch | **150** |

**Test confusion matrix** (rows: actual class; columns: predicted class):

| | Predicted licit | Predicted illicit |
|---|---:|---:|
| Actual licit | 14,019 | 409 |
| Actual illicit | 350 | 551 |

The model identified **551 of 901** labeled illicit transactions in the held-out test period. Because the classes are imbalanced, PR-AUC, precision, and recall convey more than accuracy alone. Validation PR-AUC was **0.9153** and validation F1 was **0.8743**, indicating a substantial drop in performance on later transactions.

The checkpoint and fitted preprocessing parameters are in `validated_graphsage_export/artifacts/`; evaluation summaries are in `validated_graphsage_export/evaluation/`. The corrected experiment notebook is `notebooks/Bitcoin_GNN_Leakage_Controlled.ipynb`.

## Run locally

**Python 3.13** was used for local testing. Run commands from the repository root.

```powershell
git clone https://github.com/datasriram/graph-based-bitcoin-fraud-detection.git
cd graph-based-bitcoin-fraud-detection

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run frontend/app.py
```

Open the local Streamlit URL shown in the terminal (typically `http://localhost:8501`). **FastAPI and a second terminal are not required.** If PowerShell prevents activation, use `.\.venv\Scripts\python.exe` for the installation and launch commands instead.

### Try the sample data

1. Leave **Use sample CSVs** enabled.
2. Select **uploaded** graph mode (recommended).
3. Keep the validation-selected threshold (`0.802532`) for an initial run.
4. Click **Run analysis** to view transaction scores, suspected clusters, and the interactive graph.

The sample demonstrates application functionality; its predictions are **not** a reproduction of the full chronological test benchmark.

### CSV inputs

**Nodes CSV:** `txId,time_step,f0,f1,...,f164` (166 features after the transaction ID). An optional label column may be present but is not used to calculate predictions.

**Edges CSV:** `txId1,txId2`, referencing transaction IDs in the uploaded nodes file.

The default **uploaded** mode requires transaction edges. Experimental `auto`, `knn`, and `hybrid` graph modes may produce different results and have not been benchmarked against the reported held-out test metrics. Predictions on small subgraphs may differ from full-snapshot predictions because the graph context changes.

## Deploy on Streamlit Community Cloud

1. Push the standalone application and its dependencies to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and create an app from the repository.
3. Select repository `datasriram/graph-based-bitcoin-fraud-detection`, branch `main`, and entrypoint **`frontend/app.py`**.
4. Select a compatible Python version (locally tested with **3.13**) and deploy.
5. Use the bundled sample data to verify model loading, predictions, and graph visualization.

There is no API URL, FastAPI server, Render service, or external model API key to configure. Free hosting has resource limits; large user-uploaded transaction graphs may exceed the available memory or compute budget.

## Repository structure

```text
core/                             CSV processing, preprocessing, graph building, inference
frontend/app.py                   Standalone Streamlit app and interactive visualization
model/                            GraphSAGE architecture and checkpoint loader
data/                             Sample nodes and edges
notebooks/Bitcoin_GNN_Leakage_Controlled.ipynb
validated_graphsage_export/
  artifacts/                      Model checkpoint, preprocessing values, threshold
  evaluation/                     Held-out test metrics and training history
screenshots/                      Application screenshots (if retained)
requirements.txt                  Streamlit inference dependencies
README.md
.gitignore
```

## Limitations and future work

- Chronological test **snapshots** retain graph context available at the snapshot endpoint; the benchmark is not strict online inference at each transaction's arrival time.
- High-risk connected components are **suspected clusters**, not confirmed fraud rings.
- The threshold and performance may shift as fraud patterns and class prevalence change.
- Arbitrary user-uploaded subgraphs and synthetic KNN/hybrid edges have not been independently validated.
- Further work could compare feature-only baselines and examine temporal generalization and probability calibration.

## Author

Sriram Tummala

## Usage

Academic and portfolio demonstration. Dataset use is subject to the source dataset's terms. No open-source license is asserted unless a separate license file is added.
