from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    """Exact GraphSAGE architecture from the training notebook."""

    def __init__(self, in_channels: int = 166):
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


def _torch_load(path: Path, device: str) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        if checkpoint and all(hasattr(value, "shape") for value in checkpoint.values()):
            return checkpoint
    raise RuntimeError("Model file does not look like a PyTorch state_dict.")


def _remap_state_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped = {}
    replacements = {
        "module.": "",
        "conv1": "c1",
        "conv2": "c2",
        "conv3": "c3",
        "conv4": "c4",
        "bn1": "b1",
        "bn2": "b2",
        "bn3": "b3",
        "dropout": "drop",
    }
    for key, value in state.items():
        new_key = key
        for old, new in replacements.items():
            new_key = new_key.replace(old, new)
        remapped[new_key] = value
    return remapped


def load_model(model_path: str | Path, in_channels: int = 166, device: str = "cpu") -> GraphSAGE:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = GraphSAGE(in_channels=in_channels).to(device)
    checkpoint = _torch_load(path, device)
    state = _extract_state_dict(checkpoint)

    try:
        model.load_state_dict(state)
    except RuntimeError:
        model.load_state_dict(_remap_state_keys(state))

    model.eval()
    return model
