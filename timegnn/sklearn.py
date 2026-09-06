from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.transformer import EventLogTransformer, PrefixGCNTransformer
from .data.pyg import CustomDataset, PrefixDataset, custom_collate_fn, custom_collate_prefix
from .recipes import (
    train_gat_basic,
    train_gat_status_emb,
    train_gat_time_decay,
    train_gat_time_decay_status_emb,
    train_prefix_gcn,
)


class TimeGNNClassifier:
    """Sklearn-style wrapper around TimeGNN recipes."""

    def __init__(
        self,
        *,
        model_type: str,
        case_col: str,
        event_col: str,
        time_col: str,
        status_col: Optional[str] = None,
        cat_event: Optional[List[str]] = None,
        num_event: Optional[List[str]] = None,
        seq_cols: Optional[List[str]] = None,
        cat_seq: Optional[List[str]] = None,
        num_seq: Optional[List[str]] = None,
        config: Optional[Any] = None,
        device: Optional[str] = None,
        **overrides: Any,
    ) -> None:
        self.model_type = model_type
        self.case_col = case_col
        self.event_col = event_col
        self.time_col = time_col
        self.status_col = status_col
        self.cat_event = cat_event or []
        self.num_event = num_event or []
        self.seq_cols = seq_cols or []
        self.cat_seq = cat_seq or []
        self.num_seq = num_seq or []
        self.config = config
        self.device = device
        self.overrides = overrides

        self.result_: Optional[Dict[str, Any]] = None
        self.model_ = None
        self.transformer_ = None

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        params = {
            "model_type": self.model_type,
            "case_col": self.case_col,
            "event_col": self.event_col,
            "time_col": self.time_col,
            "status_col": self.status_col,
            "cat_event": self.cat_event,
            "num_event": self.num_event,
            "seq_cols": self.seq_cols,
            "cat_seq": self.cat_seq,
            "num_seq": self.num_seq,
            "config": self.config,
            "device": self.device,
        }
        params.update(self.overrides)
        return params

    def set_params(self, **params: Any) -> "TimeGNNClassifier":
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.overrides[key] = value
        return self

    def fit(self, event: pd.DataFrame) -> "TimeGNNClassifier":
        if self.model_type == "gat_basic":
            self.result_ = train_gat_basic(
                event,
                case_index=self.case_col,
                core_event=self.event_col,
                start_time_col=self.time_col,
                cat_col_event=self.cat_event,
                num_col_event=self.num_event,
                seq_cols=self.seq_cols,
                cat_col_seq=self.cat_seq,
                num_col_seq=self.num_seq,
                config=self.config,
                device=self.device,
                **self.overrides,
            )
            self.transformer_ = EventLogTransformer(
                case_col=self.case_col,
                event_col=self.event_col,
                time_col=self.time_col,
                cat_event=self.cat_event,
                num_event=self.num_event,
                seq_cols=self.seq_cols,
                cat_seq=self.cat_seq,
                num_seq=self.num_seq,
                mode="gat_basic",
            ).fit(event)
        elif self.model_type == "gat_status":
            self.result_ = train_gat_status_emb(
                event,
                case_index=self.case_col,
                core_event=self.event_col,
                start_time_col=self.time_col,
                status_col=self.status_col,
                cat_col_event=self.cat_event,
                num_col_event=self.num_event,
                seq_cols=self.seq_cols,
                cat_col_seq=self.cat_seq,
                num_col_seq=self.num_seq,
                config=self.config,
                device=self.device,
                **self.overrides,
            )
            self.transformer_ = EventLogTransformer(
                case_col=self.case_col,
                event_col=self.event_col,
                time_col=self.time_col,
                status_col=self.status_col,
                cat_event=self.cat_event,
                num_event=self.num_event,
                seq_cols=self.seq_cols,
                cat_seq=self.cat_seq,
                num_seq=self.num_seq,
                mode="gat_status",
            ).fit(event)
        elif self.model_type == "gat_time_decay":
            self.result_ = train_gat_time_decay(
                event,
                case_index=self.case_col,
                core_event=self.event_col,
                start_time_col=self.time_col,
                cat_col_event=self.cat_event,
                num_col_event=self.num_event,
                seq_cols=self.seq_cols,
                cat_col_seq=self.cat_seq,
                num_col_seq=self.num_seq,
                config=self.config,
                device=self.device,
                **self.overrides,
            )
            self.transformer_ = EventLogTransformer(
                case_col=self.case_col,
                event_col=self.event_col,
                time_col=self.time_col,
                cat_event=self.cat_event,
                num_event=self.num_event,
                seq_cols=self.seq_cols,
                cat_seq=self.cat_seq,
                num_seq=self.num_seq,
                mode="gat_time_decay",
            ).fit(event)
        elif self.model_type == "gat_time_decay_status":
            self.result_ = train_gat_time_decay_status_emb(
                event,
                case_index=self.case_col,
                core_event=self.event_col,
                start_time_col=self.time_col,
                status_col=self.status_col,
                cat_col_event=self.cat_event,
                num_col_event=self.num_event,
                seq_cols=self.seq_cols,
                cat_col_seq=self.cat_seq,
                num_col_seq=self.num_seq,
                config=self.config,
                device=self.device,
                **self.overrides,
            )
            self.transformer_ = EventLogTransformer(
                case_col=self.case_col,
                event_col=self.event_col,
                time_col=self.time_col,
                status_col=self.status_col,
                cat_event=self.cat_event,
                num_event=self.num_event,
                seq_cols=self.seq_cols,
                cat_seq=self.cat_seq,
                num_seq=self.num_seq,
                mode="gat_time_decay_status",
            ).fit(event)
        elif self.model_type == "prefix_gcn":
            self.result_ = train_prefix_gcn(
                event,
                case_index=self.case_col,
                core_event=self.event_col,
                start_time_col=self.time_col,
                cat_col_event=self.cat_event,
                num_col_event=self.num_event,
                cat_col_seq=self.cat_seq,
                num_col_seq=self.num_seq,
                config=self.config,
                device=self.device,
                **self.overrides,
            )
            prefix_size = getattr(self.config, "prefix_size", self.overrides.get("prefix_size", 10))
            self.transformer_ = PrefixGCNTransformer(
                case_col=self.case_col,
                event_col=self.event_col,
                time_col=self.time_col,
                prefix_size=prefix_size,
                cat_event=self.cat_event,
                num_event=self.num_event,
                cat_seq=self.cat_seq,
                num_seq=self.num_seq,
            ).fit(event)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

        self.model_ = self.result_["model"]
        return self

    def predict(self, event: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.transformer_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_.eval()

        if self.model_type == "prefix_gcn":
            transformed = self.transformer_.transform(event)
            dataset = PrefixDataset(
                transformed.event_features,
                transformed.sequence_features,
                transformed.labels,
            )
            loader = DataLoader(
                dataset, batch_size=self.overrides.get("batch_size", 32), shuffle=False, collate_fn=custom_collate_prefix
            )
            preds = []
            with torch.no_grad():
                for batch_event, batch_seq, _ in loader:
                    batch_event = batch_event.to(device)
                    batch_seq = batch_seq.to(device)
                    output = self.model_(batch_event, batch_seq)
                    preds.append(output.argmax(dim=1).cpu().numpy())
            return np.concatenate(preds) if preds else np.array([])

        transformed = self.transformer_.transform(event)
        dataset = CustomDataset(transformed.event_features, transformed.labels)
        loader = DataLoader(
            dataset,
            batch_size=self.overrides.get("batch_size", 32),
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        preds = []
        with torch.no_grad():
            for batch_event, _ in loader:
                batch_event = batch_event.to(device)
                output = self.model_(batch_event)
                preds.append(output.argmax(dim=1).cpu().numpy())
        return np.concatenate(preds) if preds else np.array([])

    def predict_proba(self, event: pd.DataFrame) -> np.ndarray:
        if self.model_ is None or self.transformer_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_.eval()

        if self.model_type == "prefix_gcn":
            transformed = self.transformer_.transform(event)
            dataset = PrefixDataset(
                transformed.event_features,
                transformed.sequence_features,
                transformed.labels,
            )
            loader = DataLoader(
                dataset, batch_size=self.overrides.get("batch_size", 32), shuffle=False, collate_fn=custom_collate_prefix
            )
            probs = []
            with torch.no_grad():
                for batch_event, batch_seq, _ in loader:
                    batch_event = batch_event.to(device)
                    batch_seq = batch_seq.to(device)
                    output = self.model_(batch_event, batch_seq)
                    probs.append(torch.softmax(output, dim=1).cpu().numpy())
            return np.vstack(probs) if probs else np.array([])

        transformed = self.transformer_.transform(event)
        dataset = CustomDataset(transformed.event_features, transformed.labels)
        loader = DataLoader(
            dataset,
            batch_size=self.overrides.get("batch_size", 32),
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        probs = []
        with torch.no_grad():
            for batch_event, _ in loader:
                batch_event = batch_event.to(device)
                output = self.model_(batch_event)
                probs.append(torch.softmax(output, dim=1).cpu().numpy())
        return np.vstack(probs) if probs else np.array([])

    def score(self, event: pd.DataFrame) -> float:
        if self.model_ is None or self.transformer_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_.eval()

        if self.model_type == "prefix_gcn":
            transformed = self.transformer_.transform(event)
            dataset = PrefixDataset(
                transformed.event_features,
                transformed.sequence_features,
                transformed.labels,
            )
            loader = DataLoader(
                dataset, batch_size=self.overrides.get("batch_size", 32), shuffle=False, collate_fn=custom_collate_prefix
            )
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_event, batch_seq, labels in loader:
                    batch_event = batch_event.to(device)
                    batch_seq = batch_seq.to(device)
                    labels = labels.to(device)
                    output = self.model_(batch_event, batch_seq)
                    preds = output.argmax(dim=1)
                    correct += preds.eq(labels).sum().item()
                    total += labels.size(0)
            return float(correct / total) if total else 0.0

        transformed = self.transformer_.transform(event)
        dataset = CustomDataset(transformed.event_features, transformed.labels)
        loader = DataLoader(
            dataset,
            batch_size=self.overrides.get("batch_size", 32),
            shuffle=False,
            collate_fn=custom_collate_fn,
        )
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_event, labels in loader:
                batch_event = batch_event.to(device)
                labels = labels.to(device)
                output = self.model_(batch_event)
                output = output.view(-1, output.size(-1))
                labels = labels.view(-1)
                mask = labels != -1
                labels = labels[mask]
                preds = output.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        return float(correct / total) if total else 0.0
