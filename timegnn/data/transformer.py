from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

from .encoding import (
    encode_event_prefix_label,
    encode_pad_sequence,
    node_time_list,
    scale_time_differences_fast_fixed,
)
from ..utils.sequence import pad_sequences
from .pyg import (
    prepare_data_core,
    prepare_data_core_2edges,
    prepare_data_core_timedif,
    prepare_data_prefix,
    prepare_data_y,
)


@dataclass
class TransformOutput:
    """Container for transformer outputs."""
    event_features: List[torch.Tensor]
    labels: Any
    core_size: int
    output_size: int
    trans_size: Optional[int] = None
    sequence_features: Optional[torch.Tensor] = None


class EventLogTransformer:
    """Sklearn-style transformer that builds PyG inputs from event logs."""
    def __init__(
        self,
        *,
        case_col: str,
        event_col: str,
        time_col: str,
        status_col: Optional[str] = None,
        cat_event: Optional[List[str]] = None,
        num_event: Optional[List[str]] = None,
        seq_cols: Optional[List[str]] = None,
        cat_seq: Optional[List[str]] = None,
        num_seq: Optional[List[str]] = None,
        cat_mask: bool = True,
        num_mask: bool = True,
        mode: str = "gat_basic",
    ) -> None:
        self.case_col = case_col
        self.event_col = event_col
        self.time_col = time_col
        self.status_col = status_col
        self.cat_event = cat_event or []
        self.num_event = num_event or []
        self.seq_cols = seq_cols or []
        self.cat_seq = cat_seq or []
        self.num_seq = num_seq or []
        self.cat_mask = cat_mask
        self.num_mask = num_mask
        self.mode = mode

        self._fitted = False
        self._event_encoder: Optional[LabelEncoder] = None
        self._event_cat_encoder: Optional[OneHotEncoder] = None
        self._event_num_scaler: Optional[MinMaxScaler] = None
        self._event_num_scalers: Dict[str, MinMaxScaler] = {}
        self._event_num_medians: Optional[pd.Series] = None
        self._seq_cat_encoder: Optional[OneHotEncoder] = None
        self._seq_num_scaler: Optional[MinMaxScaler] = None
        self._seq_num_scalers: Dict[str, MinMaxScaler] = {}
        self._seq_num_medians: Optional[pd.Series] = None
        self._transition_encoder: Optional[LabelEncoder] = None
        self._time_scaler: Optional[MinMaxScaler] = None

    def fit(self, event: pd.DataFrame) -> "EventLogTransformer":
        """Fit encoders and scalers on the provided event log."""
        event = event.copy()
        self._event_encoder = LabelEncoder()
        self._event_encoder.fit(event[self.event_col].tolist() + ["EOS"])

        if self.cat_event:
            self._event_cat_encoder = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            )
            self._event_cat_encoder.fit(event[self.cat_event])

        if self.num_event:
            if self.num_mask:
                for col in self.num_event:
                    valid = event[col].replace(-1, np.nan).dropna()
                    if not valid.empty:
                        scaler = MinMaxScaler()
                        scaler.fit(valid.values.reshape(-1, 1))
                        self._event_num_scalers[col] = scaler
            else:
                self._event_num_medians = event[self.num_event].replace(-1, np.nan).median()
                filled = event[self.num_event].replace(-1, np.nan).fillna(self._event_num_medians)
                self._event_num_scaler = MinMaxScaler().fit(filled)

        sequence = self._build_sequence_table(event)
        if self.cat_seq:
            self._seq_cat_encoder = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            )
            self._seq_cat_encoder.fit(sequence[self.cat_seq])

        if self.num_seq:
            if self.num_mask:
                for col in self.num_seq:
                    valid = sequence[col].replace(-1, np.nan).dropna()
                    if not valid.empty:
                        scaler = MinMaxScaler()
                        scaler.fit(valid.values.reshape(-1, 1))
                        self._seq_num_scalers[col] = scaler
            else:
                self._seq_num_medians = sequence[self.num_seq].replace(-1, np.nan).median()
                filled = sequence[self.num_seq].replace(-1, np.nan).fillna(self._seq_num_medians)
                self._seq_num_scaler = MinMaxScaler().fit(filled)

        if self.status_col:
            transitions = self._collect_transitions(event, sequence)
            self._transition_encoder = LabelEncoder()
            self._transition_encoder.fit(transitions + ["UNK"])

        self._time_scaler = MinMaxScaler()
        time_diffs = self._collect_time_diffs(event, sequence)
        if time_diffs.size > 0:
            self._time_scaler.fit(time_diffs.reshape(-1, 1))

        self._fitted = True
        return self

    def transform(self, event: pd.DataFrame) -> TransformOutput:
        """Transform the event log into model-ready graph inputs."""
        if not self._fitted:
            raise RuntimeError("Transformer not fitted. Call fit() first.")

        sequence = self._build_sequence_table(event)
        core_encode, y_encode, core_size, output_size = self._encode_labels(event)
        event_encode = self._encode_event_features(event)
        sequence_encode = self._encode_sequence_features(sequence)

        max_num_events = event_encode.shape[1]

        if sequence_encode.size > 0:
            sequence_features_expanded = np.expand_dims(sequence_encode, axis=1)
            sequence_features_expanded = np.repeat(sequence_features_expanded, max_num_events, axis=1)
            combined_features = np.concatenate((event_encode, sequence_features_expanded), axis=2)
        else:
            combined_features = event_encode

        if self.mode == "gat_basic":
            scaled_time_diffs = self._scale_time_diffs(event, sequence)
            node_times = node_time_list(event, self.time_col, self.case_col)
            event_feature_list = prepare_data_core_timedif(
                combined_features, core_encode, scaled_time_diffs, node_times
            )
        elif self.mode == "gat_time_decay":
            node_times = node_time_list(event, self.time_col, self.case_col)
            event_feature_list = prepare_data_core(combined_features, core_encode, node_times)
        elif self.mode in {"gat_status", "gat_time_decay_status"}:
            event_trans_edge, trans_size = self._encode_transitions(event, sequence)
            scaled_time_diffs = self._scale_time_diffs(event, sequence)
            node_times = None
            if self.mode == "gat_time_decay_status":
                node_times = node_time_list(event, self.time_col, self.case_col)
            event_feature_list = prepare_data_core_2edges(
                combined_features,
                core_encode,
                scaled_time_diffs,
                event_trans_edge,
                node_times,
            )
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        y_list = prepare_data_y(combined_features, y_encode)
        trans_size = None
        if self.mode in {"gat_status", "gat_time_decay_status"}:
            _, trans_size = self._encode_transitions(event, sequence)

        return TransformOutput(
            event_features=event_feature_list,
            labels=y_list,
            core_size=core_size,
            output_size=output_size,
            trans_size=trans_size,
            sequence_features=None,
        )

    def fit_transform(self, event: pd.DataFrame) -> TransformOutput:
        """Fit the transformer and return transformed outputs."""
        return self.fit(event).transform(event)

    def _build_sequence_table(self, event: pd.DataFrame) -> pd.DataFrame:
        return event[[self.case_col] + self.seq_cols].groupby(self.case_col).first().reset_index()

    def _encode_labels(self, event: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, int, int]:
        event = event.copy()
        event["event_encoded_col"] = self._event_encoder.transform(event[self.event_col])
        eos_id = self._event_encoder.transform(["EOS"])[0]

        event_values = []
        y_values = []
        for _, group in event.groupby(self.case_col):
            input_events = group["event_encoded_col"].tolist()
            event_values.append(input_events)
            y_seq = input_events[1:] + [eos_id]
            y_values.append(y_seq)

        padded_event_values = pad_sequences(event_values, padding="post", dtype="int64", value=-1)
        padded_y_values = pad_sequences(y_values, padding="post", dtype="int64", value=-1)

        input_size = len(self._event_encoder.classes_)
        output_size = len(self._event_encoder.classes_)

        return (
            padded_event_values[..., np.newaxis],
            padded_y_values[..., np.newaxis],
            input_size,
            output_size,
        )

    def _encode_event_features(self, event: pd.DataFrame) -> np.ndarray:
        combined_features = np.array([])

        if self.cat_event:
            encoded = self._event_cat_encoder.transform(event[self.cat_event])
            if self.cat_mask:
                mask = event[self.cat_event] == "<NO_DESC>"
                expanded_mask = np.column_stack(
                    [
                        mask[col]
                        for col in self.cat_event
                        for _ in range(len(self._event_cat_encoder.categories_[self.cat_event.index(col)]))
                    ]
                )
                encoded[expanded_mask] = -1
            combined_features = encoded

        if self.num_event:
            if self.num_mask:
                scaled = np.full((len(event), len(self.num_event)), -1.0)
                for idx, col in enumerate(self.num_event):
                    valid_mask = event[col] != -1
                    if valid_mask.any() and col in self._event_num_scalers:
                        scaled[valid_mask, idx] = self._event_num_scalers[col].transform(
                            event[col][valid_mask].values.reshape(-1, 1)
                        ).flatten()
            else:
                filled = event[self.num_event].replace(-1, np.nan).fillna(self._event_num_medians)
                scaled = self._event_num_scaler.transform(filled)

            combined_features = scaled if combined_features.size == 0 else np.hstack((combined_features, scaled))

        encoded_sequences = []
        feature_length = combined_features.shape[1] if combined_features.size else 0

        for _, group in event.groupby(self.case_col):
            group_positions = event.index.get_indexer(group.index)
            encoded_sequences.append(combined_features[group_positions])

        return pad_sequences(encoded_sequences, padding="post", dtype="float32", value=-1)

    def _encode_sequence_features(self, sequence: pd.DataFrame) -> np.ndarray:
        combined_features = np.array([])

        if self.cat_seq:
            encoded = self._seq_cat_encoder.transform(sequence[self.cat_seq])
            if self.cat_mask:
                mask = sequence[self.cat_seq] == "<NO_DESC>"
                expanded_mask = np.column_stack(
                    [
                        mask[col]
                        for col in self.cat_seq
                        for _ in range(len(self._seq_cat_encoder.categories_[self.cat_seq.index(col)]))
                    ]
                )
                encoded[expanded_mask] = -1
            combined_features = encoded

        if self.num_seq:
            if self.num_mask:
                scaled = np.full((len(sequence), len(self.num_seq)), -1.0)
                for idx, col in enumerate(self.num_seq):
                    valid_mask = sequence[col] != -1
                    if valid_mask.any() and col in self._seq_num_scalers:
                        scaled[valid_mask, idx] = self._seq_num_scalers[col].transform(
                            sequence[col][valid_mask].values.reshape(-1, 1)
                        ).flatten()
            else:
                filled = sequence[self.num_seq].replace(-1, np.nan).fillna(self._seq_num_medians)
                scaled = self._seq_num_scaler.transform(filled)

            combined_features = scaled if combined_features.size == 0 else np.hstack((combined_features, scaled))

        return combined_features

    def _collect_transitions(self, event: pd.DataFrame, sequence: pd.DataFrame) -> List[str]:
        grouped = event.groupby(self.case_col)
        transitions = []
        for cid in sequence[self.case_col]:
            if cid not in grouped.groups:
                continue
            group = grouped.get_group(cid)
            ev_list = group[self.status_col].tolist()
            transitions.extend([f"{ev_list[i]}→{ev_list[i + 1]}" for i in range(len(ev_list) - 1)])
        return transitions

    def _encode_transitions(self, event: pd.DataFrame, sequence: pd.DataFrame) -> Tuple[List[np.ndarray], int]:
        grouped = event.groupby(self.case_col)
        all_transitions = []
        for cid in sequence[self.case_col]:
            if cid not in grouped.groups:
                all_transitions.append([])
                continue
            group = grouped.get_group(cid)
            ev_list = group[self.status_col].tolist()
            transitions = [f"{ev_list[i]}→{ev_list[i + 1]}" for i in range(len(ev_list) - 1)]
            all_transitions.append(transitions)

        transition_list = []
        for transitions in all_transitions:
            if transitions:
                encoded = [
                    self._transition_encoder.transform([t])[0]
                    if t in self._transition_encoder.classes_
                    else self._transition_encoder.transform(["UNK"])[0]
                    for t in transitions
                ]
                transition_list.append(np.array(encoded, dtype=np.int64))
            else:
                transition_list.append(np.array([], dtype=np.int64))

        return transition_list, len(self._transition_encoder.classes_)

    def _collect_time_diffs(self, event: pd.DataFrame, sequence: pd.DataFrame) -> np.ndarray:
        event = event.copy()
        event[self.time_col] = pd.to_datetime(event[self.time_col], errors="coerce", utc=True)
        event_sorted = event.sort_values(by=[self.case_col, self.time_col])
        grouped = event_sorted.groupby(self.case_col)
        diffs = []
        for _, group in grouped:
            times = group[self.time_col].values
            diffs.extend(np.diff(times) / np.timedelta64(1, "s"))
        return np.array(diffs)

    def _scale_time_diffs(self, event: pd.DataFrame, sequence: pd.DataFrame) -> List[np.ndarray]:
        event = event.copy()
        event[self.time_col] = pd.to_datetime(event[self.time_col], errors="coerce", utc=True)
        event_sorted = event.sort_values(by=[self.case_col, self.time_col])
        grouped = event_sorted.groupby(self.case_col)
        case_time_diffs = {}
        for case_id, group in grouped:
            times = group[self.time_col].values
            diffs = np.diff(times) / np.timedelta64(1, "s")
            case_time_diffs[case_id] = diffs

        time_diffs_list = []
        for i in range(len(sequence)):
            case_id = sequence.iloc[i][self.case_col]
            if case_id in case_time_diffs:
                time_diffs_list.append(case_time_diffs[case_id])
            else:
                time_diffs_list.append(np.array([]))

        non_empty_diffs = [diffs for diffs in time_diffs_list if len(diffs) > 0]
        if len(non_empty_diffs) == 0:
            return [np.array([]) for _ in range(len(sequence))]

        all_diffs = np.concatenate(non_empty_diffs).reshape(-1, 1)
        scaled_all_diffs = self._time_scaler.transform(all_diffs).flatten()

        scaled_time_diffs_list = []
        index = 0
        for diffs in time_diffs_list:
            if len(diffs) > 0:
                scaled_time_diffs_list.append(scaled_all_diffs[index : index + len(diffs)])
                index += len(diffs)
            else:
                scaled_time_diffs_list.append(np.array([]))

        return scaled_time_diffs_list


class PrefixGCNTransformer:
    """Transformer for prefix-based GCN inputs."""
    def __init__(
        self,
        *,
        case_col: str,
        event_col: str,
        time_col: str,
        prefix_size: int,
        cat_event: Optional[List[str]] = None,
        num_event: Optional[List[str]] = None,
        cat_seq: Optional[List[str]] = None,
        num_seq: Optional[List[str]] = None,
    ) -> None:
        self.case_col = case_col
        self.event_col = event_col
        self.time_col = time_col
        self.prefix_size = prefix_size
        self.cat_event = cat_event or []
        self.num_event = num_event or []
        self.cat_seq = cat_seq or []
        self.num_seq = num_seq or []

    def fit(self, event: pd.DataFrame) -> "PrefixGCNTransformer":
        return self

    def transform(self, event: pd.DataFrame) -> TransformOutput:
        event = event.copy()
        event = event[event.groupby(self.case_col)[self.case_col].transform("size") >= self.prefix_size]

        text_encode, event_encode, y_encode, _, output_dim = encode_event_prefix_label(
            event,
            self.event_col,
            self.cat_event,
            self.num_event,
            self.case_col,
            self.prefix_size,
            cat_mask=False,
            num_mask=False,
        )

        sequence = pd.concat(
            [g.iloc[self.prefix_size - 1 :] for _, g in event.groupby(self.case_col, sort=False)],
            ignore_index=True,
        )
        sequence_encode = encode_pad_sequence(sequence, self.cat_seq, self.num_seq)

        scaled_time_diffs = scale_time_differences_fast_fixed(event, sequence, self.time_col, self.case_col)
        event_feature_list = prepare_data_prefix(event_encode, text_encode, scaled_time_diffs)

        return TransformOutput(
            event_features=event_feature_list,
            labels=torch.tensor(y_encode, dtype=torch.long),
            core_size=output_dim,
            output_size=output_dim,
            sequence_features=torch.tensor(sequence_encode, dtype=torch.float),
        )

    def fit_transform(self, event: pd.DataFrame) -> TransformOutput:
        return self.fit(event).transform(event)
