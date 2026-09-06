from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

from ..utils.sequence import pad_sequences


def custom_onehot_encode(data: pd.DataFrame, categorical_columns: List[str], missing_value: str):
    """One-hot encode categorical columns with a custom missing value handling."""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    data_encoded = encoder.fit_transform(data[categorical_columns])

    mask = data[categorical_columns] == missing_value
    expanded_mask = np.column_stack(
        [
            mask[col]
            for col in categorical_columns
            for _ in range(len(encoder.categories_[categorical_columns.index(col)]))
        ]
    )
    data_encoded[expanded_mask] = -1
    return data_encoded, encoder


def onehot_encode(data: pd.DataFrame, categorical_columns: List[str]):
    """Standard one-hot encoding for categorical columns."""
    encoder = OneHotEncoder(sparse_output=False)
    data_encoded = encoder.fit_transform(data[categorical_columns])
    return data_encoded, encoder


def custom_scale_encode(data: pd.DataFrame, numerical_columns: List[str]):
    """Min-max scale numerical columns while preserving -1 as missing."""
    data = data.copy()
    scaler = MinMaxScaler()
    data_scaled = pd.DataFrame(index=data.index, columns=numerical_columns)

    for col in numerical_columns:
        valid_data = data[col].replace(-1, np.nan).dropna()
        if not valid_data.empty:
            scaler.fit(valid_data.values.reshape(-1, 1))
            data_scaled.loc[data[col] != -1, col] = scaler.transform(
                data[col][data[col] != -1].values.reshape(-1, 1)
            ).flatten()

    data_scaled.fillna(-1, inplace=True)
    return data_scaled.values, scaler


def median_scale_encode(data: pd.DataFrame, numerical_columns: List[str]):
    """Scale numerical columns with median imputation for missing values."""
    data = data.copy()
    data[numerical_columns] = data[numerical_columns].replace(-1, np.nan)
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[numerical_columns])
    return data_scaled, scaler


def encode_pad_event(
    event: pd.DataFrame,
    cat_col_event: List[str],
    num_col_event: List[str],
    case_index: str,
    cat_mask: bool = False,
    num_mask: bool = False,
    eos: bool = True,
):
    """Encode event-level features and pad sequences to equal length."""
    combined_features_bulk = np.array([])

    if cat_col_event:
        if cat_mask:
            event_encoded, _ = custom_onehot_encode(event, cat_col_event, "<NO_DESC>")
        else:
            event_encoded, _ = onehot_encode(event, cat_col_event)
        combined_features_bulk = event_encoded

    if num_col_event:
        if num_mask:
            event_scaled, _ = custom_scale_encode(event, num_col_event)
        else:
            event_scaled, _ = median_scale_encode(event, num_col_event)

        if combined_features_bulk.size == 0:
            combined_features_bulk = event_scaled
        else:
            combined_features_bulk = np.hstack((combined_features_bulk, event_scaled))

    encoded_sequences = []
    feature_length = combined_features_bulk.shape[1] if combined_features_bulk.size else 0
    eos_token = np.zeros((1, feature_length)) if eos and feature_length > 0 else None

    for _, group in event.groupby(case_index):
        group_indices = group.index
        group_combined_features = combined_features_bulk[group_indices]

        if eos and eos_token is not None:
            group_combined_features = np.vstack([group_combined_features, eos_token])
        encoded_sequences.append(group_combined_features)

    padded_sequences = pad_sequences(
        encoded_sequences, padding="post", dtype="float32", value=-1
    )
    return padded_sequences


def encode_pad_sequence(
    sequence: pd.DataFrame,
    cat_col_seq: List[str],
    num_col_seq: List[str],
    cat_mask: bool = False,
    num_mask: bool = False,
):
    """Encode sequence-level features into a fixed-size representation."""
    combined_features_bulk = np.array([])

    if cat_col_seq:
        if cat_mask:
            sequence_encoded, _ = custom_onehot_encode(sequence, cat_col_seq, "<NO_DESC>")
        else:
            sequence_encoded, _ = onehot_encode(sequence, cat_col_seq)
        combined_features_bulk = sequence_encoded

    if num_col_seq:
        if num_mask:
            sequence_scaled, _ = custom_scale_encode(sequence, num_col_seq)
        else:
            sequence_scaled, _ = median_scale_encode(sequence, num_col_seq)
        if combined_features_bulk.size == 0:
            combined_features_bulk = sequence_scaled
        else:
            combined_features_bulk = np.hstack((combined_features_bulk, sequence_scaled))

    return combined_features_bulk


def encode_label_event(event: pd.DataFrame, core_event: str, case_index: str):
    """Label-encode events and create shifted target sequences with EOS."""
    le_event = LabelEncoder()
    all_events = event[core_event].tolist() + ["EOS"]
    le_event.fit(all_events)

    event = event.copy()
    event["event_encoded_col"] = le_event.transform(event[core_event])
    eos_id = le_event.transform(["EOS"])[0]

    event_values = []
    y_values = []

    for _, group in event.groupby(case_index):
        input_events = group["event_encoded_col"].tolist()
        event_values.append(input_events)
        y_seq = input_events[1:] + [eos_id]
        y_values.append(y_seq)

    padded_event_values = pad_sequences(event_values, padding="post", dtype="int64", value=-1)
    padded_y_values = pad_sequences(y_values, padding="post", dtype="int64", value=-1)

    input_size = len(le_event.classes_)
    output_size = len(le_event.classes_)

    return (
        padded_event_values[..., np.newaxis],
        padded_y_values[..., np.newaxis],
        input_size,
        output_size,
        le_event,
    )


def _parse_mixed_datetime(event: pd.DataFrame, start_time_col: str, case_index: str) -> pd.Series:
    parsed = pd.to_datetime(event[start_time_col], errors="coerce", utc=True, format="mixed")
    if parsed.isna().any():
        parsed_alt = pd.to_datetime(event[start_time_col], errors="coerce", utc=True)
        parsed = parsed.fillna(parsed_alt)
    parsed = event.assign(**{start_time_col: parsed})
    parsed[start_time_col] = parsed.groupby(case_index)[start_time_col].transform(
        lambda s: s.ffill().bfill()
    )
    return parsed[start_time_col].fillna(pd.Timestamp(0, tz="UTC"))


def node_time_list(event: pd.DataFrame, start_time_col: str, case_index: str):
    """Compute normalized per-sequence time deltas."""
    event = event.copy()
    event[start_time_col] = _parse_mixed_datetime(event, start_time_col, case_index)
    event["unix_time"] = event[start_time_col].astype("int64") // 1_000_000_000

    all_time_list = []
    for _, group in event.groupby(case_index):
        delta = group["unix_time"].values[-1] - group["unix_time"].values[:-1]
        if len(delta) > 0:
            norm_delta = delta / (delta.max() + 1e-8)
        else:
            norm_delta = delta
        all_time_list.append(norm_delta)
    return all_time_list


def event_transition_edge(event: pd.DataFrame, sequence: pd.DataFrame, status: str, case_index: str):
    """Compute label-encoded transition edge types between events."""
    grouped = event.groupby(case_index)
    all_transitions = []
    all_transition_strings = []

    for cid in sequence[case_index]:
        if cid not in grouped.groups:
            all_transitions.append([])
            continue

        group = grouped.get_group(cid)
        ev_list = group[status].tolist()

        transitions = []
        for i in range(len(ev_list) - 1):
            edge = ev_list[i] + "→" + ev_list[i + 1]
            transitions.append(edge)
            all_transition_strings.append(edge)

        all_transitions.append(transitions)

    le = LabelEncoder()
    if all_transition_strings:
        le.fit(all_transition_strings)
    else:
        le.fit(["EMPTY"])

    event_transition_list = []
    for transitions in all_transitions:
        if transitions:
            encoded = le.transform(transitions)
            event_transition_list.append(np.array(encoded, dtype=np.int64))
        else:
            event_transition_list.append(np.array([], dtype=np.int64))

    trans_size = len(le.classes_) + 1
    return event_transition_list, le, trans_size


def scale_time_differences_fast_fixed(
    event: pd.DataFrame,
    sequence: pd.DataFrame,
    start_time_col: str,
    case_index: str,
):
    """Scale time differences per sequence while preserving original order."""
    event = event.copy()
    event[start_time_col] = _parse_mixed_datetime(event, start_time_col, case_index)
    event_sorted = event.sort_values(by=[case_index, start_time_col])

    grouped = event_sorted.groupby(case_index)
    case_time_diffs = {}
    for case_id, group in grouped:
        times = group[start_time_col].values
        diffs = np.diff(times) / np.timedelta64(1, "s")
        case_time_diffs[case_id] = diffs

    time_diffs_list = []
    for i in range(len(sequence)):
        case_id = sequence.iloc[i][case_index]
        if case_id in case_time_diffs:
            time_diffs_list.append(case_time_diffs[case_id])
        else:
            time_diffs_list.append(np.array([]))

    non_empty_diffs = [diffs for diffs in time_diffs_list if len(diffs) > 0]
    if len(non_empty_diffs) == 0:
        return [np.array([]) for _ in range(len(sequence))]

    all_diffs = np.concatenate(non_empty_diffs).reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled_all_diffs = scaler.fit_transform(all_diffs).flatten()

    scaled_time_diffs_list = []
    index = 0
    for diffs in time_diffs_list:
        if len(diffs) > 0:
            scaled_time_diffs_list.append(scaled_all_diffs[index : index + len(diffs)])
            index += len(diffs)
        else:
            scaled_time_diffs_list.append(np.array([]))

    return scaled_time_diffs_list


def encode_event_prefix_label(
    event: pd.DataFrame,
    core_event: str,
    cat_col_event: List[str],
    num_col_event: List[str],
    case_index: str,
    prefix_size: int,
    cat_mask: bool = False,
    num_mask: bool = False,
):
    """Encode prefix subsequences and next-event labels for prefix tasks."""
    event_copy = event[core_event].copy().to_frame()
    event_copy.loc[len(event_copy)] = "EOS"

    label_encoder = LabelEncoder()
    event_labels = label_encoder.fit_transform(event_copy[core_event])

    event_copy = event_copy[:-1]
    event_encoded = event_labels[:-1].reshape(-1, 1)

    eos_encoding = event_labels[-1]
    y_labels = event_labels[:-1]

    combined_features_bulk = np.array([])

    if cat_col_event:
        if cat_mask:
            event_cat_encoded, _ = custom_onehot_encode(event, cat_col_event, "<NO_DESC>")
        else:
            event_cat_encoded, _ = onehot_encode(event, cat_col_event)
        combined_features_bulk = event_cat_encoded

    if num_col_event:
        if num_mask:
            event_scaled, _ = custom_scale_encode(event, num_col_event)
        else:
            event_scaled, _ = median_scale_encode(event, num_col_event)

        if combined_features_bulk.size == 0:
            combined_features_bulk = event_scaled
        else:
            combined_features_bulk = np.hstack((combined_features_bulk, event_scaled))

    event_values = []
    encoded_subsequences = []
    y_values = []

    for _, group in event.groupby(case_index):
        group_indices = group.index
        group_features = combined_features_bulk[group_indices]
        input_events = event_encoded[group_indices]
        predict_events = y_labels[group_indices]

        sequence_length = len(group_features)
        for i in range(sequence_length - prefix_size + 1):
            subseq = group_features[i : i + prefix_size]
            encoded_subsequences.append(subseq)
            event_values.append(input_events[i : i + prefix_size])

            if i + prefix_size < sequence_length:
                y_values.append(predict_events[i + prefix_size])
            else:
                y_values.append(eos_encoding)

    event_values = np.array(event_values, dtype=np.int64)
    encoded_subsequences = np.array(encoded_subsequences, dtype=np.float32)
    y_values = np.array(y_values, dtype=np.int64)

    input_event_size = len(label_encoder.classes_) - 1
    output_size = len(label_encoder.classes_)

    return event_values, encoded_subsequences, y_values, input_event_size, output_size


def encode_event_prefix(
    event: pd.DataFrame,
    core_event: str,
    cat_col_event: List[str],
    num_col_event: List[str],
    case_index: str,
    prefix_size: int,
    cat_mask: bool = False,
    num_mask: bool = False,
):
    """Encode prefix subsequences for next-event prediction without core labels."""
    event_copy = event[core_event].copy().to_frame()
    event_copy.loc[len(event_copy)] = "EOS"

    event_encoded, encoder = onehot_encode(event_copy, [core_event])
    categories = encoder.categories_[0]

    label_encoder = LabelEncoder()
    label_encoder.fit(categories)
    y_labels = label_encoder.transform(event_copy[core_event])

    event_encoded = event_encoded[:-1]
    event_copy = event_copy[:-1]

    eos_encoding = y_labels[-1]
    y_labels = y_labels[:-1]

    combined_features_bulk = np.array([])

    if cat_col_event:
        if cat_mask:
            event_cat_encoded, _ = custom_onehot_encode(event, cat_col_event, "<NO_DESC>")
        else:
            event_cat_encoded, _ = onehot_encode(event, cat_col_event)
        combined_features_bulk = event_cat_encoded

    if num_col_event:
        if num_mask:
            event_scaled, _ = custom_scale_encode(event, num_col_event)
        else:
            event_scaled, _ = median_scale_encode(event, num_col_event)
        if combined_features_bulk.size == 0:
            combined_features_bulk = event_scaled
        else:
            combined_features_bulk = np.hstack((combined_features_bulk, event_scaled))

    combined_features_bulk = np.hstack((event_encoded, combined_features_bulk))

    encoded_subsequences = []
    y_values = []

    for _, group in event.groupby(case_index):
        group_indices = group.index
        group_features = combined_features_bulk[group_indices]
        predict_events = y_labels[group_indices]

        sequence_length = len(group_features)
        for i in range(sequence_length - prefix_size + 1):
            subseq = group_features[i : i + prefix_size]
            encoded_subsequences.append(subseq)

            if i + prefix_size < sequence_length:
                y_values.append(predict_events[i + prefix_size])
            else:
                y_values.append(eos_encoding)

    encoded_subsequences = np.array(encoded_subsequences, dtype=np.float32)
    y_values = np.array(y_values, dtype=np.int64)

    output_size = len(label_encoder.classes_)
    return encoded_subsequences, y_values, output_size


def length_stratified_split(event_feature_list, test_size: float = 0.2, n_bins: int = 5):
    """Split sequences into train/test with length-stratified bins."""
    sequence_lengths = [data.x.shape[0] for data in event_feature_list]
    min_len, max_len = min(sequence_lengths), max(sequence_lengths)
    bin_edges = np.linspace(min_len, max_len + 1, n_bins + 1)
    bins = np.digitize(sequence_lengths, bin_edges) - 1
    bins = np.clip(bins, 0, n_bins - 1)

    train_indices = []
    test_indices = []

    for bin_id in range(n_bins):
        bin_indices = [i for i, b in enumerate(bins) if b == bin_id]
        if len(bin_indices) == 0:
            continue

        n_test = max(1, int(len(bin_indices) * test_size))
        n_train = len(bin_indices) - n_test

        bin_indices_with_lengths = [(i, sequence_lengths[i]) for i in bin_indices]
        bin_indices_with_lengths.sort(key=lambda x: x[1])

        train_indices.extend([idx for idx, _ in bin_indices_with_lengths[:n_train]])
        test_indices.extend([idx for idx, _ in bin_indices_with_lengths[n_train:]])

    return train_indices, test_indices
