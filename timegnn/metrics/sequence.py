from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch


def predict(model, loader, device):
    """Predict token labels for all sequences in a loader."""
    model.eval()
    all_preds = []
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for event_data, labels in loader:
            event_data = event_data.to(device)
            labels = labels.to(device)

            output = model(event_data)
            output = output.view(-1, output.size(-1))
            labels = labels.view(-1)

            mask = labels != -1
            labels = labels[mask]

            preds = output.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_outputs.append(output.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_outputs = torch.cat(all_outputs)
    return all_preds, all_labels, all_outputs


def top_k_accuracy(output, target, k: int = 3) -> float:
    """Compute top-k accuracy for flattened outputs."""
    top_k_preds = output.topk(k, dim=1).indices
    target = target.view(-1, 1)
    correct = top_k_preds.eq(target).sum().item()
    total = target.size(0)
    return correct / total if total else 0.0


def predict_per_sequence(model, loader, device):
    """Return per-sequence predictions and labels without padding."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for event_data, labels in loader:
            event_data = event_data.to(device)
            labels = labels.to(device)

            output = model(event_data)

            valid_lengths = (labels != -1).sum(dim=1).tolist()
            preds_flat = output.argmax(dim=1)

            idx = 0
            for i, length in enumerate(valid_lengths):
                if length == 0:
                    all_preds.append([])
                    all_labels.append([])
                    continue

                non_padded_labels = labels[i][:length]
                preds_seq = preds_flat[idx : idx + length].cpu().tolist()
                labels_seq = non_padded_labels.cpu().tolist()

                all_preds.append(preds_seq)
                all_labels.append(labels_seq)
                idx += length

    return all_preds, all_labels


def average_bleu_score(preds_seq, labels_seq, max_n: int = 4) -> float:
    """Compute average BLEU score with smoothing for sequence predictions."""
    if len(preds_seq) != len(labels_seq):
        raise ValueError("Predictions and labels must have same length")

    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("nltk is required for BLEU scoring") from exc

    bleu_scores = []
    smoother = SmoothingFunction().method1

    for pred, label in zip(preds_seq, labels_seq):
        if len(label) == 0 or len(pred) == 0:
            continue

        actual_n = min(max_n, len(pred), len(label))
        weights = tuple([1 / actual_n] * actual_n)

        try:
            score = sentence_bleu([label], pred, weights=weights, smoothing_function=smoother)
            bleu_scores.append(score)
        except Exception:
            continue

    return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0


def compute_dls_and_exact_match(pred_seqs, true_seqs):
    """Compute Damerau-Levenshtein similarity and exact match accuracy."""
    try:
        from pyxdameraulevenshtein import damerau_levenshtein_distance as dld
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pyxdameraulevenshtein is required for DLS") from exc

    if len(pred_seqs) != len(true_seqs):
        raise ValueError("pred_seqs and true_seqs must have equal length")

    dls_scores = []
    exact_match_count = 0
    total_sequences = 0

    for pred, true in zip(pred_seqs, true_seqs):
        if not pred and not true:
            continue

        total_sequences += 1

        if not pred or not true:
            similarity = 0.0
        else:
            dist = dld(pred, true)
            max_len = max(len(pred), len(true))
            similarity = 1 - dist / max_len

        dls_scores.append(similarity)

        if len(pred) == len(true) and all(p == t for p, t in zip(pred, true)):
            exact_match_count += 1

    if total_sequences == 0:
        return 0.0, 0.0

    avg_dls = sum(dls_scores) / total_sequences
    exact_match_acc = exact_match_count / total_sequences

    return avg_dls, exact_match_acc


def sequence_level_top_k_accuracy(model, loader, device, k: int = 3) -> float:
    """Compute sequence-level top-k accuracy (all positions correct in top-k)."""
    model.eval()
    total_correct_sequences = 0
    total_sequences = 0

    with torch.no_grad():
        for event_data, labels in loader:
            event_data = event_data.to(device)
            labels = labels.to(device)

            output = model(event_data)
            top_k_preds = output.topk(k, dim=1).indices

            valid_lengths = (labels != -1).sum(dim=1).tolist()

            ptr = 0
            for i, length in enumerate(valid_lengths):
                if length == 0:
                    continue

                seq_labels = labels[i][:length]
                seq_top_k = top_k_preds[ptr : ptr + length]

                correct = True
                for pos in range(length):
                    if seq_labels[pos] not in seq_top_k[pos]:
                        correct = False
                        break

                if correct:
                    total_correct_sequences += 1
                total_sequences += 1
                ptr += length

    return total_correct_sequences / total_sequences if total_sequences > 0 else 0.0


def analyze_sequence_errors(model, loader, device, k: int = 3):
    """Analyze prediction errors by position and type."""
    preds_seq, labels_seq = predict_per_sequence(model, loader, device)

    error_positions = []
    error_types = defaultdict(int)
    seq_lengths = []

    for pred, label in zip(preds_seq, labels_seq):
        if not pred or not label:
            continue

        seq_lengths.append(len(label))

        for pos in range(len(label)):
            true_label = label[pos]
            predicted_label = pred[pos]

            if predicted_label != true_label:
                error_positions.append(pos)
                error_types[(predicted_label, true_label)] += 1

    pos_errors = np.bincount(error_positions) if error_positions else np.array([])

    seq_length_stats = {
        "min": min(seq_lengths) if seq_lengths else 0,
        "max": max(seq_lengths) if seq_lengths else 0,
        "mean": np.mean(seq_lengths) if seq_lengths else 0,
        "median": np.median(seq_lengths) if seq_lengths else 0,
    }

    return pos_errors, dict(error_types), seq_length_stats


def predict_per_sequence_with_probs(model, loader, device, k: int = 1):
    """Return per-sequence top-k predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []
    all_topk = []

    with torch.no_grad():
        for event_data, labels in loader:
            event_data = event_data.to(device)
            labels = labels.to(device)

            output = model(event_data)
            probs = torch.softmax(output, dim=1)
            _, topk_indices = torch.topk(probs, k, dim=1)

            valid_lengths = (labels != -1).sum(dim=1).tolist()

            idx = 0
            for i, length in enumerate(valid_lengths):
                if length == 0:
                    all_preds.append([])
                    all_labels.append([])
                    all_topk.append([])
                    continue

                non_padded_labels = labels[i][:length]

                seq_topk = topk_indices[idx : idx + length].cpu().tolist()
                seq_labels = non_padded_labels.cpu().tolist()

                all_preds.append([x[0] for x in seq_topk])
                all_labels.append(seq_labels)
                all_topk.append(seq_topk)
                idx += length

    return all_preds, all_labels, all_topk


def sequence_level_top_k_analysis(preds_topk, labels):
    """Analyze sequence-level top-k errors and common mistakes."""
    total_sequences = len(labels)
    correct_sequences = 0
    error_stats = {"wrong_positions": [], "common_errors": defaultdict(int)}

    for seq_topk, seq_labels in zip(preds_topk, labels):
        if not seq_labels:
            continue

        sequence_correct = True
        for pos, (topk_preds, true_label) in enumerate(zip(seq_topk, seq_labels)):
            if true_label not in topk_preds:
                sequence_correct = False
                error_stats["wrong_positions"].append(pos)
                error_stats["common_errors"][(topk_preds[0], true_label)] += 1

        if sequence_correct:
            correct_sequences += 1

    accuracy = correct_sequences / total_sequences if total_sequences > 0 else 0.0

    pos_errors = (
        np.bincount(error_stats["wrong_positions"]) if error_stats["wrong_positions"] else []
    )
    error_stats["position_errors"] = {pos: count for pos, count in enumerate(pos_errors)}

    error_stats["top_errors"] = dict(
        sorted(error_stats["common_errors"].items(), key=lambda x: x[1], reverse=True)[:5]
    )

    return accuracy, error_stats


def show_error_sequences(preds_topk, labels, num: int = 3):
    """Print a sample of sequences with top-k prediction errors."""
    for i, (seq_topk, seq_labels) in enumerate(zip(preds_topk, labels)):
        if any(true not in topk for topk, true in zip(seq_topk, seq_labels)):
            print(f"\nError Sequence #{i+1}:")
            print(f"True: {seq_labels}")
            print(f"Pred: {[topk[0] for topk in seq_topk]}")
            print("Mismatches:")
            for pos, (topk, true) in enumerate(zip(seq_topk, seq_labels)):
                if true not in topk:
                    print(f"Pos {pos}: True {true} not in {topk}")
            num -= 1
            if num == 0:
                break
