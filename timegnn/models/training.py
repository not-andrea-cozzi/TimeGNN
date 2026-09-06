"""Shared training and evaluation loops for next-event prediction models."""
from __future__ import annotations

import torch


def train_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch for next-event prediction models.

    Works with any model that takes batched event graphs and produces
    per-node logits.  Labels are expected to be padded with -1.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total_tokens = 0

    for event_data, labels in loader:
        event_data = event_data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(event_data)

        output = output.view(-1, output.size(-1))
        labels = labels.view(-1)

        mask = labels != -1
        labels = labels[mask]

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        pred = output.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total_tokens += labels.size(0)

    accuracy = correct / total_tokens if total_tokens else 0.0
    avg_loss = total_loss / total_tokens if total_tokens else 0.0
    return avg_loss, accuracy


def evaluate_epoch(model, loader, criterion, device):
    """Evaluate a next-event prediction model for one epoch.

    Returns:
        Tuple of (loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total_tokens = 0

    with torch.no_grad():
        for event_data, labels in loader:
            event_data = event_data.to(device)
            labels = labels.to(device)

            output = model(event_data)

            output = output.view(-1, output.size(-1))
            labels = labels.view(-1)

            mask = labels != -1
            labels = labels[mask]

            loss = criterion(output, labels)
            total_loss += loss.item() * labels.size(0)

            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total_tokens += labels.size(0)

    accuracy = correct / total_tokens if total_tokens else 0.0
    avg_loss = total_loss / total_tokens if total_tokens else 0.0
    return avg_loss, accuracy
