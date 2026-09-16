# TimeGNN

## Overview

TimeGNN is a research-oriented deep learning project for **chess move prediction** using **Graph Neural Networks (GNNs)**. The core idea is to model a chess position as a graph and apply **Graph Attention Networks (GATs)** that incorporate **time-decay** into the attention mechanism. This allows the model to weigh historical events (e.g., recent moves or time pressure) differently from older ones when predicting the next move.

The project includes a complete pipeline for dataset creation, training, evaluation, and inference, with a strong focus on handling large-scale data through sharded storage and memory-efficient processing.

> **Note:** The repository is under active development. The primary development branch is `dataset_creation`, which contains the most recent code and configuration files.

## Key Features

- **Time-aware GAT models** – A custom `TimeAwareGATConv` layer that applies exponential decay to attention logits based on the time difference between events.
- **Dual-path architecture** – Separate GAT branches process event features and embedding features, which are then concatenated for the final prediction.
- **Sharded dataset pipeline** – Datasets are built, cleaned, and stored as shards (`.pt` files) with a `manifest.json` index, enabling streaming and low-memory training.
- **Flexible configuration** – YAML files control dataset building, training hyperparameters, and evaluation settings.
- **Training utilities** – Includes tuning of class weights, warmup schedules, early stopping, and atomic checkpoint saving.
- **Evaluation scripts** – Dedicated scripts for loading checkpoints and evaluating models on test shards.

## Architecture

The project is organised into several Python packages and scripts:

| Directory / File | Purpose |
|------------------|---------|
| `DatasetPipeline/` | Builds chess game datasets, converts positions to PyG graphs, and writes sharded outputs. |
| `TrainPipeline/` | Contains the training loop, dataset cleaning, sharding, tuning steps, and state management. |
| `timegnn/` | Core model definitions (`gat_basic.py`, `gat_time_decay.py`), data utilities (`pyg.py`), and training helpers. |
| `Common/` | Shared utilities such as evaluators, plotting, and sparse legal-move handling. |
| `Yaml/` | Configuration files for dataset building (`build_shards.yaml`, `dataset_main.yaml`), training (`train_main.yaml`), and evaluation (`evaluate_models.yaml`). |
| `TrainMain.py` | Main training entry point: tuning, training loop, checkpointing, and shard-aware data loading. |
| `DatasetMain.py` | Main dataset entry point: builds games/puzzles, computes time statistics, and finalises splits into shards. |
| `EvaluateModels.py` | Loads trained checkpoints and evaluates them on test shards. |
| `TrainTest.py` | A smoke-test script that runs a minimal version of the training pipeline to validate functionality. |
| `requirements.txt` | Python dependencies. |

## Requirements

- Python 3.9+
- PyTorch (with CUDA support recommended)
- PyTorch Geometric
- `python-chess`
- `pandas`, `tqdm`, `scikit-learn`, `seaborn`, `pyyaml`

Install dependencies via:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/not-andrea-cozzi/TimeGNN.git
   cd TimeGNN
   ```
2. (Optional) Create and activate a virtual environment.
3. Install the required packages (see above).

## Usage

### 1. Build the dataset

Prepare the raw data (PGN files or similar) and adjust `Yaml/dataset_main.yaml` to point to your sources. Then run:

```bash
python DatasetMain.py --config Yaml/dataset_main.yaml
```

This will produce sharded datasets (e.g., `Dataset/Train/train/shard_*.pt` with a `manifest.json`).

### 2. Train a model

Configure training parameters in `Yaml/train_main.yaml`, then execute:

```bash
python TrainMain.py --config Yaml/train_main.yaml
```

The script will automatically:
- Run a tuning step for class weights and warmup schedule.
- Clean and (if needed) re-shard the dataset.
- Train the selected model (basic GAT or time-aware GAT).
- Save checkpoints atomically.

### 3. Evaluate models

Set the checkpoint paths in `Yaml/evaluate_models.yaml` and run:

```bash
python EvaluateModels.py --config Yaml/evaluate_models.yaml
```

The evaluation script loads both a basic GAT checkpoint and a time-aware GAT checkpoint for comparison.

### 4. Quick smoke test

To verify that the training pipeline works without a full dataset, run:

```bash
python TrainTest.py
```

This creates a tiny synthetic shard, runs the tuning step, and performs a few training iterations.

## Configuration

All YAML files are heavily commented and follow a consistent structure. Key sections include:

- `dataset`: paths, split ratios, shard size.
- `model_params`: dimensions, dropout, activation, number of layers.
- `training`: batch size, learning rate, epochs, early stopping patience.
- `evaluation`: test data path, checkpoint paths.

## Models

### `DualGATModel` (basic)

A standard dual-path GAT without time decay. Suitable for baseline comparisons.

### `DualGATTimeAwareModel`

Extends the basic model with `TimeAwareGATConv` layers. Each edge carries a `time_diff` attribute; the attention logit is multiplied by `exp(-λ * time_diff)`, where `λ` is a learnable or fixed decay parameter. This allows the model to focus on recent events when predicting the next move.

## For AI Agents

If you are an AI agent interacting with this repository, note the following:

- **Entry points** – `TrainMain.py` and `DatasetMain.py` are the primary CLI interfaces. They accept a `--config` argument pointing to a YAML file.
- **Data format** – Datasets are stored as shards (`shard_NNNNN.pt`) containing lists of PyTorch Geometric `Data` objects. A `manifest.json` in each split directory describes the number of shards and total samples.
- **State management** – The pipeline uses a `PipelineState` class to track completed steps (e.g., `tuning`, `clean`, `shard`). This state is persisted so that re-running a script skips already completed steps unless forced.
- **Model loading** – Checkpoints are saved as dictionaries containing `model_state_dict`, `optimizer_state_dict`, and metadata. Use `load_model` in `EvaluateModels.py` as a reference.
- **Extensibility** – To add a new model, create a class in `timegnn/models/` and register it in the training/evaluation configs.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request on the `dataset_creation` branch. Ensure that any changes maintain compatibility with the sharded data format and the pipeline state mechanism.

## License

No license file is currently present in the repository. Please contact the author for licensing information.

---

*This README was generated by analysing the repository structure and source code. For the most up-to-date information, always refer to the code itself.*