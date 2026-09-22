"""
CheckAltMates.py

Diagnostica: la metrica di 'move accuracy' usata da EvaluateModels.py /
TrainMain.py richiede che la mossa predetta coincida ESATTAMENTE con
quella etichettata come 'best_move' durante la costruzione del dataset
(la prima trovata da Stockfish nella PV, si veda Buildexternalholdout.py
riga ~394 circa). Se in una posizione esistono PIU' mosse che portano
tutte al matto nello stesso numero n di semimosse, la metrica conta come
'sbagliate' anche le alternative corrette, sottostimando l'accuratezza
reale del modello.

Questo script:
  1. Carica il modello 'basic' gia' allenato.
  2. Gira sulle posizioni dell'holdout esterno locale (quello con FEN).
  3. Per ogni posizione dove la predizione del modello NON coincide con
     l'etichetta, verifica con Stockfish se la mossa predetta porta
     comunque a matto nello stesso n (stesso criterio usato in fase di
     costruzione dataset: analbilla dopo la mossa, controlla che il lato
     che deve muovere sia sotto matto in esattamente n-1 semimosse).

Uso:
    python CheckAltMates.py --limit 100
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import chess
import chess.engine
import torch

from DatasetPipeline.Model.ChessConstants import (
    NUM_EVENT_FEATURES,
    NUM_EVENT_ID_CATEGORIES,
    MOVE_VOCAB_SIZE,
    NUM_EDGE_TYPES,
)
from DatasetPipeline.Model.PositionGraphSchema import decode_move
from TrainPipeline.Shard.ShardDataset import ShardedGraphDataset
from timegnn.data.pyg import custom_collate_graph
from timegnn.models.gat_basic import DualGATModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("check_alt_mates")


def load_basic_model(checkpoint_path: str, norm_kind: str, device: str) -> DualGATModel:
    model = DualGATModel(
        num_event_features=NUM_EVENT_FEATURES,
        num_embedding_features=NUM_EVENT_ID_CATEGORIES,
        embedding_dims=64,
        gat_hidden_dim_event=32,
        gat_hidden_dim_embed=128,
        gat_hidden_dim_concat=128,
        output_dim=MOVE_VOCAB_SIZE,
        num_heads=4,
        num_layers=3,
        dropout=0.0,
        use_batch_norm=False,
        activation="elu",
        norm_kind=norm_kind,
        edge_dim=NUM_EDGE_TYPES,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
    model.eval()
    return model


def sparse_legal_argmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    masked = logits.masked_fill(legal_mask == 0, float("-inf"))
    return masked.argmax(dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--holdout_dir", default="Dataset/ExternalHoldout/holdout_clean")
    parser.add_argument("--checkpoint", default="Models/basic_best.pt")
    parser.add_argument("--stockfish_path", default="stockfish")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--norm_kind", default="graph_norm")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    logger.info(f"Caricamento modello da {args.checkpoint} (norm_kind={args.norm_kind})...")
    model = load_basic_model(args.checkpoint, args.norm_kind, device)

    logger.info(f"Caricamento holdout da {args.holdout_dir}...")
    ds = ShardedGraphDataset(args.holdout_dir, shuffle=False)

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    limit = chess.engine.Limit(depth=12, time=0.2)

    n_total = 0
    n_correct_strict = 0
    n_correct_with_alt = 0
    n_correct_top3 = 0
    n_correct_top5 = 0
    n_no_fen = 0
    mismatches_checked = 0

    try:
        for i, (data, label) in enumerate(ds):
            if i >= args.limit:
                break
            fen = getattr(data, "fen", None)
            if fen is None:
                n_no_fen += 1
                continue

            n_total += 1
            batch = custom_collate_graph([(data, label)])
            batch_event, labels_t = batch
            batch_event = batch_event.to(device)
            with torch.no_grad():
                logits = model(batch_event)
            legal_mask = batch_event.legal_move_mask
            masked_logits = logits.masked_fill(legal_mask == 0, float("-inf"))
            pred_idx = int(masked_logits.argmax(dim=1).item())
            true_idx = int(label)

            topk_vals = min(5, int((legal_mask[0] > 0).sum().item()))
            if topk_vals > 0:
                top_indices = masked_logits[0].topk(topk_vals).indices.tolist()
                if true_idx in top_indices[:3]:
                    n_correct_top3 += 1
                if true_idx in top_indices[:5]:
                    n_correct_top5 += 1

            if pred_idx == true_idx:
                n_correct_strict += 1
                n_correct_with_alt += 1
                continue

            # Mismatch: verifica se la mossa predetta e' comunque un matto valido nello stesso n
            mismatches_checked += 1
            mate_n = int(getattr(data, "position_mate_n", 0))
            board = chess.Board(fen)

            from_sq, to_sq, promo = decode_move(pred_idx)
            try:
                pred_move = chess.Move(from_sq, to_sq, promotion=promo)
                if pred_move not in board.legal_moves:
                    continue  # mossa illegale, sicuramente sbagliata
            except Exception:
                continue

            board.push(pred_move)
            try:
                info = engine.analyse(board, limit)
            except Exception as e:
                logger.warning(f"[{i}] errore analisi stockfish: {e}")
                continue
            score = info.get("score")
            if score is None:
                continue
            rel = score.relative
            if rel.is_mate() and rel.mate() == -(mate_n - 1):
                n_correct_with_alt += 1
                logger.info(f"[{i}] mossa alternativa valida trovata: mate_n={mate_n}")

            if (i + 1) % 20 == 0:
                logger.info(f"[{i + 1}] processate finora...")
    finally:
        engine.quit()

    logger.info("=" * 60)
    logger.info(f"Totale posizioni valutate: {n_total} (saltate per FEN mancante: {n_no_fen})")
    logger.info(f"Accuratezza stretta (solo match esatto): {n_correct_strict}/{n_total} = {n_correct_strict/n_total*100:.2f}%")
    logger.info(f"Accuratezza con alternative valide: {n_correct_with_alt}/{n_total} = {n_correct_with_alt/n_total*100:.2f}%")
    logger.info(f"Accuratezza top-3: {n_correct_top3}/{n_total} = {n_correct_top3/n_total*100:.2f}%")
    logger.info(f"Accuratezza top-5: {n_correct_top5}/{n_total} = {n_correct_top5/n_total*100:.2f}%")
    logger.info(f"Mismatch controllati con Stockfish: {mismatches_checked}")
    logger.info(f"Di cui erano in realta' mosse alternative valide: {n_correct_with_alt - n_correct_strict}")


if __name__ == "__main__":
    main()
