import io
import re
import json
from collections import defaultdict
from typing import Optional, Dict, Generator

import chess.pgn
import zstandard as zstd
from tqdm import tqdm


class TimeStatsBuilder:
    """
    Calcola il tempo medio di riflessione per mossa bucketizzato per rating,
    estratto dai commenti %clk nei file PGN Lichess (.zst).
    """

    CLK_RE = re.compile(r'\[%clk (\d+):(\d+):(\d+(?:\.\d+)?)\]')

    def __init__(
        self,
        zst_path: str,
        max_games: int = 50_000,
        bucket_size: int = 100,
        min_base_time: int = 180,
        max_spent_threshold: float = 300.0,
    ):
        self.zst_path = zst_path
        self.max_games = max_games
        self.bucket_size = bucket_size
        self.min_base_time = min_base_time
        self.max_spent_threshold = max_spent_threshold

    @classmethod
    def _parse_clock(cls, comment: str) -> Optional[float]:
        m = cls.CLK_RE.search(comment or "")
        if not m:
            return None
        h, mi, s = m.groups()
        return int(h) * 3600 + int(mi) * 60 + float(s)

    @staticmethod
    def _parse_time_control(time_control: str) -> tuple[float, float]:
        if not time_control or time_control == "-":
            return 0.0, 0.0
        m = re.match(r'^(\d+)\+(\d+)$', time_control)
        if m:
            return float(m.group(1)), float(m.group(2))
        m_base = re.match(r'^(\d+)$', time_control)
        if m_base:
            return float(m_base.group(1)), 0.0
        return 0.0, 0.0

    def _stream_games(self) -> Generator[chess.pgn.Game, None, None]:
        dctx = zstd.ZstdDecompressor()
        with open(self.zst_path, "rb") as f, dctx.stream_reader(f) as r:
            text = io.TextIOWrapper(r, encoding="utf-8")
            n = 0
            while True:
                if self.max_games and n >= self.max_games:
                    break
                g = chess.pgn.read_game(text)
                if g is None:
                    break
                n += 1
                yield g

    def build(self) -> Dict[int, float]:
        bucket_sum: Dict[int, float] = defaultdict(float)
        bucket_count: Dict[int, int] = defaultdict(int)

        # Barra di avanzamento sulle partite analizzate
        pbar = tqdm(
            self._stream_games(),
            total=self.max_games,
            desc="Analisi Time Stats",
            unit="partita",
            dynamic_ncols=True
        )

        for game in pbar:
            headers = game.headers
            base_time, increment = self._parse_time_control(headers.get("TimeControl", ""))
            
            if base_time < self.min_base_time:
                continue

            try:
                white_elo = int(headers.get("WhiteElo", 0))
                black_elo = int(headers.get("BlackElo", 0))
            except ValueError:
                continue

            if white_elo <= 0 or black_elo <= 0:
                continue

            prev_clock = {
                chess.WHITE: base_time,
                chess.BLACK: base_time
            }

            node = game
            while node.variations:
                nxt = node.variation(0)
                mover_color = node.board().turn
                clk = self._parse_clock(nxt.comment)

                if clk is not None:
                    prev = prev_clock[mover_color]
                    spent = prev - clk + increment
                    
                    spent = max(0.0, spent)
                    if spent <= self.max_spent_threshold:
                        rating = white_elo if mover_color == chess.WHITE else black_elo
                        bucket = round(rating / self.bucket_size) * self.bucket_size
                        bucket_sum[bucket] += spent
                        bucket_count[bucket] += 1
                        
                    prev_clock[mover_color] = clk

                node = nxt

        return {
            bucket: round(bucket_sum[bucket] / bucket_count[bucket], 3)
            for bucket in sorted(bucket_sum.keys())
            if bucket_count[bucket] >= 30
        }

    def build_and_save(self, out_json: str) -> Dict[int, float]:
        stats = self.build()
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
        return stats


def load_avg_time_by_rating(json_path: str) -> Dict[int, float]:
    """Carica il JSON prodotto da build_and_save, convertendo le chiavi in interi."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): float(v) for k, v in raw.items()}