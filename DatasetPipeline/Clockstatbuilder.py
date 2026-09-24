from __future__ import annotations

import json
import logging
import math
import os
import random
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

logger = logging.getLogger("clock_stats")

_MIN_SIGMA = 0.3

Stat = Tuple[float, float, int]


def _bucket(rating: float, bucket_size: int) -> int:
    return int(round(rating / bucket_size) * bucket_size)


def _finalize(n: int, s: float, ss: float) -> Stat:
    mu = s / n
    var = max(ss / n - mu * mu, 0.0)
    return mu, math.sqrt(var), n


class ClockStatsBuilder:
    def __init__(
        self,
        jsonl_paths: Iterable[str],
        bucket_size: int = 100,
        min_count: int = 30,
        max_seconds: float = 300.0,
        min_seconds: float = 0.05,
    ) -> None:
        self.jsonl_paths = list(jsonl_paths)
        self.bucket_size = bucket_size
        self.min_count = min_count
        self.max_seconds = max_seconds
        self.min_seconds = min_seconds

    def _iter_records(self) -> Iterator[dict]:
        seen = set()
        for path in self.jsonl_paths:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    pid = rec.get("problem_id")
                    if pid is not None:
                        h = hash(pid)
                        if h in seen:
                            continue
                        seen.add(h)
                    yield rec

    def build(self) -> dict:
        acc_rm: Dict[Tuple[int, int], List[float]] = defaultdict(lambda: [0, 0.0, 0.0])
        acc_r: Dict[int, List[float]] = defaultdict(lambda: [0, 0.0, 0.0])
        acc_g = [0, 0.0, 0.0]

        for rec in self._iter_records():
            if not rec.get("clock_is_real"):
                continue
            clock = rec.get("clock_seconds")
            rating = rec.get("rating")
            mate_n = rec.get("mate_n")
            if clock is None or rating is None or mate_n is None:
                continue
            clock = float(clock)
            if not (self.min_seconds <= clock <= self.max_seconds):
                continue
            lv = math.log(clock)
            b = _bucket(float(rating), self.bucket_size)
            for cell in (acc_rm[(b, int(mate_n))], acc_r[b], acc_g):
                cell[0] += 1
                cell[1] += lv
                cell[2] += lv * lv

        if acc_g[0] == 0:
            raise ValueError("ClockStatsBuilder: nessun record con clock reale trovato.")

        by_rating_mate = {
            f"{b}|{m}": list(_finalize(int(c[0]), c[1], c[2]))
            for (b, m), c in sorted(acc_rm.items())
            if c[0] >= self.min_count
        }
        by_rating = {
            str(b): list(_finalize(int(c[0]), c[1], c[2]))
            for b, c in sorted(acc_r.items())
            if c[0] >= self.min_count
        }
        return {
            "bucket_size": self.bucket_size,
            "global": list(_finalize(int(acc_g[0]), acc_g[1], acc_g[2])),
            "by_rating": by_rating,
            "by_rating_mate": by_rating_mate,
        }

    def build_and_save(self, out_json: str) -> dict:
        stats = self.build()
        os.makedirs(os.path.dirname(os.path.abspath(out_json)) or ".", exist_ok=True)
        tmp = out_json + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
        os.replace(tmp, out_json)
        logger.info(
            "[ClockStatsBuilder] global n=%d, celle rating=%d, celle rating x mate_n=%d -> %s",
            stats["global"][2], len(stats["by_rating"]), len(stats["by_rating_mate"]), out_json,
        )
        return stats


class ClockSampler:
    def __init__(
        self,
        global_stats: Stat,
        by_rating: Dict[int, Stat],
        by_rating_mate: Dict[Tuple[int, int], Stat],
        bucket_size: int = 100,
        mode: str = "lognormal",
        condition_on_mate_n: bool = True,
        min_seconds: float = 0.5,
        cap_seconds: float = 300.0,
        max_bucket_distance: int = 300,
    ) -> None:
        if mode not in ("lognormal", "constant"):
            raise ValueError(f"ClockSampler: mode non valido: {mode}")
        self._global = global_stats
        self._by_rating = by_rating
        self._by_rating_mate = by_rating_mate
        self.bucket_size = bucket_size
        self.mode = mode
        self.condition_on_mate_n = condition_on_mate_n
        self.min_seconds = min_seconds
        self.cap_seconds = cap_seconds
        self.max_bucket_distance = max_bucket_distance

        self._buckets_by_mate: Dict[int, List[int]] = defaultdict(list)
        for (b, m) in by_rating_mate:
            self._buckets_by_mate[m].append(b)
        self._rating_buckets = sorted(by_rating)

    @classmethod
    def from_json(cls, path: str, **kwargs) -> "ClockSampler":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        by_rating = {int(k): tuple(v) for k, v in raw["by_rating"].items()}
        by_rating_mate: Dict[Tuple[int, int], Stat] = {}
        for k, v in raw["by_rating_mate"].items():
            b, m = k.split("|")
            by_rating_mate[(int(b), int(m))] = tuple(v)
        return cls(
            tuple(raw["global"]),
            by_rating,
            by_rating_mate,
            bucket_size=int(raw.get("bucket_size", 100)),
            **kwargs,
        )

    @classmethod
    def from_avg_time(cls, avg_time_by_rating: Dict[int, float], sigma: float = 0.9, **kwargs) -> "ClockSampler":
        if not avg_time_by_rating:
            raise ValueError("ClockSampler.from_avg_time: avg_time_by_rating vuoto.")
        by_rating = {
            int(b): (math.log(max(float(v), 1e-3)) - sigma * sigma / 2.0, sigma, 0)
            for b, v in avg_time_by_rating.items()
        }
        mu_global = sum(s[0] for s in by_rating.values()) / len(by_rating)
        return cls((mu_global, sigma, 0), by_rating, {}, **kwargs)

    def _lookup(self, rating: float, mate_n: Optional[int]) -> Stat:
        b = _bucket(rating, self.bucket_size)
        if self.condition_on_mate_n and mate_n is not None:
            candidates = self._buckets_by_mate.get(int(mate_n))
            if candidates:
                nearest = min(candidates, key=lambda c: abs(c - b))
                if abs(nearest - b) <= self.max_bucket_distance:
                    return self._by_rating_mate[(nearest, int(mate_n))]
        if self._rating_buckets:
            nearest = min(self._rating_buckets, key=lambda c: abs(c - b))
            return self._by_rating[nearest]
        return self._global

    def sample(self, rating: float, mate_n: Optional[int], seed_key: str) -> float:
        if self.mode == "constant":
            value = math.exp(self._global[0])
        else:
            mu, sigma, _ = self._lookup(float(rating), mate_n)
            sigma = max(sigma, _MIN_SIGMA)
            value = math.exp(random.Random(seed_key).gauss(mu, sigma))
        return min(max(value, self.min_seconds), self.cap_seconds)