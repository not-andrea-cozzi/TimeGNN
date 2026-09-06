import os
import json
import time
import functools
import logging

logger = logging.getLogger("pipeline")


class PipelineState:

    def __init__(self, path: str):
        self.path = path
        self._data = {}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Stato pipeline in {self.path} illeggibile ({e}), riparto da vuoto.")
                self._data = {}
        else:
            self._data = {}

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._data, f, indent=2, sort_keys=True)
        os.replace(tmp_path, self.path) 

    def is_done(self, step: str) -> bool:
        return self._data.get(step, {}).get("done", False)

    def mark_done(self, step: str, **meta):
        self._data[step] = {"done": True, **meta}
        self.save()

    def mark_failed(self, step: str, error: str):
        entry = self._data.get(step, {})
        entry.update({"done": False, "last_error": error, "last_attempt_ts": time.time()})
        self._data[step] = entry
        self.save()

    def get_meta(self, step: str) -> dict:
        return self._data.get(step, {})


def retry(max_attempts: int = 3, base_delay: float = 2.0, exceptions=(Exception,)):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt == max_attempts:
                        break
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        f"{fn.__name__} fallito (tentativo {attempt}/{max_attempts}): {e}. "
                        f"Riprovo tra {delay:.1f}s."
                    )
                    time.sleep(delay)
            raise RuntimeError(f"{fn.__name__} fallito dopo {max_attempts} tentativi") from last_exc
        return wrapper
    return decorator


def file_ready(path: str, min_size_bytes: int = 1) -> bool:
    return os.path.exists(path) and os.path.getsize(path) >= min_size_bytes


def torch_pt_ready(path: str) -> bool:
    if not file_ready(path):
        return False
    try:
        import torch
        torch.load(path, weights_only=False)
        return True
    except Exception as e:
        logger.warning(f"{path} presente ma non caricabile ({e}), verra' rigenerato.")
        return False