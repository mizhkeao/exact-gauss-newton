from __future__ import annotations
import time
import torch

class Logger:

    def __init__(self):
        self.rows: list[dict] = []
        self.train_time = 0.0
        self._t0: float | None = None
        self.grad_evals = 0
        self.jac_evals = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def start(self):
        self._t0 = time.perf_counter()

    def pause(self):
        if self._t0 is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.train_time += time.perf_counter() - self._t0
        self._t0 = None

    def elapsed(self) -> float:
        running = time.perf_counter() - self._t0 if self._t0 is not None else 0.0
        return self.train_time + running

    def count_grad(self, n: int=1):
        self.grad_evals += n

    def count_jac(self, n: int=1):
        self.jac_evals += n

    def log(self, it: int, params_or_vec, loss: float, stage: str, res, extra: dict | None=None):
        self.pause()
        params = params_or_vec if isinstance(params_or_vec, dict) else res.fm.unflatten(params_or_vec)
        with torch.no_grad():
            err = res.rel_l2({k: v.detach() for k, v in params.items()})
        row = {'iter': it, 'time': self.train_time, 'loss': loss, 'rel_l2': err, 'stage': stage, 'grad_evals': self.grad_evals, 'jac_evals': self.jac_evals}
        if extra:
            row.update(extra)
        self.rows.append(row)
        self.start()
        return err

    def peak_memory_mb(self) -> float | None:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 2 ** 20
        return None
