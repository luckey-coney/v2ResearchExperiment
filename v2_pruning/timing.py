"""Optional synchronized wall timers; CUDA synchronization perturbs latency."""
import time


class _StageTimer:
    def __init__(self, timings, name, device=None):
        self.timings, self.name, self.device = timings, name, device

    def _synchronize(self):
        if getattr(self.device, "type", None) == "cuda":
            import torch
            torch.cuda.synchronize(self.device)

    def __enter__(self):
        if self.timings is not None:
            self._synchronize()
            self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.timings is not None:
            self._synchronize()
            elapsed = (time.perf_counter() - self.started) * 1000
            self.timings[self.name] = (self.timings.get(self.name) or 0.0) + elapsed
        return False
