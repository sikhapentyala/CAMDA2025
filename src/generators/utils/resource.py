"""
Resource profiler tailored for the PGG / Private-PGM data generator.

Differences vs. ``resource_original.py``:

* Private-PGM is a pure NumPy / SciPy model running on CPU; there is no
  PyTorch CUDA allocator activity to measure, so the original profiler
  always reported ``peak_memory_gb = 0`` and a misleading "GPU"
  ``memory_type``. This version samples process RSS in a background
  thread to get an actual peak-RAM number during ``train()`` /
  ``generate()``.
* ``count_parameters`` understands the fitted ``mbi.GraphicalModel``
  stored at ``Private_PGM.model`` and sums the sizes of the clique
  log-potential tables -- the real "learned parameters" of PGM.
* GPU info is reported only as context; ``hardware_type`` /
  ``memory_type`` reflect the fact that training is CPU-bound.

The public interface (``start``, ``stop``, ``get_gpu_info``,
``get_cpu_info``, ``count_parameters``, ``save_metrics``) matches the
original so ``blue_team.py`` does not need to change.
"""

import os
import json
import time
import threading
import platform

import numpy as np
import psutil

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # torch is optional for PGM
    torch = None
    _TORCH_AVAILABLE = False


class ResourceProfiler:
    def __init__(self, device=None, sample_interval_sec: float = 0.25):
        self.device = device or ("cuda" if _cuda_available() else "cpu")
        self.sample_interval_sec = sample_interval_sec

        self.start_time = None
        self._proc = psutil.Process(os.getpid())

        # Peak-RSS sampler state (always on -- PGM is CPU)
        self._peak_rss_bytes = 0
        self._baseline_rss_bytes = 0
        self._sampler_stop = None
        self._sampler_thread = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        self.start_time = time.time()

        # Reset GPU peak counter if a GPU happens to be present; PGM
        # won't touch it, but if a future generator does we'll still
        # capture something sensible.
        if _cuda_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Start sampling RSS
        self._baseline_rss_bytes = self._proc.memory_info().rss
        self._peak_rss_bytes = self._baseline_rss_bytes
        self._sampler_stop = threading.Event()
        self._sampler_thread = threading.Thread(
            target=self._sample_rss_loop, daemon=True
        )
        self._sampler_thread.start()

    def stop(self):
        if self.start_time is None:
            raise RuntimeError("Profiler not started")

        # Stop the sampler and take one final reading
        if self._sampler_stop is not None:
            self._sampler_stop.set()
            self._sampler_thread.join()
        final_rss = self._proc.memory_info().rss
        if final_rss > self._peak_rss_bytes:
            self._peak_rss_bytes = final_rss

        elapsed = time.time() - self.start_time
        self.start_time = None

        return {
            "elapsed_sec": elapsed,
            **self.get_peak_memory(),
            **self.get_gpu_info(),
        }

    def _sample_rss_loop(self):
        while not self._sampler_stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                if rss > self._peak_rss_bytes:
                    self._peak_rss_bytes = rss
            except psutil.Error:
                pass
            # Event.wait returns early if .set() is called -> clean shutdown
            self._sampler_stop.wait(self.sample_interval_sec)

    # ------------------------------------------------------------------
    # Memory / hardware info
    # ------------------------------------------------------------------
    def get_peak_memory(self):
        """
        For PGM the meaningful number is process RSS: the working set
        of the NumPy/SciPy + mbi inference engine. We report it as the
        delta over the baseline measured at ``start()`` so the figure
        reflects what this run actually allocated, not the interpreter
        + imports that were already resident.
        """
        peak_delta_bytes = max(
            self._peak_rss_bytes - self._baseline_rss_bytes, 0
        )
        return {
            "peak_memory_gb": round(self._peak_rss_bytes / (1024 ** 3), 3),
            "peak_memory_delta_gb": round(peak_delta_bytes / (1024 ** 3), 3),
            "memory_type": "RAM",
        }

    def get_gpu_info(self):
        # Reported only as context. PGM does not use the GPU; the
        # caller should treat ``hardware_type`` as CPU for PGM runs.
        if _cuda_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            return {
                "gpu_name": props.name,
                "gpu_total_memory_gb": round(props.total_memory / (1024 ** 3), 3),
                "gpu_index": idx,
                "gpu_used_by_model": False,  # PGM is CPU-only
            }
        return {
            "gpu_name": None,
            "gpu_total_memory_gb": None,
            "gpu_index": None,
            "gpu_used_by_model": False,
        }

    def get_cpu_info(self):
        return {
            "cpu_name": platform.processor() or platform.machine(),
            "cpu_cores_physical": psutil.cpu_count(logical=False),
            "cpu_cores_logical": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
        }

    # ------------------------------------------------------------------
    # Parameter counting
    # ------------------------------------------------------------------
    def count_parameters(self, model):
        """
        Count parameters for a Private-PGM generator.

        ``model`` is expected to be the ``Private_PGM`` instance held by
        ``PGG_PGM_DataGenerator.model``. After training, its fitted
        ``mbi.GraphicalModel`` lives at ``model.model`` and holds one
        log-potential table per maximal clique in ``.potentials``.

        Returns the parameter count in millions, or ``None`` if the
        model has not been trained yet / does not look like PGM.
        Falls back to ``torch.nn.Module.parameters()`` if a PyTorch
        module is ever passed in.
        """
        if model is None:
            return None

        # PGM path: model.model is the fitted GraphicalModel
        pgm = getattr(model, "model", None)
        potentials = getattr(pgm, "potentials", None)
        if potentials:
            total = 0
            for factor in potentials.values():
                values = getattr(factor, "values", None)
                if values is None:
                    continue
                total += int(np.prod(values.shape))
            return round(total / 1e6, 6)

        # PyTorch fallback (kept so the profiler still works if reused)
        if _TORCH_AVAILABLE and hasattr(model, "parameters") and callable(
            model.parameters
        ):
            try:
                total = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
                return round(total / 1e6, 3)
            except TypeError:
                pass

        return None

    def count_parameters_detail(self, model):
        """
        Optional: per-clique breakdown for sanity checking / writeups.
        Returns ``None`` if ``model`` is not a fitted Private-PGM.
        """
        pgm = getattr(model, "model", None)
        potentials = getattr(pgm, "potentials", None)
        if not potentials:
            return None

        per_clique = {}
        for clique, factor in potentials.items():
            values = getattr(factor, "values", None)
            if values is None:
                continue
            per_clique[str(clique)] = {
                "shape": list(values.shape),
                "size": int(np.prod(values.shape)),
            }
        return {
            "num_cliques": len(per_clique),
            "total_params": sum(c["size"] for c in per_clique.values()),
            "per_clique": per_clique,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_metrics(self, metrics, split_no, experiment_name):
        out_dir = "resource_logs"
        os.makedirs(out_dir, exist_ok=True)

        fname = f"{out_dir}/{experiment_name or 'run'}_split{split_no}.json"
        with open(fname, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"[resource] saved -> {fname}")


def _cuda_available():
    return _TORCH_AVAILABLE and torch.cuda.is_available()

