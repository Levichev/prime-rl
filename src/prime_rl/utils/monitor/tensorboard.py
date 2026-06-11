import atexit
import os
from pathlib import Path
from typing import Any

import torch
import verifiers as vf

from prime_rl.configs.shared import TensorBoardConfig
from prime_rl.utils.logger import get_logger
from prime_rl.utils.monitor.base import Monitor


class TensorBoardMonitor(Monitor):
    """Mirrors scalar metrics to TensorBoard event files. Only the master rank writes events."""

    def __init__(
        self,
        config: TensorBoardConfig,
        output_dir: Path | None = None,
        keep_full_history: bool = True,
    ):
        self.logger = get_logger()
        self.history: list[dict[str, Any]] = []
        self._keep_full_history = keep_full_history
        self.writer = None

        rank = int(os.environ.get("RANK", os.environ.get("DP_RANK", "0")))
        if rank != 0:
            self.logger.warning(f"Skipping {self.__class__.__name__} initialization from non-master rank ({rank})")
            return

        from torch.utils.tensorboard import SummaryWriter

        log_dir = config.log_dir or (output_dir or Path("outputs")) / "tensorboard"
        self.writer = SummaryWriter(log_dir=str(log_dir))
        # Trainers never call monitor.close() explicitly — flush pending events on interpreter exit.
        atexit.register(self.close)
        self.logger.info(f"Logging TensorBoard events to {log_dir}")

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self._keep_full_history:
            self.history.append(metrics)
        else:
            self.history = [metrics]
        if self.writer is None:
            return
        for key, value in metrics.items():
            if key == "step":
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.item()
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)

    def log_samples(self, rollouts: list[vf.RolloutOutput], step: int) -> None:
        pass

    def log_eval_samples(self, rollouts: list[vf.RolloutOutput], env_name: str, step: int) -> None:
        pass

    def save_final_summary(self, filename: str = "final_summary.json") -> None:
        pass

    def log_distributions(self, distributions: dict[str, list[float]], step: int) -> None:
        if self.writer is None:
            return
        for key, values in distributions.items():
            if values:
                self.writer.add_histogram(key, torch.tensor(values), step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
