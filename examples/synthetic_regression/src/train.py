#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Config:
    seed: int = _env_int("SEED", 7)
    iterations: int = _env_int("ITERATIONS", 300)
    width: int = _env_int("WIDTH", 64)
    depth: int = _env_int("DEPTH", 2)
    lr: float = _env_float("LR", 0.01)
    batch_size: int = _env_int("BATCH_SIZE", 128)
    train_points: int = _env_int("TRAIN_POINTS", 512)
    val_points: int = _env_int("VAL_POINTS", 512)
    noise_std: float = _env_float("NOISE_STD", 0.02)
    device: str = os.environ.get("DEVICE", "cpu")


def target_function(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) + 0.3 * torch.cos(3.0 * x) + 0.15 * x


def build_dataset(points: int, *, seed: int, noise_std: float) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = (torch.rand(points, 1, generator=generator) * (2.0 * math.pi)) - math.pi
    y = target_function(x)
    if noise_std > 0.0:
        y = y + noise_std * torch.randn(points, 1, generator=generator)
    return x, y


def build_model(width: int, depth: int) -> nn.Module:
    layers: list[nn.Module] = []
    in_features = 1
    for _ in range(depth):
        layers.append(nn.Linear(in_features, width))
        layers.append(nn.Tanh())
        in_features = width
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)


def mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        prediction = model(x)
        return torch.mean((prediction - y) ** 2).item()


def main() -> None:
    config = Config()
    set_seed(config.seed)

    device = torch.device(config.device)
    train_x, train_y = build_dataset(config.train_points, seed=config.seed, noise_std=config.noise_std)
    val_x, val_y = build_dataset(config.val_points, seed=config.seed + 1, noise_std=0.0)

    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    model = build_model(config.width, config.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()

    start_time = time.perf_counter()
    best_val_mse = float("inf")

    for step in range(1, config.iterations + 1):
        permutation = torch.randperm(train_x.size(0), device=device)
        batch_idx = permutation[: config.batch_size]
        batch_x = train_x[batch_idx]
        batch_y = train_y[batch_idx]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch_x)
        loss = loss_fn(prediction, batch_y)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 50 == 0 or step == config.iterations:
            current_val_mse = mse(model, val_x, val_y)
            best_val_mse = min(best_val_mse, current_val_mse)
            print(
                f"step:{step} train_batch_mse:{loss.item():.8f} val_mse:{current_val_mse:.8f}",
                flush=True,
            )

    elapsed_seconds = time.perf_counter() - start_time
    final_train_mse = mse(model, train_x, train_y)
    final_val_mse = mse(model, val_x, val_y)
    best_val_mse = min(best_val_mse, final_val_mse)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(
        "final_metrics "
        f"train_mse:{final_train_mse:.8f} "
        f"val_mse:{final_val_mse:.8f} "
        f"best_val_mse:{best_val_mse:.8f} "
        f"elapsed_seconds:{elapsed_seconds:.6f} "
        f"parameter_count:{parameter_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
