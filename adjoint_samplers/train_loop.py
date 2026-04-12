# Copyright (c) Meta Platforms, Inc. and affiliates.

from omegaconf import DictConfig

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics.aggregation import MeanMetric

import adjoint_samplers.utils.train_utils as train_utils
from adjoint_samplers.components.matcher import Matcher


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def train_one_epoch(
    matcher: Matcher,
    model: torch.nn.Module,
    source: torch.nn.Module,
    optimizer: Optimizer,
    lr_schedule: LRScheduler | None,
    epoch: int,
    device: str,
    cfg: DictConfig,
):
    # build dataloader
    B = cfg.resample_batch_size
    M = max(1, matcher.resample_size // (B * cfg.world_size))
    loss_scale = matcher.loss_scale

    is_asbs_init_stage = train_utils.is_asbs_init_stage(epoch, cfg)

    for _ in range(M):
        x0 = source.sample([B,]).to(device)
        timesteps = train_utils.get_timesteps(**cfg.timesteps).to(device)
        matcher.populate_buffer(x0, timesteps, is_asbs_init_stage, epoch=epoch)

    dataloader = matcher.build_dataloader(cfg.train_batch_size)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    loader = iter(cycle(dataloader))

    model.train(True)
    for _ in range(cfg.train_itr_per_epoch):
        optimizer.zero_grad()

        data = next(loader)

        input, target = matcher.prepare_target(data, device)

        # Sanitize targets: replace NaN/Inf, then clip per-sample norms
        target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        if cfg.get("clip_target_norm", None):
            t_norms = torch.linalg.vector_norm(target, dim=-1, keepdim=True)
            t_clip = torch.clamp(
                float(cfg.clip_target_norm) / (t_norms + 1e-6), max=1.0
            )
            target = target * t_clip

        output = model(*input)
        output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

        loss = loss_scale * ((output - target)**2).mean()

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()          # discard any stale grads
            epoch_loss.update(float('nan'))
            continue                       # skip this batch entirely

        loss.backward()

        if cfg.clip_grad_norm:
            max_norm = float(cfg.clip_grad_norm) if isinstance(cfg.clip_grad_norm, (int, float)) and cfg.clip_grad_norm > 1 else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        optimizer.step()

        epoch_loss.update(loss.item())
        if lr_schedule:
            lr_schedule.step()

    return float(epoch_loss.compute().detach().cpu())
