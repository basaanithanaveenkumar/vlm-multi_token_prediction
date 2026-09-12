"""Distributed training utilities for data parallelism with PyTorch."""

from __future__ import annotations

import os
from typing import Literal

import torch
import torch.distributed as dist
from loguru import logger
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP

ParallelStrategy = Literal["none", "dp", "ddp", "fsdp"]


class DistributedConfig:
    """Configuration for distributed training."""

    def __init__(
        self,
        strategy: ParallelStrategy = "none",
        backend: str = "nccl",
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = True,
        static_graph: bool = False,
    ):
        self.strategy = strategy
        self.backend = backend
        self.find_unused_parameters = find_unused_parameters
        self.gradient_as_bucket_view = gradient_as_bucket_view
        self.static_graph = static_graph

        # Runtime state
        self.is_initialized = False
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.is_main_process = True


def setup_distributed(
    strategy: ParallelStrategy = "none",
    backend: str | None = None,
) -> DistributedConfig:
    """
    Initialize distributed training environment.

    Args:
        strategy: Parallelization strategy ('none', 'dp', 'ddp', 'fsdp')
        backend: Communication backend for DDP/FSDP ('nccl', 'gloo', 'mpi')

    Returns:
        DistributedConfig with runtime information
    """
    config = DistributedConfig(strategy=strategy)

    if strategy == "none":
        logger.info("distributed training disabled (strategy=none)")
        return config

    if strategy == "dp":
        # DataParallel doesn't require distributed initialization
        if not torch.cuda.is_available():
            logger.warning("DataParallel requires CUDA; falling back to single device")
            config.strategy = "none"
            return config

        n_gpus = torch.cuda.device_count()
        if n_gpus < 2:
            logger.warning(
                "DataParallel requires multiple GPUs; found {}; using single device", n_gpus
            )
            config.strategy = "none"
        else:
            logger.info("DataParallel enabled with {} GPUs", n_gpus)
        return config

    # DDP and FSDP require torch.distributed
    if strategy in ("ddp", "fsdp"):
        # Auto-detect backend if not specified
        if backend is None:
            if torch.cuda.is_available():
                backend = "nccl"
            else:
                backend = "gloo"
        config.backend = backend

        # Check if already initialized (e.g., by torchrun)
        if dist.is_available() and dist.is_initialized():
            config.is_initialized = True
            config.rank = dist.get_rank()
            config.world_size = dist.get_world_size()
            config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            config.is_main_process = config.rank == 0
            logger.info(
                "distributed already initialized: strategy={} backend={} rank={}/{} local_rank={}",
                strategy,
                backend,
                config.rank,
                config.world_size,
                config.local_rank,
            )
            return config

        # Initialize from environment variables (torchrun sets these)
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            config.rank = int(os.environ["RANK"])
            config.world_size = int(os.environ["WORLD_SIZE"])
            config.local_rank = int(os.environ.get("LOCAL_RANK", 0))

            # Initialize process group
            if not dist.is_available():
                logger.error("torch.distributed not available; cannot use {}", strategy)
                config.strategy = "none"
                return config

            try:
                dist.init_process_group(
                    backend=backend,
                    init_method="env://",
                    rank=config.rank,
                    world_size=config.world_size,
                )
                config.is_initialized = True
                config.is_main_process = config.rank == 0

                if config.is_main_process:
                    logger.info(
                        "initialized {} with backend={} world_size={} local_rank={}",
                        strategy.upper(),
                        backend,
                        config.world_size,
                        config.local_rank,
                    )
            except Exception as e:
                logger.error("failed to initialize distributed: {}", e)
                config.strategy = "none"
                return config
        else:
            logger.warning(
                "{} requested but RANK/WORLD_SIZE not set; "
                "use 'torchrun' or 'torch.distributed.launch'; falling back to single device",
                strategy.upper(),
            )
            config.strategy = "none"

    return config


def cleanup_distributed(config: DistributedConfig) -> None:
    """Clean up distributed training resources."""
    if config.is_initialized and dist.is_initialized():
        dist.destroy_process_group()
        logger.debug("destroyed distributed process group")


def wrap_model_parallel(
    model: torch.nn.Module,
    config: DistributedConfig,
    device: torch.device | str,
) -> torch.nn.Module:
    """
    Wrap model with appropriate parallelization strategy.

    Args:
        model: The model to wrap
        config: Distributed configuration
        device: Target device

    Returns:
        Wrapped model (or original if strategy is 'none')
    """
    if config.strategy == "none":
        return model

    if config.strategy == "dp":
        # DataParallel: simple multi-GPU on single node
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            model = DP(model, device_ids=device_ids)
            logger.info("wrapped model with DataParallel on {} GPUs", len(device_ids))
        else:
            logger.warning("DataParallel requested but insufficient GPUs; using single device")
        return model

    if config.strategy == "ddp":
        # DistributedDataParallel: multi-GPU, multi-node
        if not config.is_initialized:
            logger.error("DDP requested but distributed not initialized")
            return model

        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(config.local_rank)
            device = torch.device(f"cuda:{config.local_rank}")
            model = model.to(device)

        model = DDP(
            model,
            device_ids=[config.local_rank] if torch.cuda.is_available() else None,
            output_device=config.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=config.find_unused_parameters,
            gradient_as_bucket_view=config.gradient_as_bucket_view,
            static_graph=config.static_graph,
        )
        logger.info(
            "wrapped model with DDP on rank={} local_rank={} device={}",
            config.rank,
            config.local_rank,
            device,
        )
        return model

    if config.strategy == "fsdp":
        # Fully Sharded Data Parallel: memory-efficient multi-GPU
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        except ImportError:
            logger.error("FSDP requires PyTorch >= 1.12; falling back to DDP")
            config.strategy = "ddp"
            return wrap_model_parallel(model, config, device)

        if not config.is_initialized:
            logger.error("FSDP requested but distributed not initialized")
            return model

        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(config.local_rank)
            device = torch.device(f"cuda:{config.local_rank}")
            model = model.to(device)

        # Configure FSDP with auto-wrapping for large models
        auto_wrap_policy = size_based_auto_wrap_policy(
            min_num_params=1e6  # Wrap modules with >1M parameters
        )

        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=config.local_rank if torch.cuda.is_available() else None,
            mixed_precision=None,  # Can be configured for fp16/bf16
        )
        logger.info(
            "wrapped model with FSDP on rank={} local_rank={} device={}",
            config.rank,
            config.local_rank,
            device,
        )
        return model

    logger.warning("unknown strategy={}; returning unwrapped model", config.strategy)
    return model


def get_world_size(config: DistributedConfig | None = None) -> int:
    """Get the number of processes in distributed training."""
    if config and config.is_initialized:
        return config.world_size
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank(config: DistributedConfig | None = None) -> int:
    """Get the rank of current process."""
    if config and config.is_initialized:
        return config.rank
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process(config: DistributedConfig | None = None) -> bool:
    """Check if current process is the main process (rank 0)."""
    if config:
        return config.is_main_process
    return get_rank(config) == 0


def barrier(config: DistributedConfig | None = None) -> None:
    """Synchronize all processes."""
    if config and config.is_initialized:
        dist.barrier()
    elif dist.is_available() and dist.is_initialized():
        dist.barrier()


def reduce_dict(
    metrics: dict[str, float],
    config: DistributedConfig | None = None,
    average: bool = True,
) -> dict[str, float]:
    """
    Reduce metrics across all processes.

    Args:
        metrics: Dictionary of metric name -> value
        config: Distributed configuration
        average: If True, average across processes; otherwise sum

    Returns:
        Reduced metrics dictionary
    """
    if not config or not config.is_initialized:
        return metrics

    if not dist.is_initialized():
        return metrics

    world_size = get_world_size(config)
    if world_size == 1:
        return metrics

    reduced = {}
    for key, value in metrics.items():
        tensor = torch.tensor(value, dtype=torch.float32, device=torch.cuda.current_device())
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if average:
            tensor /= world_size
        reduced[key] = tensor.item()

    return reduced
