# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, field_validator, model_validator
import torch


class ModelEnum(str, Enum):
    """Models supported by train.py."""

    unet = "unet"


class ActivationEnum(str, Enum):
    sigmoid = "sigmoid"
    hard_sigmoid = "hard_sigmoid"


class WeightEnum(str, Enum):
    """Weights supported by train.py."""

    imagenet = "imagenet"
    random = "random"
    ssl = "ssl"


class OptimizerEnum(str, Enum):
    """Optimizers supported by train.py."""

    adam = "adam"
    rmsprop = "rmsprop"
    sgd = "sgd"
    adamw = "adamw"


class LossEnum(str, Enum):
    """Losses supported by train.py."""

    mse = "mse"
    huber = "ignored_nan_huber"
    mae = "mae"


class SchedulerEnum(str, Enum):
    """Schedulers supported by train.py."""

    cosine = "cosine"


class PrecisionEnum(str, Enum):
    """Precision modes for training."""

    float16 = "16-mixed"


class TrainerConfig(BaseModel):
    """Validate input from yaml and/or argparse before passing to train.py."""

    # model params
    segmentation_model_name: ModelEnum = ModelEnum.unet
    backbone_name: str = "resnet18"
    weight_init: WeightEnum = WeightEnum.imagenet

    # optimizer params
    optimizer: OptimizerEnum = OptimizerEnum.adamw
    lr: float = 0.001
    weight_decay: float = 0.01
    patience: int = 6
    scheduler: SchedulerEnum = SchedulerEnum.cosine
    beta1: float = 0.9
    beta2: float = 0.999

    # loss params
    losses: List[LossEnum]  # List of losses for each target band

    # data module params
    patch_size: int = 512
    batch_size: int = 24
    num_workers: int = 6
    batches_per_epoch: int = 256

    # trainer params
    gpu_ids: List[int] = [0]
    seed: int = 0
    max_epochs: int = 30
    log_dir: str = "logs/"
    output_dir: str = "model_runs/"

    # input data params
    experiment_short_name: str = "example_experiment"

    # generated during validation if not explicitly passed in
    index: Optional[str] = None
    experiment_name: Optional[str] = None

    band_normalizers: Optional[List[List[float]]] = None
    in_channels: int = 3
    precision: PrecisionEnum = PrecisionEnum.float16
    activation: ActivationEnum = ActivationEnum.hard_sigmoid
    val_ratio: float = 0.01
    erase_blocks_in_priors: bool = False
    erase_priors_p: float = 1
    erase_priors_scale: List[float] = [0.0, 1.0]
    erase_priors_ratio: List[float] = [0.3, 33]
    priors_bands: Optional[List[int]] = None
    target_bands: List[int] = [2]
    target_normalizers: List[float] = [1.0]
    unet_heads: int = 2
    base_weight: float = 2.0
    weight_temperature: float = 1.0  # Temperature parameter for weight smoothing

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
        use_enum_values = True
        validate_assignment = True

    @field_validator("band_normalizers")
    def validate_band_normalizers(cls, normalizers) -> Optional[List[List[float]]]:
        if normalizers is None:
            return None
        if not all(len(n) == 2 for n in normalizers):
            raise ValueError("Each band normalizer must be [min, max]")
        return normalizers

    @field_validator("val_ratio")
    def validate_val_ratio(cls, val_ratio) -> float:
        if not 0 < val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        return val_ratio

    @field_validator("gpu_ids")
    def validate_gpus(cls, gpu_ids) -> List[int]:
        available_gpus = torch.cuda.device_count()
        for gpu_id in gpu_ids:
            if gpu_id >= available_gpus:
                raise ValueError(
                    f"Found only {available_gpus} GPU(s). Cannot use {gpu_id}."
                )
        print(f"Using the following GPU(s): {gpu_ids}.")
        return gpu_ids

    @field_validator("in_channels")
    def validate_channels(cls, in_channels) -> int:
        if in_channels < 1:
            raise ValueError("in_channels must be at least 1.")
        return in_channels

    @field_validator("beta1", "beta2")
    def validate_betas(cls, v) -> float:
        if not 0 < v < 1:
            raise ValueError("Beta parameters must be between 0 and 1")
        return v

    @field_validator("weight_init")
    def validate_weight_init(cls, v) -> WeightEnum:
        if isinstance(v, WeightEnum) or v in [e.value for e in WeightEnum]:
            return v
        raise ValueError(f"Invalid weight_init value: {v}")

    @model_validator(mode="after")
    def validate_beta_ordering(self):
        if self.optimizer == OptimizerEnum.adamw and self.beta1 >= self.beta2:
            raise ValueError("For AdamW optimizer, beta1 must be less than beta2")
        return self

    @classmethod
    def validate_experiment_name(cls, model):
        if model.experiment_name is not None:
            print("Using provided experiment name.")
            return

        print("Constructing experiment name.")
        short_name = model.experiment_short_name
        segmentation_model_name = model.segmentation_model_name
        backbone_name = model.backbone_name
        weight_init = model.weight_init
        lr = model.lr
        batch_size = model.batch_size
        experiment_name = f"{short_name}--{segmentation_model_name}--{backbone_name}--{weight_init}--lr_{lr}--bs_{batch_size}"
        model.__dict__["experiment_name"] = experiment_name

    @classmethod
    def validate_output_dir(cls, model):
        print("Validating output directories.")
        log_dir = model.log_dir
        output_dir = model.output_dir
        experiment = model.experiment_name

        output_log_dir = Path(log_dir)
        output_run_dir = Path(output_dir) / experiment

        output_log_dir.mkdir(parents=True, exist_ok=True)
        model.__dict__["log_dir"] = str(output_log_dir)

        output_run_dir.mkdir(parents=True, exist_ok=True)
        model.__dict__["output_dir"] = str(output_run_dir)

    @model_validator(mode="after")
    def validate_model(self):
        TrainerConfig.validate_experiment_name(self)
        TrainerConfig.validate_output_dir(self)
        return self
