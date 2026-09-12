"""Base run config composed from section models."""

from pydantic import Field

from hale_vlm.core.config.sections import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    SampleConfig,
    TrainConfig,
    VizConfig,
)
from hale_vlm.core.config.sections.common import StrictModel


class RunConfig(StrictModel):
    variant: str
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    sample: SampleConfig = Field(default_factory=SampleConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    viz: VizConfig = Field(default_factory=VizConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    device: str | None = None
