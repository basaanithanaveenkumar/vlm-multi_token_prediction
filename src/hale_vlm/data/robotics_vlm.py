"""Convert robotics (VLA) samples into VLM training samples."""

from __future__ import annotations

from hale_vlm.data.types import Modality, RoboticsVLMMode, VLASample, VLMSample


def format_robotics_instruction(task: str, mode: RoboticsVLMMode) -> str:
    task = task.strip()
    if mode == RoboticsVLMMode.PRETRAINING:
        return f"Robot task: {task}"
    if mode == RoboticsVLMMode.FINETUNING:
        return f"Execute the following manipulation task: {task}"
    if mode == RoboticsVLMMode.INSTRUCTION_TUNING:
        return f"Instruction: {task}\nDescribe the robot action needed to complete this task."
    return task


def vla_sample_to_vlm(sample: VLASample, mode: RoboticsVLMMode) -> VLMSample:
    """Map a robotics frame into a vision-language sample for VLM training."""
    text = format_robotics_instruction(sample.task, mode)
    modality = Modality.MULTI_IMAGE if len(sample.images) > 1 else Modality.IMAGE
    return VLMSample(
        dataset=sample.dataset,
        modality=modality,
        text=text,
        stage=None,
        category=f"robotics:{sample.stage.value}",
        images=list(sample.images),
        metadata={
            "source": "vla_registry",
            "robotics_mode": mode.value,
            "embodiment": sample.embodiment.value,
            "has_action": sample.action is not None,
            "has_state": sample.state is not None,
        },
    )
