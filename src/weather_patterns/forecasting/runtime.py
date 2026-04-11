from __future__ import annotations


class GpuRuntimeRequirementError(RuntimeError):
    """Raised when a model stage is requested without an available GPU runtime."""


def resolve_model_device(
    device: str = "cuda",
    require_gpu: bool = True,
    stage_name: str = "model stage",
) -> str:
    """
    Resolve the runtime device for CUDA-bound stages.

    Discovery, training, and inference are GPU-only by project contract when the
    caller sets `require_gpu=True`.
    """

    normalized = device.strip().lower()
    if require_gpu and normalized != "cuda":
        raise GpuRuntimeRequirementError(
            f"{stage_name.capitalize()} is GPU-only. Configure the runtime device as 'cuda'."
        )
    if not require_gpu:
        return normalized

    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency in current MVP
        raise GpuRuntimeRequirementError(
            f"PyTorch with CUDA support is required for {stage_name}."
        ) from exc

    if not torch.cuda.is_available():
        raise GpuRuntimeRequirementError(
            f"CUDA GPU is required for {stage_name}, but no CUDA device is available."
        )
    return "cuda"
