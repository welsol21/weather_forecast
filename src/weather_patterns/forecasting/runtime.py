from __future__ import annotations


class GpuRuntimeRequirementError(RuntimeError):
    """Raised when a model stage is requested without an available GPU runtime."""


def resolve_model_device(device: str = "cuda", require_gpu: bool = True) -> str:
    """
    Resolve the runtime device for model stages.

    Training and inference are GPU-only by project contract. This helper is meant
    to be called by future model-training and model-inference entry points.
    """

    normalized = device.strip().lower()
    if require_gpu and normalized != "cuda":
        raise GpuRuntimeRequirementError(
            "Model training and inference are GPU-only. Configure the model device as 'cuda'."
        )
    if not require_gpu:
        return normalized

    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency in current MVP
        raise GpuRuntimeRequirementError(
            "PyTorch with CUDA support is required for GPU-only model stages."
        ) from exc

    if not torch.cuda.is_available():
        raise GpuRuntimeRequirementError(
            "CUDA GPU is required for model training and inference, but no CUDA device is available."
        )
    return "cuda"
