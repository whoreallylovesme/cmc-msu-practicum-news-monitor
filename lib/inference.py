def resolve_device(artifact):
    import torch

    requested = artifact.get("device", "auto")
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is requested but unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device {requested}")
    return requested
