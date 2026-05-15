"""训练设备选择工具。"""

from __future__ import annotations


def resolve_torch_device(device: str | None = 'auto') -> str:
    """按 CUDA、Apple MPS、CPU 的优先级解析 PyTorch 设备。"""

    requested = str(device or 'auto').strip().lower()
    if requested in {'', 'auto', 'gpu', 'accelerator'}:
        try:
            import torch
        except ModuleNotFoundError:
            return 'cpu'

        if torch.cuda.is_available():
            return 'cuda'
        try:
            mps_backend = torch.backends.mps
        except AttributeError:
            mps_backend = None
        if mps_backend is not None and mps_backend.is_available():
            return 'mps'
        return 'cpu'

    if requested == 'cuda':
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError('Requested --device cuda, but torch.cuda.is_available() is False.')
        return 'cuda'

    if requested == 'mps':
        import torch

        try:
            mps_backend = torch.backends.mps
        except AttributeError:
            mps_backend = None
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError('Requested --device mps, but torch.backends.mps.is_available() is False.')
        return 'mps'

    return requested


def log_resolved_device(requested_device: str | None, resolved_device: str, *, prefix: str = '[Device]') -> None:
    """打印设备选择结果，便于确认是否走上 GPU/MPS。"""

    requested = str(requested_device or 'auto')
    if requested.lower() == str(resolved_device).lower():
        print(f'{prefix} Using {resolved_device}')
    else:
        print(f'{prefix} Auto-selected {resolved_device} from requested={requested}')
