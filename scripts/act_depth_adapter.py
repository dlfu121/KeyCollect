"""Runtime ACT adapter for single-channel metric depth observations."""

from __future__ import annotations

from typing import Any

import torch


def expand_single_channel_image(
    module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, ...]:
    """Repeat a BCHW depth tensor for a standard three-channel ResNet."""
    if not inputs:
        return inputs
    image, *rest = inputs
    if image.ndim == 4 and image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    return (image, *rest)


def install_act_depth_adapter() -> None:
    """Install an idempotent ACT constructor patch for RGB plus depth inputs."""
    from lerobot.policies.act.modeling_act import ACT

    if getattr(ACT, "_keycollect_depth_adapter_installed", False):
        return

    original_init = ACT.__init__

    def init_with_depth_adapter(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if hasattr(self, "backbone"):
            self.backbone.register_forward_pre_hook(expand_single_channel_image)

    ACT.__init__ = init_with_depth_adapter
    ACT._keycollect_depth_adapter_installed = True
