"""Tests for the ACT single-channel depth compatibility adapter."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.act_depth_adapter import expand_single_channel_image, install_act_depth_adapter
from scripts.train_act_depth import ensure_metric_depth_training_args


class ActDepthAdapterTest(unittest.TestCase):
    def test_training_args_force_metric_depth(self) -> None:
        args = ["train_act_depth.py", "--policy.type=act"]
        ensure_metric_depth_training_args(args)
        self.assertIn("--dataset.depth_output_unit=m", args)

        with self.assertRaisesRegex(ValueError, "must be 'm'"):
            ensure_metric_depth_training_args(
                ["train_act_depth.py", "--dataset.depth_output_unit=mm"]
            )

    def test_repeats_depth_and_preserves_metric_values(self) -> None:
        depth = torch.tensor([[[[0.25, 2.5]]]], dtype=torch.float32)
        adapted = expand_single_channel_image(torch.nn.Identity(), (depth,))[0]

        self.assertEqual(adapted.shape, (1, 3, 1, 2))
        for channel in range(3):
            torch.testing.assert_close(adapted[:, channel], depth[:, 0])

    def test_leaves_rgb_unchanged(self) -> None:
        rgb = torch.rand(2, 3, 8, 8)
        adapted = expand_single_channel_image(torch.nn.Identity(), (rgb,))[0]

        self.assertIs(adapted, rgb)

    def test_install_is_idempotent(self) -> None:
        from lerobot.policies.act.modeling_act import ACT

        install_act_depth_adapter()
        patched_init = ACT.__init__
        install_act_depth_adapter()

        self.assertIs(ACT.__init__, patched_init)

    def test_act_forward_accepts_rgb_and_metric_depth(self) -> None:
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy

        install_act_depth_adapter()
        config = ACTConfig(
            input_features={
                "observation.state": PolicyFeature(FeatureType.STATE, (4,)),
                "observation.images.table_camera": PolicyFeature(FeatureType.VISUAL, (3, 32, 32)),
                "observation.images.table_camera_depth": PolicyFeature(FeatureType.VISUAL, (1, 32, 32)),
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (4,))},
            chunk_size=2,
            n_action_steps=2,
            pretrained_backbone_weights=None,
            dim_model=32,
            n_heads=4,
            dim_feedforward=64,
            n_encoder_layers=1,
            n_decoder_layers=1,
            use_vae=False,
        )
        policy = ACTPolicy(config)
        batch = {
            "observation.state": torch.rand(2, 4),
            "observation.images.table_camera": torch.rand(2, 3, 32, 32),
            "observation.images.table_camera_depth": torch.rand(2, 1, 32, 32) * 2.0,
            "action": torch.rand(2, 2, 4),
            "action_is_pad": torch.zeros(2, 2, dtype=torch.bool),
        }

        loss, loss_dict = policy(batch)

        self.assertTrue(torch.isfinite(loss))
        self.assertIn("l1_loss", loss_dict)


if __name__ == "__main__":
    unittest.main()
