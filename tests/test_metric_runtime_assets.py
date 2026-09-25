"""Focused checks that metric constructors consume local, selected weights."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import metrics_images  # noqa: E402
from fvd_metric import fvd  # noqa: E402


class RuntimeAssetTests(unittest.TestCase):
    def test_lpips_uses_local_symlink_and_restores_hub(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "alexnet-owt-7be5be79.pth"
            asset.write_bytes(b"fake validated asset")
            for original_override in (None, str(Path(directory) / "existing-hub")):
                with self.subTest(original_override=original_override), patch.object(
                    torch.hub, "_hub_dir", original_override
                ):
                    previous_hub_dir = torch.hub.get_dir()
                    seen = []

                    def native_constructor(*, net_type):
                        self.assertEqual(net_type, "alex")
                        hub_dir = Path(torch.hub.get_dir())
                        self.assertNotEqual(hub_dir, Path(previous_hub_dir))
                        checkpoint = hub_dir / "checkpoints" / asset.name
                        self.assertTrue(checkpoint.is_symlink())
                        self.assertEqual(checkpoint.resolve(), asset)
                        with self.assertRaisesRegex(RuntimeError, "validated local asset"):
                            torch.hub.download_url_to_file("https://invalid.example/weight", checkpoint)
                        seen.append(hub_dir)
                        return object()

                    with patch.object(metrics_images, "resolve_asset", return_value=asset) as resolver, patch.object(
                        metrics_images, "LearnedPerceptualImagePatchSimilarity", side_effect=native_constructor
                    ):
                        metrics_images._make_lpips_metric()

                    resolver.assert_called_once_with("alexnet")
                    self.assertEqual(torch.hub._hub_dir, original_override)
                    self.assertEqual(torch.hub.get_dir(), previous_hub_dir)
                    self.assertFalse(seen[0].exists())

    def test_lpips_restores_hub_after_constructor_error(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "alexnet-owt-7be5be79.pth"
            asset.write_bytes(b"fake validated asset")
            for original_override in (None, str(Path(directory) / "existing-hub")):
                with self.subTest(original_override=original_override), patch.object(
                    torch.hub, "_hub_dir", original_override
                ):
                    previous_hub_dir = torch.hub.get_dir()
                    with patch.object(metrics_images, "resolve_asset", return_value=asset), patch.object(
                        metrics_images, "LearnedPerceptualImagePatchSimilarity", side_effect=ValueError("constructor failed")
                    ):
                        with self.assertRaisesRegex(ValueError, "constructor failed"):
                            metrics_images._make_lpips_metric()
                    self.assertEqual(torch.hub._hub_dir, original_override)
                    self.assertEqual(torch.hub.get_dir(), previous_hub_dir)

    def test_fid_passes_resolved_path_to_native_constructor(self):
        asset = Path("/selected/inception.pth")
        captured = []

        def native_constructor(**kwargs):
            captured.append(kwargs)
            raise RuntimeError("stop before image loading")

        with patch.object(metrics_images, "resolve_asset", return_value=asset) as resolver, patch.object(
            metrics_images, "FrechetInceptionDistance", side_effect=native_constructor
        ):
            with self.assertRaisesRegex(RuntimeError, "stop before image loading"):
                metrics_images.compute_fid([], [], batch_size=1, device="cpu")
        resolver.assert_called_once_with("inception")
        self.assertEqual(captured, [{"feature": 2048, "feature_extractor_weights_path": str(asset)}])

    def test_fvd_loads_selected_path_and_honors_preferred_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "i3d_torchscript.pt"
            asset.write_bytes(b"fake validated asset")
            preferred = str(Path(directory) / "loaded_models")
            detector = unittest.mock.MagicMock()
            detector.eval.return_value = detector
            detector.to.return_value = detector
            fvd._feature_detector_cache.clear()
            with patch.dict(os.environ, {"UNILIP_FVD_CACHE_DIR": preferred}), patch.object(
                fvd, "resolve_asset", return_value=asset
            ) as resolver, patch.object(fvd.torch.jit, "load", return_value=detector) as loader:
                first = fvd.get_feature_detector(fvd._DEFAULT_DETECTOR_URL, "cpu")
                second = fvd.get_feature_detector(fvd._DEFAULT_DETECTOR_URL, "cpu")
            self.assertIs(first, detector)
            self.assertIs(second, detector)
            resolver.assert_any_call("i3d", preferred_dirs=(preferred,))
            loader.assert_called_once_with(str(asset))
            fvd._feature_detector_cache.clear()

    def test_fvd_open_url_never_downloads_missing_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "runtime downloads are disabled"):
                fvd.open_url("https://invalid.example/i3d.pt", cache_dir=directory)


if __name__ == "__main__":
    unittest.main()
