from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import metric_assets
import prepare_metric_assets


class MetricAssetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.evaluator = self.root / "evaluator"
        self.evaluator.mkdir()
        self.legacy = self.root / "UniLIP" / "loaded_models"
        self.legacy.mkdir(parents=True)
        self.good = {name: f"valid {name}".encode() for name in metric_assets.ASSETS}
        self.specs = {
            name: metric_assets.AssetSpec(
                (f"{name}.bin",), f"https://example.test/{name}", hashlib.sha256(data).hexdigest()
            )
            for name, data in self.good.items()
        }
        asset_patch = patch.dict(metric_assets.ASSETS, self.specs, clear=True)
        asset_patch.start()
        self.addCleanup(asset_patch.stop)

    def write(self, directory: Path, name: str, *, content: bytes | None = None) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.bin"
        path.write_bytes(self.good[name] if content is None else content)
        return path

    def test_preferred_weight_reused_without_network_or_mutation(self):
        preferred = self.root / "preferred" / "hub" / "checkpoints"
        path = self.write(preferred, "i3d")
        before = path.stat().st_mtime_ns
        with patch.object(metric_assets.urllib.request, "urlopen", side_effect=AssertionError("network")):
            selected = metric_assets.prepare_assets(
                ("i3d",), evaluator_dir=self.evaluator,
                preferred_dirs=(self.root / "preferred",), unilip_roots=(),
            )
        self.assertEqual(selected["i3d"]["path"], str(path))
        self.assertEqual(selected["i3d"]["source"], "preferred")
        self.assertEqual(path.stat().st_mtime_ns, before)
        self.assertFalse((self.evaluator / "loaded_models" / "i3d.bin").exists())
        manifest = json.loads((self.evaluator / "loaded_models" / "metric_asset_manifest.json").read_text())
        self.assertEqual(manifest, selected)

    def test_empty_legacy_directory_falls_back_to_evaluator(self):
        local = self.write(self.evaluator / "loaded_models", "alexnet")
        selected = metric_assets.resolve_asset("alexnet", evaluator_dir=self.evaluator)
        self.assertEqual(selected, local)

    def test_partial_sources_are_selected_independently(self):
        preferred = self.root / "custom"
        i3d = self.write(preferred / "checkpoints", "i3d")
        alexnet = self.write(self.legacy, "alexnet")
        inception = self.write(self.evaluator / "loaded_models", "inception")
        with patch.object(metric_assets.urllib.request, "urlopen", side_effect=AssertionError("network")):
            selected = metric_assets.prepare_assets(
                evaluator_dir=self.evaluator, preferred_dirs=(preferred,)
            )
        self.assertEqual({k: v["path"] for k, v in selected.items()}, {
            "alexnet": str(alexnet), "inception": str(inception), "i3d": str(i3d),
        })

    def test_invalid_hash_is_rejected_and_valid_fallback_used(self):
        preferred = self.root / "preferred"
        bad = self.write(preferred, "i3d", content=b"wrong")
        good = self.write(self.legacy, "i3d")
        self.assertEqual(metric_assets.resolve_asset(
            "i3d", evaluator_dir=self.evaluator, preferred_dirs=(preferred,)
        ), good)
        self.assertEqual(bad.read_bytes(), b"wrong")

    def test_missing_reports_setup_command_and_invalid_path(self):
        bad = self.write(self.legacy, "alexnet", content=b"truncated")
        with self.assertRaises(metric_assets.MissingAssetError) as error:
            metric_assets.resolve_asset("alexnet", evaluator_dir=self.evaluator)
        self.assertIn(str(bad), str(error.exception))
        self.assertIn("prepare_metric_assets.py", str(error.exception))
        with self.assertRaises(metric_assets.MissingAssetError):
            metric_assets.prepare_assets(("alexnet",), evaluator_dir=self.evaluator, check=True)

    def test_download_missing_to_evaluator_without_touching_external(self):
        external = self.write(self.legacy, "alexnet")
        before = external.stat().st_mtime_ns

        def get(url: str, timeout: int):
            self.assertEqual(url, self.specs["inception"].url)
            return io.BytesIO(self.good["inception"])

        with patch.object(metric_assets.urllib.request, "urlopen", side_effect=get) as network:
            selected = metric_assets.prepare_assets(
                ("alexnet", "inception"), evaluator_dir=self.evaluator
            )
        self.assertEqual(network.call_count, 1)
        self.assertEqual(selected["alexnet"]["path"], str(external))
        self.assertEqual(external.stat().st_mtime_ns, before)
        local = self.evaluator / "loaded_models" / "inception.bin"
        self.assertEqual(local.read_bytes(), self.good["inception"])
        self.assertEqual(selected["inception"]["source"], "downloaded to evaluator")

    def test_bad_download_never_replaces_existing_file(self):
        local = self.write(self.evaluator / "loaded_models", "i3d", content=b"old bad file")
        with patch.object(metric_assets.urllib.request, "urlopen", return_value=io.BytesIO(b"partial")), \
                patch.object(metric_assets.time, "sleep"):
            with self.assertRaises(RuntimeError):
                metric_assets.prepare_assets(("i3d",), evaluator_dir=self.evaluator)
        self.assertEqual(local.read_bytes(), b"old bad file")
        self.assertFalse(list(local.parent.glob("*.part")))

    def test_unilip_alias_and_environment(self):
        alias = self.root / "UniLP" / "loaded_models"
        path = self.write(alias, "alexnet")
        self.assertEqual(metric_assets.resolve_asset("alexnet", evaluator_dir=self.evaluator), path)
        configured = self.root / "configured" / "loaded_models"
        other = self.write(configured, "alexnet")
        with patch.dict(metric_assets.os.environ, {"CSGO_UNILIP_ROOT": str(configured.parent)}):
            self.assertEqual(metric_assets.resolve_asset("alexnet", evaluator_dir=self.evaluator), other)

    def test_asset_specific_preference_and_config_relative_path(self):
        configured = self.root / "custom" / "loaded_models"
        i3d = self.write(configured, "i3d")
        alexnet = self.write(self.legacy, "alexnet")
        self.evaluator.joinpath("benchmark_v2.yaml").write_text(
            "continuous:\n  fvd_cache_dir: ../custom/loaded_models\n"
        )
        preferred_i3d = prepare_metric_assets._i3d_preferred(self.evaluator / "benchmark_v2.yaml")
        self.assertEqual(preferred_i3d, (configured,))
        selected = metric_assets.prepare_assets(
            ("alexnet", "i3d"), evaluator_dir=self.evaluator,
            preferred_dirs_by_asset={"i3d": preferred_i3d}, check=True,
        )
        self.assertEqual(selected["i3d"]["path"], str(i3d))
        self.assertEqual(selected["alexnet"]["path"], str(alexnet))
        env_cache = self.root / "env_cache"
        self.write(env_cache, "i3d")
        with patch.dict(metric_assets.os.environ, {"UNILIP_FVD_CACHE_DIR": str(env_cache)}):
            self.assertEqual(
                prepare_metric_assets._i3d_preferred(self.evaluator / "benchmark_v2.yaml"),
                (env_cache, configured),
            )


if __name__ == "__main__":
    unittest.main()
