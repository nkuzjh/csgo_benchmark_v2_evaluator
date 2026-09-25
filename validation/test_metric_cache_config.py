"""Cache selection plumbing only: no metric computation or weight download."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_eval


class CacheConfigTests(unittest.TestCase):
    def test_yaml_preference_uses_config_directory_then_resolved_asset(self):
        selected = Path("/shared/evaluator/loaded_models/i3d_torchscript.pt")
        config = {"continuous": {"fvd_cache_dir": "../UniLIP/loaded_models"}}
        with patch.dict(os.environ, {}, clear=True), \
                patch("metric_assets.resolve_asset", return_value=selected) as resolve, \
                patch.object(Path, "mkdir", side_effect=AssertionError("No runtime directory creation")):
            result = run_eval._set_fvd_cache(config, None, config_path="/shared/evaluator/benchmark_v2.yaml")
            resolve.assert_called_once_with("i3d", preferred_dirs=[Path("/shared/UniLIP/loaded_models")])
            self.assertEqual(result, str(selected.parent))
            self.assertEqual(os.environ["UNILIP_FVD_CACHE_DIR"], result)

    def test_cli_then_env_preference(self):
        config = {"continuous": {"fvd_cache_dir": "/yaml"}}
        with patch.dict(os.environ, {"UNILIP_FVD_CACHE_DIR": "/env"}, clear=True), \
                patch("metric_assets.resolve_asset", return_value=Path("/chosen/i3d_torchscript.pt")) as resolve:
            run_eval._set_fvd_cache(config, "/cli")
            resolve.assert_called_once_with("i3d", preferred_dirs=[Path("/cli"), Path("/env"), Path("/yaml")])
        with patch.dict(os.environ, {"UNILIP_FVD_CACHE_DIR": "/env"}, clear=True), \
                patch("metric_assets.resolve_asset", return_value=Path("/chosen/i3d_torchscript.pt")) as resolve:
            run_eval._set_fvd_cache(config, None)
            resolve.assert_called_once_with("i3d", preferred_dirs=[Path("/env"), Path("/yaml")])

    def test_no_config_uses_resolver_defaults(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch("metric_assets.resolve_asset", return_value=Path("/chosen/i3d_torchscript.pt")) as resolve:
            run_eval._set_fvd_cache({"continuous": {}}, None)
            resolve.assert_called_once_with("i3d", preferred_dirs=[])


if __name__ == "__main__":
    unittest.main()
