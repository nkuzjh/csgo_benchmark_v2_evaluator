"""Explicitly prepare or check the metric model weights."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from metric_assets import MissingAssetError, prepare_assets


def _i3d_preferred(config_path: Path) -> tuple[Path, ...]:
    """Match run_eval's environment-over-config order for the FVD cache."""
    preferred: list[Path] = []
    configured_env = os.environ.get("UNILIP_FVD_CACHE_DIR")
    if configured_env:
        preferred.append(Path(configured_env).expanduser().resolve())
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        configured = (config.get("continuous") or {}).get("fvd_cache_dir")
        if configured:
            path = Path(configured).expanduser()
            preferred.append(path.resolve() if path.is_absolute() else (config_path.parent / path).resolve())
    return tuple(preferred)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify locally available weights without downloading")
    parser.add_argument("--evaluator-dir", help="Evaluator directory; defaults to this script's directory")
    parser.add_argument("--config", help="Benchmark YAML path; defaults to EVALUATOR_DIR/benchmark_v2.yaml")
    parser.add_argument("--preferred-dir", action="append", default=[], help="Preferred weight directory (repeatable)")
    parser.add_argument("--unilip-root", action="append", default=[], help="UniLIP project root (repeatable)")
    args = parser.parse_args()
    evaluator = Path(args.evaluator_dir).expanduser().resolve() if args.evaluator_dir else Path(__file__).resolve().parent
    config_path = Path(args.config).expanduser().resolve() if args.config else evaluator / "benchmark_v2.yaml"
    try:
        selected = prepare_assets(
            evaluator_dir=evaluator, preferred_dirs=args.preferred_dir,
            preferred_dirs_by_asset={"i3d": _i3d_preferred(config_path)},
            unilip_roots=args.unilip_root, check=args.check,
        )
    except (MissingAssetError, RuntimeError, OSError, yaml.YAMLError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(selected, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
