"""Locate and explicitly prepare the fixed metric model weights.

Resolution is read-only. Downloads are available only through ``prepare_assets``
or the companion preparation command, never from a metric computation. The
digests below were audited from historically used local cache files; the I3D
digest is not represented as an independently published upstream checksum.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class AssetSpec:
    filenames: tuple[str, ...]
    url: str
    sha256: str


ASSETS = {
    "alexnet": AssetSpec(
        ("alexnet-owt-7be5be79.pth",),
        "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
        "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
    ),
    "inception": AssetSpec(
        ("weights-inception-2015-12-05-6726825d.pth",),
        "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth",
        "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2",
    ),
    "i3d": AssetSpec(
        ("5780f6fd48bed6b4f055c5cac089dbee_i3d_torchscript.pt", "i3d_torchscript.pt"),
        "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1",
        "bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2",
    ),
}


class MissingAssetError(FileNotFoundError):
    """No usable copy of a metric weight was found."""


def _evaluator_dir(value: str | os.PathLike[str] | None) -> Path:
    return Path(value).expanduser().resolve() if value is not None else Path(__file__).resolve().parent


def _as_paths(values: Iterable[str | os.PathLike[str]] | None) -> list[Path]:
    return [Path(value).expanduser().resolve() for value in values or ()]


def _unilip_roots(evaluator_dir: Path, supplied: Iterable[str | os.PathLike[str]] | None) -> list[Path]:
    roots = _as_paths(supplied)
    for key in ("CSGO_UNILIP_ROOT", "UNILIP_ROOT"):
        if os.environ.get(key):
            roots.append(Path(os.environ[key]).expanduser().resolve())
    roots.extend((evaluator_dir.parent / "UniLIP", evaluator_dir.parent / "UniLP"))
    return roots


def _candidate_files(base: Path, spec: AssetSpec) -> Iterable[Path]:
    # A user may supply the weight itself, a loaded_models directory, a hub
    # root, or a project root. These locations cover the established caches.
    if base.is_file():
        if base.name in spec.filenames:
            yield base
        return
    for directory in (
        base,
        base / "loaded_models",
        base / "hub" / "checkpoints",
        base / "checkpoints",
        base / "hub" / "checkpoints" / "checkpoints",
        base / "checkpoints" / "checkpoints",
        base / "loaded_models" / "hub" / "checkpoints",
        base / "loaded_models" / "checkpoints",
    ):
        for filename in spec.filenames:
            yield directory / filename


@lru_cache(maxsize=128)
def _sha256_cached(path: str, device: int, inode: int, size: int, mtime_ns: int, ctime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid(path: Path, spec: AssetSpec) -> bool:
    try:
        stat = path.stat()
        if not path.is_file() or stat.st_size == 0:
            return False
        return _sha256_cached(
            str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        ) == spec.sha256
    except (OSError, ValueError):
        return False


def _locate(
    name: str,
    *,
    evaluator_dir: str | os.PathLike[str] | None = None,
    preferred_dirs: Iterable[str | os.PathLike[str]] = (),
    unilip_roots: Iterable[str | os.PathLike[str]] | None = None,
) -> tuple[Path, str]:
    if name not in ASSETS:
        raise ValueError(f"Unknown metric asset {name!r}; expected one of {', '.join(ASSETS)}")
    evaluator = _evaluator_dir(evaluator_dir)
    spec = ASSETS[name]
    sources = [("preferred", path) for path in _as_paths(preferred_dirs)]
    sources += [("UniLIP", root / "loaded_models") for root in _unilip_roots(evaluator, unilip_roots)]
    sources.append(("evaluator", evaluator / "loaded_models"))
    seen: set[Path] = set()
    invalid: list[Path] = []
    for source, base in sources:
        for path in _candidate_files(base, spec):
            if path in seen:
                continue
            seen.add(path)
            if path.is_file():
                if _valid(path, spec):
                    return path, source
                invalid.append(path)
    detail = f"; rejected invalid SHA256 at: {', '.join(map(str, invalid))}" if invalid else ""
    raise MissingAssetError(
        f"Missing valid {name} metric weight ({' or '.join(spec.filenames)}){detail}. "
        f"Run: python {evaluator / 'prepare_metric_assets.py'}"
    )


def resolve_asset(
    name: str,
    *,
    evaluator_dir: str | os.PathLike[str] | None = None,
    preferred_dirs: Iterable[str | os.PathLike[str]] = (),
    unilip_roots: Iterable[str | os.PathLike[str]] | None = None,
) -> Path:
    """Return a SHA256-verified local weight. This function never downloads."""
    return _locate(name, evaluator_dir=evaluator_dir, preferred_dirs=preferred_dirs,
                   unilip_roots=unilip_roots)[0]


def _download(spec: AssetSpec, destination: Path, *, retries: int = 3, timeout: int = 60) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=f".{destination.name}.", suffix=".part",
                delete=False,
            ) as output:
                temporary = Path(output.name)
                with urllib.request.urlopen(spec.url, timeout=timeout) as response:
                    while block := response.read(1024 * 1024):
                        output.write(block)
                output.flush()
                os.fsync(output.fileno())
            if not _valid(temporary, spec):
                raise ValueError(f"Downloaded {destination.name} failed SHA256 verification")
            os.replace(temporary, destination)
            return
        except (OSError, ValueError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 4))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download verified metric weight to {destination}: {last_error}") from last_error


def prepare_assets(
    names: Iterable[str] = ASSETS,
    *,
    evaluator_dir: str | os.PathLike[str] | None = None,
    preferred_dirs: Iterable[str | os.PathLike[str]] = (),
    preferred_dirs_by_asset: Mapping[str, Iterable[str | os.PathLike[str]]] | None = None,
    unilip_roots: Iterable[str | os.PathLike[str]] | None = None,
    check: bool = False,
) -> dict[str, dict[str, str]]:
    """Reuse valid weights independently; download missing ones only when not checking.

    Write a record of the selected assets. Runtime resolution always scans and
    revalidates files, so an old manifest cannot select a stale path.
    """
    evaluator = _evaluator_dir(evaluator_dir)
    preferred_dirs = tuple(preferred_dirs)
    preferred_dirs_by_asset = {
        name: tuple(paths) for name, paths in (preferred_dirs_by_asset or {}).items()
    }
    unilip_roots = tuple(unilip_roots) if unilip_roots is not None else None
    selected: dict[str, dict[str, str]] = {}
    failures: list[str] = []
    for name in names:
        if name not in ASSETS:
            raise ValueError(f"Unknown metric asset {name!r}")
        try:
            candidates = preferred_dirs + preferred_dirs_by_asset.get(name, ())
            path, source = _locate(name, evaluator_dir=evaluator, preferred_dirs=candidates,
                                   unilip_roots=unilip_roots)
        except MissingAssetError as error:
            if check:
                failures.append(str(error))
                continue
            path = evaluator / "loaded_models" / ASSETS[name].filenames[0]
            _download(ASSETS[name], path)
            if not _valid(path, ASSETS[name]):
                raise RuntimeError(f"Downloaded {name} weight did not pass SHA256 verification")
            source = "downloaded to evaluator"
        selected[name] = {"path": str(path), "source": source, "sha256": ASSETS[name].sha256}
    manifest = evaluator / "loaded_models" / "metric_asset_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=manifest.parent,
                                         prefix=".metric_asset_manifest.", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            json.dump(selected, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, manifest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if failures:
        raise MissingAssetError("\n".join(failures))
    return selected
