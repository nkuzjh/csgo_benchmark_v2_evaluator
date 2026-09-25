"""Discrete image metrics copied from UniLIP's CSGO metric path.

The resize, batch aggregation and metric implementations intentionally follow
``benchmark_csgo_v1.py``.  Only Table 1 metrics are exposed here.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
import threading
from typing import Sequence
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from metric_assets import resolve_asset


_LPIPS_HUB_LOCK = threading.Lock()
_ALEXNET_HUB_FILENAME = "alexnet-owt-7be5be79.pth"


def _make_lpips_metric() -> LearnedPerceptualImagePatchSimilarity:
    """Let native TorchMetrics load the validated AlexNet through Torch Hub.

    TorchMetrics 1.8.2 does not expose an AlexNet weights path. Its torchvision
    constructor checks this Torch Hub filename before considering a download.
    The temporary symlink avoids copying the large checkpoint or changing the
    source asset, and the download guard makes version drift fail locally.
    """

    alexnet_path = resolve_asset("alexnet")
    with _LPIPS_HUB_LOCK, tempfile.TemporaryDirectory(prefix="csgo-lpips-") as temporary_dir:
        checkpoint_dir = Path(temporary_dir) / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / _ALEXNET_HUB_FILENAME).symlink_to(alexnet_path.resolve())
        # Preserve whether Torch Hub had an explicit override at all. Restoring
        # only get_dir() would freeze an environment-derived default directory.
        previous_hub_override = torch.hub._hub_dir
        try:
            torch.hub.set_dir(temporary_dir)
            with patch.object(
                torch.hub,
                "download_url_to_file",
                side_effect=RuntimeError("LPIPS AlexNet weights must be available as a validated local asset"),
            ):
                return LearnedPerceptualImagePatchSimilarity(net_type="alex")
        finally:
            torch.hub._hub_dir = previous_hub_override


class PairedImageDataset(Dataset):
    """Pair manifest-selected GT/prediction files with the UniLIP transform."""

    def __init__(
        self,
        gt_paths: Sequence[str | Path],
        pred_paths: Sequence[str | Path],
        size: int,
        *,
        uint8: bool = False,
    ):
        if len(gt_paths) != len(pred_paths):
            raise ValueError("GT and prediction path counts differ")
        if not gt_paths:
            raise ValueError("No image pairs selected")
        self.gt_paths = [Path(path) for path in gt_paths]
        self.pred_paths = [Path(path) for path in pred_paths]
        if uint8:
            self.transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
                transforms.PILToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.gt_paths)

    def __getitem__(self, index: int):
        gt_img = Image.open(self.gt_paths[index]).convert("RGB")
        pred_img = Image.open(self.pred_paths[index]).convert("RGB")
        return self.transform(gt_img), self.transform(pred_img)


class SingleImageDataset(Dataset):
    def __init__(self, paths: Sequence[str | Path], size: int):
        if not paths:
            raise ValueError("No image files selected")
        self.paths = [Path(path) for path in paths]
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
            transforms.PILToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        return self.transform(Image.open(self.paths[index]).convert("RGB"))


def compute_psnr_ssim(
    gt_paths: Sequence[str | Path],
    pred_paths: Sequence[str | Path],
    *,
    size: int,
    batch_size: int,
    device: str,
    num_workers: int = 4,
) -> tuple[float, float]:
    """Return batch-metric outputs weighted by batch size, as UniLIP does."""

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    dataset = PairedImageDataset(gt_paths, pred_paths, size=size, uint8=False)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    with torch.no_grad():
        for gt_batch, pred_batch in loader:
            gt_batch = gt_batch.to(device)
            pred_batch = pred_batch.to(device)
            total_psnr += psnr_metric(pred_batch, gt_batch).item() * gt_batch.size(0)
            total_ssim += ssim_metric(pred_batch, gt_batch).item() * gt_batch.size(0)
            count += gt_batch.size(0)
    return total_psnr / count, total_ssim / count


def compute_lpips(
    gt_paths: Sequence[str | Path],
    pred_paths: Sequence[str | Path],
    *,
    size: int,
    batch_size: int,
    device: str,
    num_workers: int = 4,
) -> float:
    lpips_metric = _make_lpips_metric().to(device)
    dataset = PairedImageDataset(gt_paths, pred_paths, size=size, uint8=False)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    total_lpips = 0.0
    count = 0
    with torch.no_grad():
        for gt_batch, pred_batch in loader:
            gt_batch = gt_batch.to(device) * 2.0 - 1.0
            pred_batch = pred_batch.to(device) * 2.0 - 1.0
            score = lpips_metric(pred_batch, gt_batch)
            total_lpips += score.item() * gt_batch.size(0)
            count += gt_batch.size(0)
    return total_lpips / count


def _sobel_edge_map(image_batch: torch.Tensor, edge_quantile: float) -> torch.Tensor:
    rgb_to_gray = image_batch[:, 0:1] * 0.299 + image_batch[:, 1:2] * 0.587 + image_batch[:, 2:3] * 0.114
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image_batch.device,
        dtype=image_batch.dtype,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=image_batch.device,
        dtype=image_batch.dtype,
    ).view(1, 1, 3, 3)
    grad_x = F.conv2d(rgb_to_gray, sobel_x, padding=1)
    grad_y = F.conv2d(rgb_to_gray, sobel_y, padding=1)
    edge_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)
    flat_edge_mag = edge_mag.flatten(start_dim=1)
    thresholds = torch.quantile(flat_edge_mag, edge_quantile, dim=1).view(-1, 1, 1, 1)
    return edge_mag > thresholds


def compute_boundary_metrics(
    gt_paths: Sequence[str | Path],
    pred_paths: Sequence[str | Path],
    *,
    size: int,
    batch_size: int,
    device: str,
    edge_quantile: float = 0.85,
    edge_tolerance: int = 2,
    num_workers: int = 4,
) -> dict[str, float]:
    dataset = PairedImageDataset(gt_paths, pred_paths, size=size, uint8=False)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    kernel_size = edge_tolerance * 2 + 1
    total_pred_edges = 0.0
    total_gt_edges = 0.0
    total_precision_hits = 0.0
    total_recall_hits = 0.0
    with torch.no_grad():
        for gt_batch, pred_batch in loader:
            gt_batch = gt_batch.to(device)
            pred_batch = pred_batch.to(device)
            gt_edges = _sobel_edge_map(gt_batch, edge_quantile)
            pred_edges = _sobel_edge_map(pred_batch, edge_quantile)
            gt_edges_dilated = F.max_pool2d(
                gt_edges.float(), kernel_size=kernel_size, stride=1, padding=edge_tolerance
            ) > 0
            pred_edges_dilated = F.max_pool2d(
                pred_edges.float(), kernel_size=kernel_size, stride=1, padding=edge_tolerance
            ) > 0
            total_precision_hits += (pred_edges & gt_edges_dilated).sum().item()
            total_recall_hits += (gt_edges & pred_edges_dilated).sum().item()
            total_pred_edges += pred_edges.sum().item()
            total_gt_edges += gt_edges.sum().item()

    precision = total_precision_hits / total_pred_edges if total_pred_edges > 0 else 0.0
    recall = total_recall_hits / total_gt_edges if total_gt_edges > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"Boundary_F1": float(f1), "Boundary_Precision": float(precision), "Boundary_Recall": float(recall)}


def compute_fid(
    gt_paths: Sequence[str | Path],
    pred_paths: Sequence[str | Path],
    *,
    batch_size: int,
    device: str,
    num_workers: int = 4,
) -> float:
    fid = FrechetInceptionDistance(
        feature=2048,
        feature_extractor_weights_path=str(resolve_asset("inception")),
    ).to(device)
    gt_dataset = SingleImageDataset(gt_paths, size=299)
    pred_dataset = SingleImageDataset(pred_paths, size=299)
    gt_loader = DataLoader(gt_dataset, batch_size=batch_size, num_workers=num_workers)
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, num_workers=num_workers)
    for batch_uint8 in gt_loader:
        fid.update(batch_uint8.to(device), real=True)
    for batch_uint8 in pred_loader:
        fid.update(batch_uint8.to(device), real=False)
    return float(fid.compute().item())


__all__ = [
    "PairedImageDataset",
    "compute_psnr_ssim",
    "compute_lpips",
    "compute_boundary_metrics",
    "compute_fid",
]
