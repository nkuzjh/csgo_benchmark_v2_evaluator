"""
DISCLAIMER
This implementation is largely inspired by the implemenation from the StyleGAN-V repository
https://github.com/universome/stylegan-v
However, it is adapted to videos in main memory and simplified.
The authors of the StyleGAN-V repository verified the consistency of their PyTorch implementation with the original Tensorflow implementation.
The original implementation can be found here: https://github.com/google-research/google-research/tree/master/frechet_video_distance

The link used to download the pretrained feature extraction model was provided by the StyleGAN-V authors. I cannot garantuee it is still working.
"""

import numpy as np
import scipy
from torch.utils.data import DataLoader, TensorDataset
import torch
import hashlib
import os
import glob
from pathlib import Path

from metric_assets import resolve_asset

_feature_detector_cache = dict()
_DEFAULT_DETECTOR_URL = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'

# Keep the upstream URL argument, but resolve it to an existing local file.
def _detector_path(detector_url, cache_dir=None):
    if detector_url == _DEFAULT_DETECTOR_URL:
        preferred_dir = cache_dir or os.environ.get('UNILIP_FVD_CACHE_DIR')
        return resolve_asset('i3d', preferred_dirs=(preferred_dir,) if preferred_dir else ())

    local_path = Path(detector_url).expanduser()
    if local_path.is_file():
        return local_path

    # A nonstandard URL may still use an existing upstream-named cache file.
    # Runtime metric evaluation never fetches missing weights.
    cache_root = Path(cache_dir or os.environ.get('UNILIP_FVD_CACHE_DIR', './loaded_models'))
    url_md5 = hashlib.md5(detector_url.encode('utf-8')).hexdigest()
    cache_files = [Path(path) for path in glob.glob(str(cache_root / (url_md5 + '_*'))) if Path(path).is_file()]
    if len(cache_files) == 1:
        return cache_files[0]
    raise FileNotFoundError(
        f'FVD detector weights for {detector_url!r} are unavailable locally; '
        'runtime downloads are disabled'
    )


def open_url(url, num_attempts=10, verbose=False, cache_dir=None):
    assert num_attempts >= 1
    del verbose
    return open(_detector_path(url, cache_dir=cache_dir), 'rb')

# Load directly from the selected file so TorchScript sees its actual path.
def get_feature_detector(detector_url, device):
    detector_path = _detector_path(detector_url)
    key = (str(detector_path), str(device))
    if key not in _feature_detector_cache:
        _feature_detector_cache[key] = torch.jit.load(str(detector_path)).eval().to(device)
    return _feature_detector_cache[key]


"""
This function is used to first extract feature representation vectors of the videos using a pretrained model
Then the mean and covariance of the representation vectors are calculated and returned
"""
def compute_feature_stats(data, detector_url, detector_kwargs, batch_size, max_items, device):
    # if wanted reduce the number of elements used for calculating the FVD
    num_items = len(data)
    if max_items:
        num_items = min(num_items, max_items)
    data = data[:num_items]
    
    # load the pretrained feature extraction modeö
    detector = get_feature_detector(detector_url, device=device)

    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size)
    all_features = []
    for batch in loader:
        batch = batch[0]
        # if more than 3 channels are available we split the channel dimension into chunks of 3 and concatenate to batch dimension
        if batch.size(1) != 3:
            pad_size = 3 - (batch.size(1) % 3)
            pad = torch.zeros(batch.size(0), pad_size, batch.size(2), batch.size(3), batch.size(4), device=batch.device)
            batch = torch.cat([batch, pad], dim=1)
            batch = torch.cat(torch.chunk(batch, chunks=batch.size(1)//3, dim=1), dim=0)
        batch = batch.to(device)
        # extract feature vector using pretrained model
        features = detector(batch, **detector_kwargs)
        features = features.detach().cpu().numpy()
        all_features.append(features)
    # concatenate batches to one numpy array
    stacked_features = np.concatenate(all_features, axis=0)

    # calculate mean and covariance matrix across the extracted features
    mu = np.mean(stacked_features, axis=0)
    sigma = np.cov(stacked_features, rowvar=False)

    return mu, sigma

"""
This function calculates the Frechet Video Distance of two tensors representing a collection of videos
The input tensors should have shape num_videos x channels x num_frames x width x height
As the calculation of frechet video distance can be expensive max_items can be defined to estimate FVD on a subset
"""
def compute_fvd(y_true: torch.Tensor, y_pred: torch.Tensor, max_items: int, device: torch.device, batch_size: int):
    # URL from StyleGAN-V repository!
    detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=True, resize=True, return_features=True) # Return raw features before the softmax layer.

    # calculate the mean and covariance matrix of the representation vectors for ground truth and predicted videos
    mu_true, sigma_true = compute_feature_stats(y_true, detector_url, detector_kwargs, batch_size, max_items, device)
    mu_pred, sigma_pred = compute_feature_stats(y_pred, detector_url, detector_kwargs, batch_size, max_items, device)

    # FVD is calculated as the mahalanobis distance between the representation vector statistics
    m = np.square(mu_pred - mu_true).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_pred, sigma_true), disp=False)
    fvd = np.real(m + np.trace(sigma_pred + sigma_true - s * 2))
    return float(fvd)
