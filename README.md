# CSGO Benchmark v2 independent shared evaluator

This directory is the canonical, model-independent evaluator for the CSGO
Benchmark v2 Seen-10 tasks. The example workstation copy is at
`/home/jiahao/task/csgo_benchmark_v2_eval_general`; existing project-local
`csgo_benchmark_v2_eval` directories are retained as read-only copies. Set
`SHARED_EVAL_DIR` to the absolute path of this directory on your server:

```bash
SHARED_EVAL_DIR="${SHARED_EVAL_DIR:-/home/jiahao/task/csgo_benchmark_v2_eval_general}"
EVAL_PYTHON="$SHARED_EVAL_DIR/.venv/bin/python"
DATA_ROOT="${DATA_ROOT:-/home/jiahao/task/UniLIP/data/csgo_benchmark_v2}"
```

Install the evaluator once, explicitly, from any working directory:

```bash
bash "$SHARED_EVAL_DIR/setup_env.sh"
"$EVAL_PYTHON" --version
```

The setup script creates its own Python 3.11 environment at `.venv`. If
`python3.11` is unavailable on `PATH`, it bootstraps a separate Python 3.11
runtime into `.python311` using conda. It never uses the UniLIP or OpenVLA
environment implicitly. It installs `torch==2.7.1` and
`torchvision==0.22.1` CPU wheels by default, then the pinned packages in
`requirements.txt`. To select CUDA 12.8 wheels explicitly, run
`CSGO_EVAL_TORCH_BACKEND=cu128 bash "$SHARED_EVAL_DIR/setup_env.sh"`.
Re-running setup is safe; the backend must be selected explicitly on each
setup invocation. The installed backend and exact versions are recorded in
`.venv/install-manifest.json` and `.venv/pip-freeze.txt`.

Every project can call `"$SHARED_EVAL_DIR/.venv/bin/python"` directly. Running
the evaluator never installs or verifies packages.

For the target server path, install after copying this evaluator directory:

```bash
cd /home/user/yc57963/task/csgo_benchmark_v2_eval_general
bash setup_env.sh
EVAL_PYTHON="$PWD/.venv/bin/python"
```

Runtime code reads only `minimal_dataset_report.json`,
`benchmark_manifest.json`, the published split files, the published GT images
under `images/`, the mapped radar files under `radars/`, and
`calibration/z_calibration.json` under `DATA_ROOT`. It imports no UniLIP,
ControlAR, X-VLA, or other model code and never installs packages.
`requirements.txt` lists metric-only dependencies; `setup_env.sh` supplies the
pinned PyTorch/torchvision wheels in the isolated evaluator environment.

Localization input is a JSONL file named `predictions.jsonl`, or a directory
containing that file. Every row has `map_name`, `sample_id`, and
`pred_x`, `pred_y`, `pred_z`, `pred_pitch`, `pred_yaw`. `sample_id` accepts
either the bare `file_frame` (`file_num123_frame_0042`) or
`map/file_frame` (`de_dust2/file_num123_frame_0042`); when the map is included,
it must agree with `map_name`.

The default prediction pose space is normalized and ordered as
`[x, y, z, pitch, yaw]`. X and Y are physical coordinates divided by `1024`;
Z is `(physical_z - z_min) / (z_max - z_min)` using the published per-map
calibration without clipping; pitch and yaw are degrees divided by `360`.
Normalized values are finite numeric values and are not required to be in
`[0, 1]`; predictions are not clipped. Thus normalized X/Y/Z are dimensionless,
while the reported physical `XY_Dist` and `Z_Dist` are coordinate units and
`Pitch_Dist`/`Yaw_Dist` are degrees. Use `--pose-space physical` only when the
same prediction fields are already physical coordinates and degrees. Yaw uses
the shortest circular distance over a 360-degree period.

Generation input is a 448x448 RGB JPEG at
`<pred-root>/<map>/<file_frame>.jpg`; `<pred-root>` may be the `gen_imgs`
directory itself or its parent task output directory. The `.jpg` extension and
the exact manifest identity are part of formal coverage. The evaluator joins
identities from the published manifest rather than rediscovering a split from
the image tree.

The three task commands below use the real X-VLA localization output root and
the ControlAR generation roots. Choose a new, empty output directory for each
formal run:

```bash
"$EVAL_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" localization \
  --config "$SHARED_EVAL_DIR/benchmark_v2.yaml" \
  --pred-root "/home/jiahao/task/X-VLA/outputs/csgo_benchmark_v2_seen10/X-VLA/seed_0/localization" \
  --data-root "$DATA_ROOT" \
  --output "/home/jiahao/task/X-VLA/outputs/csgo_benchmark_v2_seen10/X-VLA/seed_0/metrics/shared_eval/localization"

"$EVAL_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" discrete \
  --config "$SHARED_EVAL_DIR/benchmark_v2.yaml" \
  --pred-root "/home/jiahao/task/ControlAR/outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0/discrete/gen_imgs" \
  --data-root "$DATA_ROOT" \
  --output "/home/jiahao/task/ControlAR/outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0/evaluation_shared/discrete"

"$EVAL_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" continuous \
  --config "$SHARED_EVAL_DIR/benchmark_v2.yaml" \
  --pred-root "/home/jiahao/task/ControlAR/outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0/continuous/gen_imgs" \
  --data-root "$DATA_ROOT" \
  --output "/home/jiahao/task/ControlAR/outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0/evaluation_shared/continuous"
```

The evaluator reports these metrics per map and in `summary_equal_map.json`:

| Task | Metrics |
| --- | --- |
| localization | `XY_Dist`, `Z_Dist`, `Pitch_Dist`, `Yaw_Dist` |
| discrete generation | `PSNR`, `SSIM`, `LPIPS`, `Boundary_F1`, `FID` |
| continuous generation | `PSNR`, `SSIM`, `LPIPS`, `TWE`, `TDE`, `FVD` |

The map macro is fixed: average the ten per-map values with equal weight in
the published Seen-10 order. Formal localization requires exactly every
`seen_discrete_test` identity. Formal generation requires complete exact image
coverage for every expected map/sample and rejects missing or extra images.
All maps and metrics finish before any formal result is written; a non-empty
output directory is rejected and existing result files are never overwritten.
Formal metrics must be finite.

Smoke commands read a limited prefix for localization/discrete and never write
formal per-map or summary files. Continuous smoke reads complete published
clips (`--max-clips`, default `1`) so temporal metrics can be inspected;
`--frame-only` reads only the first frame of the first clip and reports paired
PSNR/SSIM/LPIPS, skipping TWE/TDE and distribution metrics. Smoke output
contains `smoke_only: true`, and a diagnostic infinite PSNR is serialized as
the string `"Infinity"`; this does not make it valid for formal output.

```bash
"$EVAL_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" smoke localization \
  --config "$SHARED_EVAL_DIR/benchmark_v2.yaml" \
  --pred-root "/home/jiahao/task/X-VLA/outputs/csgo_benchmark_v2_seen10/X-VLA/seed_0/localization" \
  --data-root "$DATA_ROOT" --limit 1

"$EVAL_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" smoke discrete \
  --config "$SHARED_EVAL_DIR/benchmark_v2.yaml" \
  --pred-root "/home/jiahao/task/ControlAR/outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0/discrete/gen_imgs" \
  --data-root "$DATA_ROOT" --limit 1

"$EVAL_PYTHON" "$SHARED_EVAL_DIR/run_eval.py" smoke continuous \
  --config "$SHARED_EVAL_DIR/benchmark_v2.yaml" \
  --pred-root "/home/jiahao/task/ControlAR/outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0/continuous/gen_imgs" \
  --data-root "$DATA_ROOT" --frame-only
```

Continuous FVD uses the existing I3D weights. Cache selection follows
`--fvd-cache-dir` (CLI), then `UNILIP_FVD_CACHE_DIR`, then
`continuous.fvd_cache_dir` in the selected config, then
`$SHARED_EVAL_DIR/loaded_models`. Relative CLI and environment paths resolve
from the current working directory; a relative config value resolves from the
config file's directory. The checked-in config therefore resolves its default
to `/home/jiahao/task/UniLIP/loaded_models`, reusing the existing weights.
FID uses the installed torchmetrics/torch-fidelity weights.

`BenchmarkData(data_root).rows(split)` provides the manifest-driven data
contract for `seen_train`, `seen_validation`, `seen_discrete_test` and
`seen_continuous`; continuous rows retain `clip_id` and zero-based
`frame_index`, and `BenchmarkData(data_root).clips()` returns whole clips in
published order. Each row includes `sample_id`, `map_name`, `file_frame`,
absolute image/radar paths, normalized and raw pose, Z calibration, and
continuous identity fields.

The copied metric implementations preserve the benchmark defaults: paired
images are 448 pixels, continuous FVD windows are 16 frames with stride 16 at
224 pixels, frame-gap threshold 2, and minimum track length 4. TWE/TDE use
resized uint8 RGB values and the published flow/warp settings. See
`THIRD_PARTY_NOTICES.md` for attribution and `LICENSE` for the evaluator
license.

Existing compatibility evidence is kept under
`validation/compatibility/`: `localization_xvla/summary_equal_map.json` records
the complete X-VLA localization comparison, and
`canonical_metric_parity.json` records the six-metric canonical parity and I3D
cache load. These files are validation artifacts; generation smoke output is
not a formal benchmark result.

The synthetic parity check is reproducible with
`$EVAL_PYTHON "$SHARED_EVAL_DIR/validation/canonical_metric_parity.py"`.
Add `--run-fvd-smoke --device cuda` to exercise the copied FVD implementation
on two synthetic 16-frame 224x224 clips per side.
