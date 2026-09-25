# Shared evaluator compatibility acceptance

Verified on 2026-09-16 with `/home/jiahao/miniconda3/envs/UniLIP/bin/python`.
The two project-local evaluator directories were read-only references. All 36
files, including existing bytecode files, matched the pre-change SHA-256
snapshot after acceptance. The published dataset was not modified or copied.

## Localization

The real X-VLA prediction file contains 20,000 samples, 2,000 for each Seen-10
map. Evaluation through the shared CLI produced exactly the same per-map and
equal-map metric values as the existing X-VLA results. The new result is
[`localization_xvla/summary_equal_map.json`](localization_xvla/summary_equal_map.json).

Both bare frame IDs and `map/frame` IDs produced identical metrics. Missing,
duplicate, non-finite, and invalid-map predictions were rejected. Multi-turn
yaw used the shortest circular distance. The actual X-VLA wrapper also
evaluated all 20,000 predictions successfully with an unavailable model
Python, using only the metric environment; a repeat attempt refused to
overwrite the result. That wrapper check used a temporary result directory.

## Generation

The real ControlAR smoke predictions were read directly from
`/home/jiahao/task/ControlAR/outputs/csgo_benchmark_v2_smoke/ControlAR/seed_0`.
On the same CPU environment, the old and new evaluators returned identical
metrics for both tasks:

- Discrete: PSNR, SSIM, LPIPS, Boundary_F1 on
  `discrete/gen_imgs/cs_agency/file_num68_frame_421.jpg`.
- Continuous frame-only: PSNR, SSIM, LPIPS on
  `continuous/gen_imgs/cs_agency/file_num3_frame_205.jpg`.

These checks are nonformal smoke results; a single frame cannot provide
temporal or distribution metrics. Both formal generation commands rejected
these incomplete prediction sets without creating result directories.

The synthetic canonical comparison in
[`canonical_metric_parity.json`](canonical_metric_parity.json) returned zero
absolute difference for PSNR, SSIM, Boundary_F1, FID, TWE and TDE. FVD
calculation functions were AST-identical to the reference, and the cached
I3D TorchScript model loaded successfully. No new full generation benchmark
or FVD forward score is claimed by this acceptance.

## Integration

- Both wrappers route formal and smoke evaluation to the shared directory.
- Default paths and `SHARED_EVAL_DIR` overrides were checked.
- Evaluation works without the model Python; missing evaluators fail before
  smoke training begins.
- FVD cache resolution was checked for CLI, environment, and config paths.
  The relocated default resolves to `/home/jiahao/task/UniLIP/loaded_models`
  and finds the existing I3D weights.
- Config-relative cache paths resolve from the selected config directory;
  CLI/environment relative paths resolve from the working directory.
- Nonempty result directories are rejected before metric computation.
- Shell syntax checks and in-memory Python compilation passed.
