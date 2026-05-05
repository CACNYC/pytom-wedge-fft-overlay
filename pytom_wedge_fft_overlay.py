#!/usr/bin/env python3
"""
pytom_wedge_fft_overlay.py

Version: 2026-05-05a, developer-review label cleanup.

Visualize the PyTom binary tomogram missing-wedge model and compare it, in the same
array/display convention, to the XZ Fourier log-magnitude of the tomogram.

This script reproduces the binary tomogram wedge that TMJob applies to the tomogram
filter with per_tilt_weighting=False. It does not reproduce the full per-tilt/CTF/dose
template wedge used for template modulation when --per-tilt-weighting is enabled.

The primary goal is an apples-to-apples orientation check:
  1. Read the tomogram exactly as PyTom reads it: pytom_tm.io.read_mrc()
     (MRC zyx storage -> PyTom internal xyz array).
  2. Pad to the same real-FFT-friendly shape as pytom_match_template.py.
  3. Build the PyTom binary tomogram wedge using pytom_tm.weights.create_wedge()
     with per_tilt_weighting=False, as TMJob does for the tomogram filter.
  4. Compute the tomogram rFFT in the same PyTom-oriented array.
  5. Put both objects in the same reduced-Fourier display convention:
       fftshift axes x and y; keep the reduced +z Fourier half-axis unreversed.
  6. Extract the same central XZ slab and overlay the wedge contour on the FFT.

Run example:
  python pytom_wedge_fft_overlay.py \
    --tomogram path/to/tomo.mrc \
    --tilt-angles path/to/angles.tlt \
    --voxel-size-angstrom 10.371 \
    --outdir wedge_check_runA \
    --slab-y 7 \
    --also-negated

For a direct Run A vs Run B test, either run the script twice with the two .tlt
files, or run once with --also-negated.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Iterable, Optional

import numpy as np

# Use a headless backend so the script works on cluster/login nodes.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import next_fast_len, rfftn


def _prepend_pytom_source_if_requested(pytom_src: Optional[pathlib.Path]) -> None:
    """Allow using an unpacked source tree without installing it."""
    if pytom_src is None:
        return
    pytom_src = pytom_src.resolve()
    candidate_src = pytom_src / "src"
    if candidate_src.is_dir():
        sys.path.insert(0, str(candidate_src))
    else:
        sys.path.insert(0, str(pytom_src))


def _import_pytom(pytom_src: Optional[pathlib.Path] = None):
    """Import the exact PyTom functions used by this tool."""
    _prepend_pytom_source_if_requested(pytom_src)
    try:
        from pytom_tm.io import read_mrc, read_tlt_file
        from pytom_tm.dataclass import TiltSeriesMetaData
        from pytom_tm.weights import create_wedge, radial_reduced_grid
    except Exception as exc:  # pragma: no cover - useful diagnostic on user systems
        raise SystemExit(
            "Could not import pytom_tm. Activate the same conda environment used for "
            "pytom_match_template.py, or pass --pytom-src /path/to/pytom-match-pick.\n"
            f"Import error: {exc}"
        ) from exc
    return read_mrc, read_tlt_file, TiltSeriesMetaData, create_wedge, radial_reduced_grid


def _odd_int(value: str) -> int:
    """Parse a positive odd integer for argparse.

    argparse passes option values as strings before type conversion.  The first
    version of this script compared that string directly to an integer, which
    made valid values such as --slab-y 7 fail argparse validation.
    """
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("slab width must be an integer") from exc
    if n < 1:
        raise argparse.ArgumentTypeError("slab width must be >= 1")
    if n % 2 != 1:
        raise argparse.ArgumentTypeError("slab width must be odd so it is centered")
    return n


def _parse_center_crop(spec: Optional[str]) -> Optional[tuple[int, int, int]]:
    if spec is None:
        return None
    parts = spec.lower().replace(",", "x").split("x")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--crop-center must look like 512x512x512")
    vals = tuple(int(p) for p in parts)
    if any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("--crop-center dimensions must be positive")
    return vals  # type: ignore[return-value]


def _center_crop_xyz(arr: np.ndarray, target: Optional[tuple[int, int, int]]) -> np.ndarray:
    """Crop a PyTom internal xyz array around its center."""
    if target is None:
        return arr
    slices = []
    for n, t in zip(arr.shape, target):
        t = min(t, n)
        start = (n - t) // 2
        slices.append(slice(start, start + t))
    return np.ascontiguousarray(arr[tuple(slices)])


def _stride_xyz(arr: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return arr
    return np.ascontiguousarray(arr[::stride, ::stride, ::stride])


def _sanitize_nonfinite(arr: np.ndarray, policy: str) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Handle NaN/Inf before FFT display/scoring.

    A single NaN in real space can make the entire FFT NaN.  PyTom's MRC reader is
    still used first; this optional cleanup is only for the diagnostic FFT view.
    """
    finite_mask = np.isfinite(arr)
    n_total = int(arr.size)
    n_nonfinite = int(n_total - int(finite_mask.sum()))
    info: dict[str, float | int | str] = {
        "policy": policy,
        "total_voxels": n_total,
        "nonfinite_voxels": n_nonfinite,
        "nonfinite_fraction": float(n_nonfinite / n_total) if n_total else 0.0,
        "replacement_value": 0.0,
    }
    if n_nonfinite == 0 or policy == "keep":
        return arr, info

    out = np.asarray(arr, dtype=np.float32).copy()
    if policy == "zero":
        replacement = 0.0
    elif policy == "mean":
        replacement = float(np.mean(out[finite_mask])) if finite_mask.any() else 0.0
    else:  # argparse should prevent this
        raise ValueError(f"Unknown nan policy: {policy}")
    out[~finite_mask] = replacement
    info["replacement_value"] = float(replacement)
    return out, info


def _pad_like_pytom(tomo_xyz: np.ndarray) -> np.ndarray:
    """Pad as pytom_tm.tmjob.TMJob.start_job does for fast_tomo."""
    fast_shape = tuple(next_fast_len(int(s), real=True) for s in tomo_xyz.shape)
    fast = np.zeros(fast_shape, dtype=np.float32)
    fast[: tomo_xyz.shape[0], : tomo_xyz.shape[1], : tomo_xyz.shape[2]] = tomo_xyz.astype(
        np.float32, copy=False
    )
    return fast


def _central_indices(n: int, slab_width: int) -> slice:
    width = min(slab_width, n)
    start = max(0, n // 2 - width // 2)
    stop = min(n, start + width)
    start = max(0, stop - width)
    return slice(start, stop)


def _xz_slab_reduced(arr_reduced_shifted_xy: np.ndarray, slab_y: int) -> np.ndarray:
    """Mean XZ slab from array with shape x,y,z_reduced."""
    ys = _central_indices(arr_reduced_shifted_xy.shape[1], slab_y)
    return np.mean(arr_reduced_shifted_xy[:, ys, :], axis=1)


def _robust_limits(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> tuple[float, float]:
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
        if hi <= lo:
            hi = lo + 1.0
    return float(lo), float(hi)


def _read_angles_text(path: pathlib.Path) -> list[float]:
    # Fallback reader used only for summary if PyTom parser is unavailable elsewhere.
    vals: list[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals.append(float(line.split()[0]))
    return vals


def _save_png_matrix(
    img: np.ndarray,
    out: pathlib.Path,
    title: str,
    xlabel: str = "+kz, reduced Fourier half-axis",
    ylabel: str = "kx, fftshifted",
    cmap: str = "gray",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    if vmin is None or vmax is None:
        vmin, vmax = _robust_limits(img)
    im = ax.imshow(img, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _save_overlay(
    fft_xz: np.ndarray,
    wedge_xz: np.ndarray,
    out: pathlib.Path,
    title: str,
    threshold: float = 0.5,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    vmin, vmax = _robust_limits(fft_xz)
    im = ax.imshow(fft_xz, origin="lower", aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
    # contour at the transition between sampled and missing wedge.
    try:
        ax.contour(wedge_xz, levels=[threshold], origin="lower", linewidths=1.25)
    except Exception:
        # If a contour cannot be found, still write the FFT panel.
        pass
    ax.set_title(title)
    ax.set_xlabel("+kz, reduced Fourier half-axis")
    ax.set_ylabel("kx, fftshifted")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out, dpi=200)
    plt.close(fig)


def _score_wedge_orientation(
    fft_logmag_xz: np.ndarray,
    wedge_xz: np.ndarray,
    radius_xz: np.ndarray,
    threshold: float = 0.5,
    freq_min: float = 0.05,
    freq_max: float = 0.95,
) -> dict[str, float]:
    """
    Simple orientation score: if wedge_xz is aligned, the tomogram power should be
    lower where PyTom says the wedge is missing than where PyTom says it is sampled.

    This is a heuristic, not a proof. It works best for asymmetric/one-sided data and
    reasonably masked/cropped tomograms.
    """
    shell = (radius_xz >= freq_min) & (radius_xz <= freq_max)
    sampled = shell & (wedge_xz >= threshold)
    missing = shell & (wedge_xz < threshold)
    # Use log magnitude because raw FFT magnitude is dominated by low frequencies and outliers.
    sampled_vals_raw = fft_logmag_xz[sampled]
    missing_vals_raw = fft_logmag_xz[missing]
    sampled_vals = sampled_vals_raw[np.isfinite(sampled_vals_raw)]
    missing_vals = missing_vals_raw[np.isfinite(missing_vals_raw)]
    result = {
        "threshold": float(threshold),
        "freq_min": float(freq_min),
        "freq_max": float(freq_max),
        "sampled_n": int(sampled_vals_raw.size),
        "missing_n": int(missing_vals_raw.size),
        "sampled_finite_n": int(sampled_vals.size),
        "missing_finite_n": int(missing_vals.size),
        "sampled_mean_log_magnitude": float(np.mean(sampled_vals)) if sampled_vals.size else math.nan,
        "missing_mean_log_magnitude": float(np.mean(missing_vals)) if missing_vals.size else math.nan,
        "sampled_minus_missing_mean_log_magnitude": math.nan,
        "sampled_median_log_magnitude": float(np.median(sampled_vals)) if sampled_vals.size else math.nan,
        "missing_median_log_magnitude": float(np.median(missing_vals)) if missing_vals.size else math.nan,
        "sampled_minus_missing_median_log_magnitude": math.nan,
    }
    if sampled_vals.size and missing_vals.size:
        result["sampled_minus_missing_mean_log_magnitude"] = (
            result["sampled_mean_log_magnitude"] - result["missing_mean_log_magnitude"]
        )
        result["sampled_minus_missing_median_log_magnitude"] = (
            result["sampled_median_log_magnitude"] - result["missing_median_log_magnitude"]
        )
    return result


@dataclass
class AngleSummary:
    label: str
    n_angles: int
    first_angle: float
    last_angle: float
    min_angle: float
    max_angle: float
    sampled_range_degrees: str
    approximate_missing_wedge_opening_degrees: float
    approximate_missing_wedge_center_degrees: float


def _angle_summary(label: str, angles: Iterable[float]) -> AngleSummary:
    a = np.asarray(list(angles), dtype=float)
    amin = float(np.min(a))
    amax = float(np.max(a))
    return AngleSummary(
        label=label,
        n_angles=int(a.size),
        first_angle=float(a[0]),
        last_angle=float(a[-1]),
        min_angle=amin,
        max_angle=amax,
        sampled_range_degrees=f"{amin:g} to {amax:g}",
        approximate_missing_wedge_opening_degrees=float(180.0 - (amax - amin)),
        approximate_missing_wedge_center_degrees=float(90.0 + (amin + amax) / 2.0),
    )


def _make_one_wedge_case(
    label: str,
    angles: list[float],
    fast_shape: tuple[int, int, int],
    voxel_size: float,
    TiltSeriesMetaData,
    create_wedge,
    fft_xz: np.ndarray,
    radius_xz: np.ndarray,
    outdir: pathlib.Path,
    slab_y: int,
    threshold: float,
    freq_min: float,
    freq_max: float,
) -> dict:
    meta = TiltSeriesMetaData(tilt_angles=[float(a) for a in angles])
    wedge_reduced = create_wedge(
        fast_shape,
        meta,
        voxel_size,
        cut_off_radius=1.0,
        per_tilt_weighting=False,
    ).astype(np.float32)

    # Same display convention as the tomogram FFT: shift x and y only.
    wedge_view = np.fft.fftshift(wedge_reduced, axes=(0, 1))
    wedge_xz = _xz_slab_reduced(wedge_view, slab_y=slab_y)

    _save_png_matrix(
        wedge_xz,
        outdir / f"{label}_pytom_binary_wedge_xz.png",
        f"{label}: PyTom binary tomogram wedge, XZ slab",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    _save_overlay(
        fft_xz,
        wedge_xz,
        outdir / f"{label}_overlay_fft_with_wedge_contour_xz.png",
        f"{label}: tomogram FFT XZ with PyTom binary wedge contour",
        threshold=threshold,
    )
    np.save(outdir / f"{label}_wedge_xz.npy", wedge_xz)
    score = _score_wedge_orientation(
        fft_xz,
        wedge_xz,
        radius_xz,
        threshold=threshold,
        freq_min=freq_min,
        freq_max=freq_max,
    )
    return {
        "angle_summary": asdict(_angle_summary(label, angles)),
        "orientation_score": score,
        "files": {
            "wedge_xz_png": str(outdir / f"{label}_pytom_binary_wedge_xz.png"),
            "overlay_png": str(outdir / f"{label}_overlay_fft_with_wedge_contour_xz.png"),
            "wedge_xz_npy": str(outdir / f"{label}_wedge_xz.npy"),
        },
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Create apples-to-apples XZ overlays of a tomogram FFT and the PyTom "
            "binary missing-wedge model generated from a .tlt file."
        )
    )
    p.add_argument("--tomogram", required=True, type=pathlib.Path, help="Input tomogram MRC")
    p.add_argument("--tilt-angles", required=True, type=pathlib.Path, help=".tlt/.rawtlt file")
    p.add_argument("--voxel-size-angstrom", required=True, type=float, help="Tomogram voxel size in Å")
    p.add_argument("--outdir", required=True, type=pathlib.Path, help="Output directory")
    p.add_argument(
        "--pytom-src",
        type=pathlib.Path,
        default=None,
        help="Optional path to pytom-match-pick source tree or its src directory",
    )
    p.add_argument(
        "--slab-y",
        type=_odd_int,
        default=7,
        help="Number of central PyTom-Y Fourier planes to average for XZ view; default 7",
    )
    p.add_argument(
        "--crop-center",
        type=_parse_center_crop,
        default=None,
        help="Optional center crop after PyTom MRC read, e.g. 512x512x512",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Optional integer stride after crop for memory-saving orientation checks; default 1",
    )
    p.add_argument(
        "--also-negated",
        action="store_true",
        help="Also generate the model for -1 × the supplied tilt angles",
    )
    p.add_argument(
        "--wedge-threshold",
        type=float,
        default=0.5,
        help="Contour / sampled-vs-missing threshold for wedge values; default 0.5",
    )
    p.add_argument(
        "--score-freq-min",
        type=float,
        default=0.05,
        help="Minimum normalized Fourier radius for scoring; 0=center, 1=Nyquist; default 0.05",
    )
    p.add_argument(
        "--score-freq-max",
        type=float,
        default=0.90,
        help="Maximum normalized Fourier radius for scoring; default 0.90",
    )
    p.add_argument(
        "--nan-policy",
        choices=("zero", "mean", "keep"),
        default="zero",
        help=(
            "How to handle NaN/Inf voxels before computing the diagnostic FFT. "
            "A single NaN makes the entire FFT NaN. Default: zero."
        ),
    )
    p.add_argument(
        "--save-npy",
        action="store_true",
        help="Save tomogram FFT XZ and radius XZ arrays as .npy in addition to PNGs",
    )
    args = p.parse_args(argv)

    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    if args.voxel_size_angstrom <= 0:
        raise SystemExit("--voxel-size-angstrom must be > 0")

    args.outdir.mkdir(parents=True, exist_ok=True)

    read_mrc, read_tlt_file, TiltSeriesMetaData, create_wedge, radial_reduced_grid = _import_pytom(args.pytom_src)

    print("Reading tomogram exactly via pytom_tm.io.read_mrc() ...", flush=True)
    tomo_xyz = read_mrc(args.tomogram)
    original_shape = tuple(int(s) for s in tomo_xyz.shape)
    print(f"PyTom internal xyz tomogram shape: {original_shape}", flush=True)

    if args.crop_center is not None:
        tomo_xyz = _center_crop_xyz(tomo_xyz, args.crop_center)
        print(f"After center crop: {tuple(tomo_xyz.shape)}", flush=True)
    if args.stride > 1:
        tomo_xyz = _stride_xyz(tomo_xyz, args.stride)
        print(f"After stride {args.stride}: {tuple(tomo_xyz.shape)}", flush=True)

    tomo_xyz, nonfinite_info = _sanitize_nonfinite(tomo_xyz, args.nan_policy)
    if nonfinite_info["nonfinite_voxels"]:
        print(
            "Non-finite tomogram voxels detected before FFT: "
            f"{nonfinite_info['nonfinite_voxels']} / {nonfinite_info['total_voxels']} "
            f"({nonfinite_info['nonfinite_fraction']:.3g}); "
            f"nan-policy={args.nan_policy}, replacement={nonfinite_info['replacement_value']}",
            flush=True,
        )

    effective_voxel_size = float(args.voxel_size_angstrom) * int(args.stride)
    fast_tomo = _pad_like_pytom(tomo_xyz)
    fast_shape = tuple(int(s) for s in fast_tomo.shape)
    print(f"PyTom-style fast FFT shape: {fast_shape}", flush=True)

    print("Reading tilt angles via pytom_tm.io.read_tlt_file() ...", flush=True)
    angles = [float(a) for a in read_tlt_file(args.tilt_angles)]
    if len(angles) < 2:
        raise SystemExit("Need at least two tilt angles to create a wedge")
    print(f"Read {len(angles)} angles: first={angles[0]:g}, last={angles[-1]:g}, min={min(angles):g}, max={max(angles):g}", flush=True)

    print("Computing tomogram rFFT in the same PyTom-oriented padded array ...", flush=True)
    fft_reduced = rfftn(fast_tomo)
    # Log magnitude/power. This is for display and scoring only.
    fft_log = np.log1p(np.abs(fft_reduced).astype(np.float64))
    fft_view = np.fft.fftshift(fft_log, axes=(0, 1))
    fft_xz = _xz_slab_reduced(fft_view, slab_y=args.slab_y)
    _save_png_matrix(
        fft_xz,
        args.outdir / "tomogram_fft_log_magnitude_xz.png",
        "Tomogram rFFT log magnitude, PyTom-oriented XZ slab",
        cmap="gray",
    )

    # Radius grid for the same display/slice convention; used only for scoring mask.
    # PyTom's radial_reduced_grid() is already centered in x and y.  Unlike the
    # rFFT data and wedge returned by create_wedge(), it should NOT be fftshifted
    # here.  Shifting it moves high frequencies into the central-y slab and can
    # make the scoring shell empty, yielding NaN scores.
    radius = radial_reduced_grid(fast_shape)
    radius_xz = _xz_slab_reduced(radius, slab_y=args.slab_y)

    if args.save_npy:
        np.save(args.outdir / "tomogram_fft_log_magnitude_xz.npy", fft_xz)
        np.save(args.outdir / "radius_xz.npy", radius_xz)

    cases = {}
    cases["input_angles"] = _make_one_wedge_case(
        "input_angles",
        angles,
        fast_shape,
        effective_voxel_size,
        TiltSeriesMetaData,
        create_wedge,
        fft_xz,
        radius_xz,
        args.outdir,
        args.slab_y,
        args.wedge_threshold,
        args.score_freq_min,
        args.score_freq_max,
    )

    if args.also_negated:
        neg_angles = [-a for a in angles]
        cases["negated_angles"] = _make_one_wedge_case(
            "negated_angles",
            neg_angles,
            fast_shape,
            effective_voxel_size,
            TiltSeriesMetaData,
            create_wedge,
            fft_xz,
            radius_xz,
            args.outdir,
            args.slab_y,
            args.wedge_threshold,
            args.score_freq_min,
            args.score_freq_max,
        )

    summary = {
        "script": pathlib.Path(__file__).name,
        "tomogram": str(args.tomogram),
        "tilt_angles_file": str(args.tilt_angles),
        "voxel_size_angstrom_input": float(args.voxel_size_angstrom),
        "effective_voxel_size_angstrom_after_stride": float(effective_voxel_size),
        "original_pytom_xyz_shape": original_shape,
        "working_xyz_shape_after_crop_stride": tuple(int(s) for s in tomo_xyz.shape),
        "nonfinite_tomogram_handling": nonfinite_info,
        "fast_fft_shape": fast_shape,
        "slab_y": int(args.slab_y),
        "display_convention": (
            "Reduced real-FFT volume with x and y fftshifted; +kz is the unreversed "
            "reduced Fourier half-axis. Tomogram FFT and PyTom wedge use the same convention."
        ),
        "tomogram_fft_png": str(args.outdir / "tomogram_fft_log_magnitude_xz.png"),
        "cases": cases,
        "interpretation_hint": (
            "For each case, a larger sampled_minus_missing_* score means the PyTom-modeled "
            "sampled region has higher tomogram FFT log-magnitude than the modeled missing region. "
            "This is a heuristic orientation diagnostic, not a proof. Compare overlays visually."
        ),
    }
    with open(args.outdir / "wedge_fft_overlay_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nWrote:")
    print(f"  {args.outdir / 'tomogram_fft_log_magnitude_xz.png'}")
    for name, case in cases.items():
        print(f"  {case['files']['wedge_xz_png']}")
        print(f"  {case['files']['overlay_png']}")
    print(f"  {args.outdir / 'wedge_fft_overlay_summary.json'}")
    print("\nOrientation scores:")
    for name, case in cases.items():
        score = case["orientation_score"]
        print(
            f"  {name}: sampled-minus-missing mean log magnitude = "
            f"{score['sampled_minus_missing_mean_log_magnitude']:.6g}; "
            f"median = {score['sampled_minus_missing_median_log_magnitude']:.6g}; "
            f"sampled_n = {score['sampled_n']}; missing_n = {score['missing_n']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
