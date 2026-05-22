"""Phase 2 encoder for las2veg-rgb.

Pipeline:
    preview_indicators.tif (6-band float32, ratios in [0,1])
        -> quantize each indicator to 16 steps (linear, value*15 rounded)
        -> pack pairs of 4-bit values into 8-bit channels (high<<4 | low)
        -> 4-band RGBA uint8 GeoTIFF

Packing layout (Plan D):
    R: high4 = density_z3       low4 = canopy_height_p95
    G: high4 = density_z2       low4 = occupancy_z2
    B: high4 = density_z1       low4 = occupancy_z1
    A: always 255 (unused for data)

Run:
    python scripts/phase2_encode.py \
        --input data/output/preview_indicators.tif \
        --output data/output/encoded_rgba.tif \
        [--verify]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import numpy as np
import rasterio
from rasterio.enums import ColorInterp

REPO_ROOT = Path(__file__).resolve().parents[1]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("phase2")

INDICATOR_BANDS: tuple[str, ...] = (
    "density_z1",
    "density_z2",
    "density_z3",
    "occupancy_z1",
    "occupancy_z2",
    "canopy_height_p95",
)

PACKING: tuple[tuple[str, str], ...] = (
    ("density_z3", "canopy_height_p95"),  # R
    ("density_z2", "occupancy_z2"),       # G
    ("density_z1", "occupancy_z1"),       # B
)

# 指標ごとの量子化スケール (= 「16段階で何の値域までを表現するか」)
# - occupancy: 実データが 0-0.4 程度に集中 → 0.5 で量子化 (1段階 0.033)
# - canopy_height_p95: Phase 1 では 0-1 (=0-100m) で正規化、Phase 2 で実用域を
#   0-0.6 (=0-60m) として量子化 (1段階 4m、世界 99.5% カバー、ユーカリ巨木の一部のみ clip)
# - density: フルレンジ 0-1.0 (1段階 0.067)
# >>> ブラウザ側のデコードもこれに合わせる: value = (encoded / 15) * MAX_VALUE
QUANTIZE_MAX: dict[str, float] = {
    "density_z1":        1.0,
    "density_z2":        1.0,
    "density_z3":        1.0,
    "occupancy_z1":      0.5,
    "occupancy_z2":      0.5,
    "canopy_height_p95": 0.6,
}


def quantize_to_4bit(values: np.ndarray, max_value: float = 1.0) -> np.ndarray:
    """Map float [0, max_value] to uint8 [0, 15] via uniform-width 16-bin quantization.

    Each bin N covers the half-open range [N/16, (N+1)/16) * max_value.
    Bin 15 also captures the closed endpoint (value == max_value).
    Values outside [0, max_value] are clipped. NaN becomes 0.

    Decode side should use: value = (encoded + 0.5) / 16 * max_value  (bin center)
                       or: value =  encoded       / 16 * max_value  (bin lower bound)
    """
    clipped = np.clip(np.nan_to_num(values, nan=0.0), 0.0, max_value)
    scaled = clipped / max_value * 16.0
    return np.minimum(np.floor(scaled), 15.0).astype(np.uint8)


def pack_4bit_pair(high4: np.ndarray, low4: np.ndarray) -> np.ndarray:
    """Pack two 4-bit arrays into a single uint8: (high<<4) | (low & 0x0F)."""
    return ((high4 & 0x0F) << 4 | (low4 & 0x0F)).astype(np.uint8)


def unpack_4bit_pair(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of pack_4bit_pair. Returns (high4, low4) as uint8 arrays."""
    high4 = (packed >> 4) & 0x0F
    low4 = packed & 0x0F
    return high4, low4


def read_indicators(input_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read a 6-band float32 GeoTIFF and return per-indicator arrays + profile."""
    with rasterio.open(input_path) as src:
        if src.count != len(INDICATOR_BANDS):
            raise click.ClickException(
                f"Expected {len(INDICATOR_BANDS)} bands in {input_path}, "
                f"got {src.count}. Re-run phase1_preview.py to regenerate."
            )
        descriptions = list(src.descriptions)
        if descriptions != list(INDICATOR_BANDS):
            log.warning(
                "Band descriptions %s differ from expected %s. "
                "Assuming positional order anyway.",
                descriptions,
                list(INDICATOR_BANDS),
            )
        data = {
            name: src.read(idx + 1).astype(np.float32)
            for idx, name in enumerate(INDICATOR_BANDS)
        }
        profile = src.profile
    return data, profile


def write_rgba(out_path: Path, rgba: np.ndarray, profile: dict) -> None:
    """Write a 4-band uint8 GeoTIFF preserving CRS/transform from profile."""
    out_profile = profile.copy()
    out_profile.update(
        dtype="uint8",
        count=4,
        nodata=None,
        compress="deflate",
        photometric="RGB",
    )
    with rasterio.open(out_path, "w", **out_profile) as dst:
        for i in range(4):
            dst.write(rgba[i], i + 1)
        dst.colorinterp = [
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        ]
        for i, name in enumerate(["R_z3_chp95", "G_z2_occ2", "B_z1_occ1", "A_unused"]):
            dst.set_band_description(i + 1, name)


def encode(indicators: dict[str, np.ndarray]) -> np.ndarray:
    """Build a (4, H, W) uint8 RGBA array from indicator dict.

    Each indicator is quantized with its own QUANTIZE_MAX (most are 1.0,
    occupancy_z1/z2 use 0.5 to better fit observed data distribution).
    """
    height, width = indicators["density_z1"].shape
    rgba = np.zeros((4, height, width), dtype=np.uint8)

    quantized = {
        name: quantize_to_4bit(arr, QUANTIZE_MAX[name])
        for name, arr in indicators.items()
    }

    for ch_idx, (high_name, low_name) in enumerate(PACKING):
        rgba[ch_idx] = pack_4bit_pair(quantized[high_name], quantized[low_name])

    rgba[3] = 255  # A channel unused
    return rgba


def verify_roundtrip(
    indicators: dict[str, np.ndarray], rgba: np.ndarray
) -> dict[str, dict[str, float]]:
    """Decode the RGBA back to ratios and report per-indicator round-trip stats.

    Decoding uses bin center: value = (encoded + 0.5) / 16 * quantize_max.
    Each bin N covers [N/16, (N+1)/16) * quantize_max, so the center is
    the unbiased reconstruction.
    """
    decoded: dict[str, np.ndarray] = {}
    for ch_idx, (high_name, low_name) in enumerate(PACKING):
        high4, low4 = unpack_4bit_pair(rgba[ch_idx])
        decoded[high_name] = (high4.astype(np.float32) + 0.5) / 16.0 * QUANTIZE_MAX[high_name]
        decoded[low_name]  = (low4.astype(np.float32) + 0.5) / 16.0 * QUANTIZE_MAX[low_name]

    stats: dict[str, dict[str, float]] = {}
    for name in INDICATOR_BANDS:
        # クリップ範囲外の値は復元誤差として大きく出るので、フェアな比較のため
        # 元値も QUANTIZE_MAX で clip してから差を取る
        original_clipped = np.clip(indicators[name], 0.0, QUANTIZE_MAX[name])
        recovered = decoded[name]
        diff = np.abs(original_clipped - recovered)
        # 半 bin 幅 = (1/16)/2 = 1/32 が理論最大誤差 (bin 中央復号 + 端を除く)
        max_step_error = QUANTIZE_MAX[name] / 32.0
        stats[name] = {
            "quantize_max": QUANTIZE_MAX[name],
            "expected_max_error": max_step_error,
            "max_abs_error": float(diff.max()),
            "mean_abs_error": float(diff.mean()),
            "p95_abs_error": float(np.percentile(diff, 95)),
        }
    return stats


@click.command()
@click.option(
    "--input",
    "input_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input multi-band float32 GeoTIFF from Phase 1. "
         "Defaults to data/output/latest/preview_indicators.tif.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output 4-band uint8 RGBA GeoTIFF. "
         "Defaults to <input-dir>/encoded_rgba.tif.",
)
@click.option(
    "--verify",
    is_flag=True,
    help="After encoding, decode the result and report round-trip errors.",
)
def main(input_path: Path | None, output_path: Path | None, verify: bool) -> None:
    """Phase 2: pack 6 ratio indicators into a 4-bit RGBA GeoTIFF."""
    # Resolve defaults from Phase 1 run directory
    if input_path is None:
        # Lazy import to avoid hard dependency on src/ path
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from las2veg_rgb.runs import find_latest_run  # noqa: E402

        latest = find_latest_run(Path("data/output"))
        if latest is None:
            raise click.ClickException(
                "No run directory found under data/output/. "
                "Run phase1_preview.py first or pass --input explicitly."
            )
        input_path = latest / "preview_indicators.tif"
        if not input_path.exists():
            raise click.ClickException(
                f"Expected {input_path} but it does not exist. "
                "Run phase1_preview.py first."
            )
        log.info("Auto-detected input from latest run: %s", input_path)

    if output_path is None:
        output_path = input_path.parent / "encoded_rgba.tif"
        log.info("Auto-detected output path: %s", output_path)

    log.info("Reading indicators from %s", input_path)
    indicators, profile = read_indicators(input_path)
    log.info("Loaded shape=%s (H, W)", indicators["density_z1"].shape)

    log.info("Quantizing to 16 steps and packing into RGBA...")
    log.info("  per-indicator quantize_max:")
    for name in INDICATOR_BANDS:
        log.info("    %-20s -> max %.2f (1 step = %.4f)",
                 name, QUANTIZE_MAX[name], QUANTIZE_MAX[name] / 15.0)
    rgba = encode(indicators)

    log.info("Writing RGBA GeoTIFF to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_rgba(output_path, rgba, profile)

    if verify:
        log.info("Verifying round-trip (encode -> decode -> compare)...")
        stats = verify_roundtrip(indicators, rgba)
        log.info("Round-trip error stats (expected: max_abs <= quantize_max/30):")
        for name, st in stats.items():
            log.info(
                "  %-18s qmax=%.2f exp<=%.4f  max=%.4f  mean=%.4f  p95=%.4f",
                name,
                st["quantize_max"],
                st["expected_max_error"],
                st["max_abs_error"],
                st["mean_abs_error"],
                st["p95_abs_error"],
            )

    log.info("Done.")


if __name__ == "__main__":
    main()
