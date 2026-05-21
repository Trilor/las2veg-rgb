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
from pathlib import Path

import click
import numpy as np
import rasterio
from rasterio.enums import ColorInterp

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


def quantize_to_4bit(values: np.ndarray) -> np.ndarray:
    """Map float [0, 1] to uint8 [0, 15] via linear 16-step quantization.

    Values outside [0, 1] are clipped. NaN becomes 0.
    """
    clipped = np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)
    return np.round(clipped * 15.0).astype(np.uint8)


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
    """Build a (4, H, W) uint8 RGBA array from indicator dict."""
    height, width = indicators["density_z1"].shape
    rgba = np.zeros((4, height, width), dtype=np.uint8)

    quantized = {name: quantize_to_4bit(arr) for name, arr in indicators.items()}

    for ch_idx, (high_name, low_name) in enumerate(PACKING):
        rgba[ch_idx] = pack_4bit_pair(quantized[high_name], quantized[low_name])

    rgba[3] = 255  # A channel unused
    return rgba


def verify_roundtrip(
    indicators: dict[str, np.ndarray], rgba: np.ndarray
) -> dict[str, dict[str, float]]:
    """Decode the RGBA back to ratios and report per-indicator round-trip stats."""
    decoded: dict[str, np.ndarray] = {}
    for ch_idx, (high_name, low_name) in enumerate(PACKING):
        high4, low4 = unpack_4bit_pair(rgba[ch_idx])
        decoded[high_name] = high4.astype(np.float32) / 15.0
        decoded[low_name] = low4.astype(np.float32) / 15.0

    stats: dict[str, dict[str, float]] = {}
    for name in INDICATOR_BANDS:
        original = indicators[name]
        recovered = decoded[name]
        diff = np.abs(original - recovered)
        stats[name] = {
            "max_abs_error": float(diff.max()),
            "mean_abs_error": float(diff.mean()),
            "p95_abs_error": float(np.percentile(diff, 95)),
        }
    return stats


@click.command()
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Input multi-band float32 GeoTIFF from Phase 1.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output 4-band uint8 RGBA GeoTIFF.",
)
@click.option(
    "--verify",
    is_flag=True,
    help="After encoding, decode the result and report round-trip errors.",
)
def main(input_path: Path, output_path: Path, verify: bool) -> None:
    """Phase 2: pack 6 ratio indicators into a 4-bit RGBA GeoTIFF."""
    log.info("Reading indicators from %s", input_path)
    indicators, profile = read_indicators(input_path)
    log.info("Loaded shape=%s (H, W)", indicators["density_z1"].shape)

    log.info("Quantizing to 16 steps and packing into RGBA...")
    rgba = encode(indicators)

    log.info("Writing RGBA GeoTIFF to %s", output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_rgba(output_path, rgba, profile)

    if verify:
        log.info("Verifying round-trip (encode -> decode -> compare)...")
        stats = verify_roundtrip(indicators, rgba)
        log.info("Round-trip error stats (expected: max_abs <= 1/30 ≈ 0.034):")
        for name, st in stats.items():
            log.info(
                "  %-18s max=%.4f  mean=%.4f  p95=%.4f",
                name,
                st["max_abs_error"],
                st["mean_abs_error"],
                st["p95_abs_error"],
            )

    log.info("Done.")


if __name__ == "__main__":
    main()
