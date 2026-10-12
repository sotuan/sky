from pathlib import Path
import time

import numpy as np
import healpy as hp

from ULSA.sky_map.produce_absorbed_sky_map import absorption_JRZ


# ==========================================
# USER SETTINGS
# ==========================================

FREQUENCIES_MHZ = np.arange(1, 11, 1)

NSIDE = 64

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "ulsa_maps"

# ==========================================
# GENERATE SKY MAPS
# ==========================================

def generate_map(freq_mhz):

    freq_label = f"{freq_mhz:g}"

    filename = (
        OUTPUT_DIR
        / f"ulsa_{freq_label}MHz_nside{NSIDE}.npy"
    )

    # Skip previously generated maps
    if filename.exists():
        print(f"Already exists, skipping: {filename}")
        return

    print(
        f"\nGenerating ULSA map at "
        f"{freq_mhz:g} MHz, NSIDE={NSIDE}"
    )

    start = time.perf_counter()

    model = absorption_JRZ(
        v=float(freq_mhz),
        nside=NSIDE,
        index_type="constant_index",
        distance=50.0,
        using_raw_diffuse=False,
        using_default_params=True,
        input_spectral_index=None,
        critical_dis=False,
        output_absorp_free_skymap=False,
    )

    result = np.asarray(model.mpi())

    # Verify output dimensions
    npix = hp.nside2npix(NSIDE)

    if result.shape != (npix, 2):
        raise ValueError(
            f"Unexpected output shape: {result.shape}"
        )

    pixels = result[:, 0].astype(int)
    temperatures = result[:, 1]

    # Verify pixel indices
    if not np.array_equal(
        np.sort(pixels),
        np.arange(npix)
    ):
        raise ValueError(
            "Invalid or duplicate HEALPix pixel indices"
        )

    # Reconstruct Galactic RING map
    sky_map = np.empty(npix, dtype=float)
    sky_map[pixels] = temperatures

    if not np.all(np.isfinite(sky_map)):
        raise ValueError("Non-finite temperatures")

    if np.any(sky_map <= 0):
        raise ValueError("Non-positive temperatures")

    # Save the map
    np.save(filename, sky_map)

    elapsed = time.perf_counter() - start

    print(f"Saved: {filename}")
    print(f"Elapsed time: {elapsed / 60:.1f} minutes")
    print(f"Mean temperature: {sky_map.mean():.3e} K")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    for freq in FREQUENCIES_MHZ:
        generate_map(freq)

    print("\nAll ULSA maps completed.")

