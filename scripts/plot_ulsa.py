import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================
# USER SETTINGS
# ==========================================

FREQUENCIES_MHZ = [2, 5, 10]

INPUT_DIR = Path(".")
OUTPUT_DIR = Path("plots_ulsa")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# GENERATE PLOTS
# ==========================================

for freq_mhz in FREQUENCIES_MHZ:

    freq_label = f"{freq_mhz:g}"

    input_file = INPUT_DIR / f"ulsa_{freq_label}MHz.npy"
    output_file = OUTPUT_DIR / f"ulsa_{freq_label}MHz_32.png"

    if not input_file.exists():
        print(f"File not found: {input_file}")
        continue

    # Load ULSA two-column output
    result = np.load(input_file)

    pixels = result[:, 0].astype(int)
    temperatures = result[:, 1]

    # Determine NSIDE automatically
    npix = len(pixels)
    nside = hp.npix2nside(npix)

    # Reconstruct HEALPix map in RING ordering
    sky_map = np.empty(npix, dtype=float)
    sky_map[pixels] = temperatures

    # Print map statistics
    print(f"\nFrequency: {freq_mhz:g} MHz")
    print(f"NSIDE: {nside}")
    print(f"Minimum temperature: {np.min(sky_map):.3e} K")
    print(f"Maximum temperature: {np.max(sky_map):.3e} K")
    print(f"Mean temperature: {np.mean(sky_map):.3e} K")

    # Plot logarithmic sky temperature
    hp.mollview(
        np.log10(np.maximum(sky_map, 1.0)),
        nest=False,
        coord="G",
        title=f"ULSA sky temperature at {freq_mhz:g} MHz",
        unit="log10(K)",
        cmap="viridis",
    )

    hp.graticule()

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close("all")

    print(f"Saved: {output_file}")

print("\nAll requested ULSA plots completed.")

