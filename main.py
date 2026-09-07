import healpy as hp
import numpy as np

from Sky_Model import (
    create_gsm,
    generate_sky_map,
    print_sky_diagnostics,
)

from Galactic_to_ICRF import (
    get_icrs_pixel_vectors,
)

from beam import (
    gaussian_beam,
    beam_averaged_temperature,
)

from Boresight_Dir import zenith_j2000


# ============================================================
# Configuration
# ============================================================

FREQUENCY_MHZ = 250.0

FWHM_DEG = 60.0

WORK_NSIDE = 128

UTC = "2029-10-29T12:00:00"


# ============================================================
# 1. Load GSM2016
# ============================================================

print("\nLoading GSM2016...")

gsm = create_gsm()


# ============================================================
# 2. Generate sky temperature map
# ============================================================

print()
print(
    f"Generating sky at {FREQUENCY_MHZ} MHz..."
)

sky = generate_sky_map(
    gsm,
    frequency_mhz=FREQUENCY_MHZ,
    work_nside=WORK_NSIDE,
)

print_sky_diagnostics(sky)


# ============================================================
# 3. Generate fixed celestial vectors for HEALPix pixels
# ============================================================

nside = hp.get_nside(sky)

sky_vectors = get_icrs_pixel_vectors(
    nside
)

print()
print(
    "Sky-vector array shape:",
    sky_vectors.shape
)


# ============================================================
# 4. Obtain lunar local zenith from your SPICE code
# ============================================================

b_j2000 = zenith_j2000(UTC)

b_j2000 = np.asarray(
    b_j2000,
    dtype=np.float64
)

b_j2000 /= np.linalg.norm(b_j2000)

print()
print("Observation UTC:")
print(UTC)

print()
print("LFT3 local zenith in J2000:")
print(b_j2000)

print(
    "Boresight norm:",
    np.linalg.norm(b_j2000)
)


# ============================================================
# 5. Gaussian beam
# ============================================================

beam, cos_angle = gaussian_beam(
    sky_vectors=sky_vectors,
    boresight=b_j2000,
    fwhm_deg=FWHM_DEG,
    apply_lunar_horizon=True,
)


# ============================================================
# 6. Beam-averaged sky temperature
# ============================================================

T_beam = beam_averaged_temperature(
    sky_temperature=sky,
    beam=beam,
)


# ============================================================
# 7. Results
# ============================================================

visible_pixels = np.sum(
    cos_angle > 0
)

print()
print("============================")
print("RESULT")
print("============================")

print(
    f"Observation UTC: {UTC}"
)

print(
    f"Frequency:       {FREQUENCY_MHZ:.1f} MHz"
)

print(
    f"Gaussian FWHM:   {FWHM_DEG:.1f} deg"
)

print(
    f"Visible pixels:  {visible_pixels:,}"
)

print(
    f"Beam temperature: {T_beam:.3f} K"
)
