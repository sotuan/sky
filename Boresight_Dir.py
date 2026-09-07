from pathlib import Path

import numpy as np
import spiceypy as spice


# ============================================================
# 1. Load relevant SPICE Kernels
# ============================================================

# -------------------------------------------------
# Directories for kernels
# -------------------------------------------------

KERNEL_DIR = Path.home() / "Documents" / "LFT3" / "SkyT" / "SPICE_kernels"

kernels = [
    KERNEL_DIR / "naif0012.tls",
    KERNEL_DIR / "moon_pa_de440_200625.bpc",
    KERNEL_DIR / "moon_de440_250416.tf",
]


# -------------------------------------------------
# Check that every kernel exists
# -------------------------------------------------

for kernel in kernels:
    if not kernel.exists():
        raise FileNotFoundError(
            f"Kernel not found: {kernel}"
        )

# -------------------------------------------------
# Load kernels
# -------------------------------------------------

for kernel in kernels:
    spice.furnsh(str(kernel))
    print(f"Loaded: {kernel.name}")


# -------------------------------------------------
# Confirm
# -------------------------------------------------

print(f"\nSPICE kernels loaded: {spice.ktotal('ALL')}")


# ============================================================
# 2. LFT3 site
# ============================================================
#
# Here we explicitly ASSUME that the published LFT3
# latitude/longitude are interpreted as MOON_ME coordinates.
#

LAT_DEG = -23.78930
LON_DEG = 182.1373

lat = np.deg2rad(LAT_DEG)
lon = np.deg2rad(LON_DEG)


# ============================================================
# 3. Zenith direction in MOON_ME
# ============================================================
#
# Spherical Moon:
#
#     x = cos(lat) cos(lon)
#     y = cos(lat) sin(lon)
#     z = sin(lat)
#
# Because this is already a unit vector from the lunar centre,
# it is also the local zenith direction for a spherical Moon.
#

b_me = np.array([
    np.cos(lat) * np.cos(lon),
    np.cos(lat) * np.sin(lon),
    np.sin(lat),
])

b_me /= np.linalg.norm(b_me)

print("\nLFT3 local zenith in MOON_ME:")
print(b_me)
print("Norm:", np.linalg.norm(b_me))


# ============================================================
# 4. Function: MOON_ME zenith -> J2000/ICRF
# ============================================================

def zenith_j2000(utc):
    """
    Calculate the local-zenith direction of LFT3 in J2000/ICRF (difference between J2000 and ICRF should be negligible for most practical applications).

    Parameters
    ----------
    utc : str
        UTC time understood by SPICE, e.g.
        "2026-09-04T12:00:00"

    Returns
    -------
    b_j2000 : numpy.ndarray, shape (3,)
        Unit vector pointing toward the LFT3 local zenith,
        expressed in J2000/ICRF coordinates.
    """

    # UTC -> ephemeris time
    et = spice.str2et(utc)

    # Rotation matrix:
    #
    #     MOON_ME -> J2000
    #
    R = spice.pxform(
        "MOON_ME",
        "J2000",
        et,
    )

    # Transform the fixed lunar zenith
    b_j2000 = R @ b_me

    # Remove tiny numerical normalization error
    b_j2000 /= np.linalg.norm(b_j2000)
    
    #Clean up SPICE kernel pool
    spice.kclear()

    return b_j2000


# ============================================================
# 5. Example
# ============================================================

#utc = "2029-09-22T12:00:00"

#b = zenith_j2000(utc)

#print()
#print("UTC:", utc)
#print("Zenith in J2000:")
#print(b)
#print("Norm:", np.linalg.norm(b))


# ============================================================
# Convert J2000 unit vector to RA/Dec
# ============================================================

#radius, ra, dec = spice.recrad(b)

#ra_deg = np.rad2deg(ra)
#dec_deg = np.rad2deg(dec)

#print()
#print(f"RA  = {ra_deg:.6f} deg")
#print(f"Dec = {dec_deg:.6f} deg")
