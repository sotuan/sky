from pathlib import Path


# ============================================================
# Path to SPICE kernels
# ============================================================

KERNEL_DIR = Path(Path.home() / "Documents" / "LFT3" / "sky" / "SPICE_kernels")


# ============================================================
# LFT3 site (latitude and longitude)
# ============================================================

SITE_LAT_DEG = -23.789
SITE_LON_DEG = 182.137


# ============================================================
# Observation times
# ============================================================

START_DATE = "2028-07-01T00:00:00"
STOP_DATE = "2029-01-01T00:00:00"
TIME_STEP_DAYS = 1


# ============================================================
# HEALPix resolution used in the calculation
# ============================================================

# GSM2016 low-resolution map is NSIDE=64.
WORK_NSIDE = 64


# ============================================================
# Antenna orientations
#
# Convention:
#   az =   0 deg -> local north
#   az =  90 deg -> local east
#   el =   0 deg -> local horizon
#   el =  90 deg -> local zenith
#
# HF/VHF:
#   orientation describes the dipole axis.
#
# UHF:
#   orientation describes the formed-beam boresight.
# ============================================================

ANTENNA_ORIENTATIONS = {

    "HF": {
        "az_deg": 0.0,
        "el_deg": 0.0,
    },

    "VHF": {
        "az_deg": 0.0,
        "el_deg": 0.0,
    },

    "UHF": {
        "az_deg": 0.0,
        "el_deg": 90.0,
    },
}


# ============================================================
# Horizon model
#
# Below the lunar horizon we currently assume:
#
#     T_Moon = 0 K
#
# i.e. blocked sky contributes nothing.
# ============================================================

APPLY_LUNAR_HORIZON = True
MOON_BRIGHTNESS_K = 0.0


# ============================================================
# Physical constants
# ============================================================

K_B = 1.380649e-23
