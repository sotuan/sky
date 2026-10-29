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

# Choose low-frequency (below 10 MHz) sky model ("ulsa" or "powerlaw")
LOW_FREQ_SKY_MODEL = "ulsa"

# ============================================================
# Active antennas
# ============================================================

ACTIVE_ANTENNAS = ["UHF"]

# ============================================================
# Beam models (see beam_registry.py)
# ============================================================

BEAM_MODELS = {
    "HF": "analytic",
    "VHF": "analytic",
    "UHF": "analytic",
}

# ============================================================
# Experiment orientation relative to local lunar NEZ
# ============================================================

EXPERIMENT_ORIENTATION = {
    "alpha_deg": 0.0,
    "beta_deg": 0.0,
    "gamma_deg": 0.0,
}

# ============================================================
# Antenna orientations
#
# Convention:
#   az =   0 deg -> experiment X axis
#   az =  90 deg -> experiment Y axis
#   el =   0 deg -> experiment XY plane
#   el =  90 deg -> experiment Z axis (direction toward the top)
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
