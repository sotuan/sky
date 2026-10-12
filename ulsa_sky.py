from pathlib import Path
from functools import lru_cache

import healpy as hp
import numpy as np


ULSA_DIR = Path(__file__).resolve().parent / "ulsa_maps"

T_CMB = 2.7255
GLUING_FREQ_MHZ = 10.0


@lru_cache(maxsize=16)
def load_ulsa_map(freq_mhz, nside):
    """
    Load a ULSA Galactic RING HEALPix map.

    Frequency: MHz
    Temperature: K
    """

    freq = float(freq_mhz)
    label = f"{freq:g}"

    filename = (
        ULSA_DIR
        / f"ulsa_{label}MHz_nside{nside}.npy"
    )

    if not filename.exists():
        raise FileNotFoundError(
            f"ULSA map not found: {filename}"
        )

    sky = np.load(filename)

    expected_npix = hp.nside2npix(nside)

    if sky.shape != (expected_npix,):
        raise ValueError(
            f"Incorrect map shape: {sky.shape}. "
            f"Expected ({expected_npix},)"
        )

    if not np.all(np.isfinite(sky)):
        raise ValueError("ULSA map contains invalid values")

    if np.any(sky <= 0):
        raise ValueError("ULSA map contains non-positive values")

    return sky



def get_ulsa_sky(
    freq_mhz,
    gsm_reference,
    nside,
    normalize_to_gsm=True,
):
    """
    Return ULSA sky temperature at any frequency
    between 1 and 10 MHz.

    Uses log-log interpolation between cached maps.
    """

    freq = float(freq_mhz)

    if not (1.0 <= freq <= 10.0):
        raise ValueError(
            "ULSA supports 1 <= freq <= 10 MHz"
        )

    # Find neighboring integer frequencies
    f1 = int(np.floor(freq))
    f2 = int(np.ceil(freq))

    sky1 = load_ulsa_map(f1, nside)

    if f1 == f2:
        sky_ulsa = sky1.copy()

    else:
        sky2 = load_ulsa_map(f2, nside)

        weight = (
            (np.log(freq) - np.log(f1))
            /
            (np.log(f2) - np.log(f1))
        )

        sky_ulsa = np.exp(
            (1.0 - weight) * np.log(sky1)
            +
            weight * np.log(sky2)
        )

    if not normalize_to_gsm:
        return sky_ulsa

    reference_ulsa = load_ulsa_map(
        10.0,
        nside,
    )

    gsm_reference = np.asarray(
        gsm_reference,
        dtype=float,
    )

    if gsm_reference.shape != sky_ulsa.shape:
        raise ValueError(
            "GSM16 and ULSA maps have different shapes"
        )

    if not np.all(np.isfinite(gsm_reference)):
        raise ValueError(
            "GSM16 reference contains invalid values"
        )

    ratio = sky_ulsa / reference_ulsa

    return (
        T_CMB
        + (gsm_reference - T_CMB) * ratio
    )


