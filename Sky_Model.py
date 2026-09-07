import numpy as np
import healpy as hp

from pygdsm import GlobalSkyModel16


def create_gsm():
    """
    Create the GSM2016 model.

    TRJ = Rayleigh-Jeans brightness temperature, in Kelvin.
    PCHIP is used because it is shape-preserving and avoids some
    interpolation overshoot problems.
    """

    gsm = GlobalSkyModel16(
        freq_unit="MHz",
        data_unit="TRJ",
        resolution="hi",
        interpolation="pchip",
        include_cmb=True,
    )

    return gsm


def generate_sky_map(gsm, frequency_mhz, work_nside=128):
    """
    Generate the GSM2016 sky temperature map.

    Parameters
    ----------
    gsm : GlobalSkyModel16
        Initialized PyGDSM model.

    frequency_mhz : float
        Observing frequency in MHz.

    work_nside : int
        HEALPix NSIDE used for the beam calculation.

    Returns
    -------
    sky : ndarray
        Sky brightness temperature in K_RJ.
        Map is in Galactic coordinates, RING ordering.
    """

    sky = gsm.generate(frequency_mhz)

    sky = np.asarray(sky, dtype=np.float64)

    original_nside = hp.get_nside(sky)

    print(f"GSM native NSIDE = {original_nside}")
    print(f"GSM pixels       = {len(sky):,}")

    # Reduce resolution for beam integration.
    #
    # GSM2016 hi-resolution output has NSIDE=1024,
    # which is 12,582,912 pixels. That is unnecessary
    # for a broad beam of many degrees.
    if work_nside != original_nside:

        sky = hp.ud_grade(
            sky,
            nside_out=work_nside,
            order_in="RING",
            order_out="RING",
            power=0,
        )

    print(f"Working NSIDE    = {hp.get_nside(sky)}")
    print(f"Working pixels   = {len(sky):,}")

    return sky


def print_sky_diagnostics(sky):
    """
    Print some useful GSM diagnostics.
    """

    print()
    print("Sky-map diagnostics")
    print("-------------------")
    print(f"Minimum: {np.nanmin(sky):.3f} K")
    print(f"Maximum: {np.nanmax(sky):.3f} K")
    print(f"Mean:    {np.nanmean(sky):.3f} K")
    print(f"Median:  {np.nanmedian(sky):.3f} K")
    print(f"Negative pixels: {np.sum(sky < 0)}")
    print(f"NaN pixels:      {np.sum(~np.isfinite(sky))}")
