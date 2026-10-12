import numpy as np
import healpy as hp

from pygdsm import GlobalSkyModel16

from ulsa_sky import get_ulsa_sky


def generate_sky_map_combined(
    gsm,
    frequency_mhz,
    nside,
    sky_model,
    gsm_reference_10mhz=None,
    reference_map_11mhz=None,
):
    """
    Generate a sky map using GSM16 and optionally ULSA.

    sky_model:
        "ulsa":
            ULSA below 10 MHz, normalized to GSM16
            at 10 MHz. GSM16 at >= 10 MHz.

        "powerlaw":
            Original -2.7 extrapolation below
            10 MHz. GSM16 at >= 10 MHz.
    """

    freq = float(frequency_mhz)

    # GSM16 for frequencies >= 10 MHz
    if freq >= 10.0:
        return generate_sky_map(
            gsm,
            freq,
            nside,
        )

    # Original power-law extrapolation
    if sky_model == "powerlaw":

        if reference_map_11mhz is None:
            raise ValueError(
                "11-MHz reference map required"
            )

        return generate_sky_map_with_hf_extrapolation(
            gsm,
            freq,
            nside,
            reference_map_11mhz,
        )

    # New absorption-aware ULSA model
    if sky_model == "ulsa":

        if gsm_reference_10mhz is None:
            raise ValueError(
                "10-MHz GSM16 reference map required"
            )

        return get_ulsa_sky(
            freq,
            gsm_reference_10mhz,
            nside=nside,
            normalize_to_gsm=True,
        )

    raise ValueError(
        f"Unknown sky model: {sky_model}"
    )


def create_gsm():

    return GlobalSkyModel16(
        freq_unit="MHz",
        data_unit="TRJ",
        resolution="low",
        interpolation="pchip",
        include_cmb=True,
    )


def generate_sky_map(
    gsm,
    frequency_mhz,
    work_nside,
):
    """
    Generate a GSM2016 sky map for frequencies >= 10 MHz.
    """

    if frequency_mhz < 10.0:
        raise ValueError(
            "GSM2016 is not used below 10 MHz. "
            "Use generate_sky_map_with_hf_extrapolation()."
        )

    sky = np.asarray(
        gsm.generate(
            float(frequency_mhz)
        ),
        dtype=np.float64,
    )

    native_nside = hp.get_nside(sky)

    if native_nside != work_nside:

        sky = hp.ud_grade(
            sky,
            nside_out=work_nside,
            order_in="RING",
            order_out="RING",
            power=0,
        )

    return sky


def generate_hf_reference_map(
    gsm,
    work_nside,
):
    """
    Generate the 11 MHz GSM map used as the reference
    for the HF extrapolation.
    """

    return generate_sky_map(
        gsm,
        11.0,
        work_nside,
    )


def generate_sky_map_with_hf_extrapolation(
    gsm,
    frequency_mhz,
    work_nside,
    reference_map_11mhz,
):
    """
    Generate a sky map for any LFT3 frequency.

    For f <= 10 MHz:
        T(f) = T(11 MHz) * (f/11 MHz)^(-2.7)

    For f >= 10 MHz:
        use GSM2016 directly.
    """

    frequency_mhz = float(frequency_mhz)

    if frequency_mhz <= 10.0:

        beta = -2.7

        return (
            reference_map_11mhz
            * (frequency_mhz / 11.0) ** beta
        )

    return generate_sky_map(
        gsm,
        frequency_mhz,
        work_nside,
    )


def print_sky_diagnostics(sky):

    finite = np.isfinite(sky)

    print(
        f"min     = {np.nanmin(sky):.3f} K"
    )

    print(
        f"max     = {np.nanmax(sky):.3f} K"
    )

    print(
        f"mean    = {np.nanmean(sky):.3f} K"
    )

    print(
        f"median  = {np.nanmedian(sky):.3f} K"
    )

    print(
        "negative pixels =",
        np.sum(sky < 0)
    )

    print(
        "non-finite pixels =",
        np.sum(~finite)
    )
