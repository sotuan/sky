import numpy as np


def gaussian_beam(
    sky_vectors,
    boresight,
    fwhm_deg,
    apply_lunar_horizon=True,
):
    """
    Calculate Gaussian beam weights for all sky pixels.

    Parameters
    ----------
    sky_vectors : ndarray, shape (Npix, 3)
        Sky-pixel unit vectors in J2000/ICRS.

    boresight : ndarray, shape (3,)
        Beam boresight unit vector in J2000.

    fwhm_deg : float
        Full width at half maximum of the Gaussian POWER beam.

    apply_lunar_horizon : bool
        If True, set all directions below the lunar horizon
        to zero.

    Returns
    -------
    beam : ndarray
        Beam weights.

    cos_angle : ndarray
        Cosine of angular separation from boresight.
    """

    boresight = np.asarray(
        boresight,
        dtype=np.float64
    )

    boresight /= np.linalg.norm(boresight)

    # Dot product with every sky pixel:
    #
    # cos(theta_i) = b . s_i
    cos_angle = sky_vectors @ boresight

    # Protect arccos from tiny floating-point excursions.
    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0
    )

    theta = np.arccos(cos_angle)

    fwhm_rad = np.deg2rad(fwhm_deg)

    beam = np.exp(
        -4.0 * np.log(2.0)
        * (theta / fwhm_rad)**2
    )

    # For a zenith-pointing beam on a spherical Moon:
    #
    #     cos(theta) > 0
    #
    # is exactly the condition that the sky direction
    # is above the local horizon.
    if apply_lunar_horizon:

        visible = cos_angle > 0.0

        beam[~visible] = 0.0

    return beam, cos_angle


def beam_averaged_temperature(
    sky_temperature,
    beam,
):
    """
    Calculate beam-weighted sky brightness temperature.

    Because all HEALPix pixels at a fixed NSIDE have equal
    solid angle, DeltaOmega cancels between numerator and
    denominator.

    Returns
    -------
    T_beam : float
        Beam-averaged sky brightness temperature in K.
    """

    sky_temperature = np.asarray(sky_temperature)
    beam = np.asarray(beam)

    # Ignore NaNs/infs if present.
    valid = (
        np.isfinite(sky_temperature)
        & np.isfinite(beam)
    )

    numerator = np.sum(
        beam[valid] *
        sky_temperature[valid]
    )

    denominator = np.sum(
        beam[valid]
    )

    if denominator <= 0:
        raise RuntimeError(
            "Beam integral is zero."
        )

    return numerator / denominator
