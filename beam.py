import numpy as np
from scipy.special import j1

from lft3_specs import Band


def half_wave_dipole_beam_from_cos(
    cos_theta
):

    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0,
    )

    sin_theta = np.sqrt(
        np.maximum(
            0.0,
            1.0 - cos_theta**2
        )
    )

    field = np.zeros_like(
        cos_theta
    )

    valid = (
        sin_theta > 1e-12
    )

    field[valid] = (
        np.cos(
            0.5
            * np.pi
            * cos_theta[valid]
        )
        / sin_theta[valid]
    )

    return field**2


def hf_beam_from_cos(
    cos_theta,
    frequency_mhz,
):

    cos_theta = np.clip(
        cos_theta,
        -1.0,
        1.0,
    )

    sin_theta = np.sqrt(
        np.maximum(
            0.0,
            1.0 - cos_theta**2
        )
    )

    small_field = (
        sin_theta
    )

    half_field = np.zeros_like(
        cos_theta
    )

    full_field = np.zeros_like(
        cos_theta
    )

    valid = (
        sin_theta > 1e-12
    )

    half_field[valid] = (
        np.cos(
            0.5
            * np.pi
            * cos_theta[valid]
        )
        / sin_theta[valid]
    )

    full_field[valid] = (
        np.cos(
            np.pi
            * cos_theta[valid]
        )
        + 1.0
    ) / (
        2.0
        * sin_theta[valid]
    )

    f = frequency_mhz

    if f < 10.0:

        field = small_field

    elif f < 25.0:

        w = (
            (f - 10.0)
            / 15.0
        )

        field = (
            (1.0 - w)
            * small_field
            +
            w
            * half_field
        )

    else:

        w = (
            (f - 25.0)
            / 25.0
        )

        field = (
            (1.0 - w)
            * half_field
            +
            w
            * full_field
        )

    return field**2


def airy_beam_from_cos(
    cos_alpha,
    frequency_mhz,
    diameter_m,
    lambda_scale=300.0,
):

    cos_alpha = np.clip(
        cos_alpha,
        -1.0,
        1.0,
    )

    sin_alpha = np.sqrt(
        np.maximum(
            0.0,
            1.0 - cos_alpha**2
        )
    )

    wavelength = (
        lambda_scale
        / frequency_mhz
    )

    x = (
        np.pi
        * diameter_m
        / wavelength
        * sin_alpha
    )

    beam = np.ones_like(x)

    valid = (
        np.abs(x) > 1e-12
    )

    beam[valid] = (
        2.0
        * j1(x[valid])
        / x[valid]
    )**2

    # Keep only the forward formed beam.
    beam[cos_alpha < 0.0] = 0.0

    return beam


def calculate_beam(
    band,
    frequency_mhz,
    cos_angle,
):

    if band.beam_type == "hf_dipole":

        return hf_beam_from_cos(
            cos_angle,
            frequency_mhz,
        )

    if band.beam_type == "half_wave_dipole":

        return (
            half_wave_dipole_beam_from_cos(
                cos_angle
            )
        )

    if band.beam_type == "airy":

        return airy_beam_from_cos(
            cos_angle,
            frequency_mhz,
            band.aperture_diameter_m,
            band.airy_lambda_scale,
        )

    raise ValueError(
        f"Unknown beam type: "
        f"{band.beam_type}"
    )
    
    
def apply_lunar_horizon(
    beam,
    sky_temperature,
    cos_zenith,
    moon_temperature=0.0,
):
    """
    Apply the lunar horizon.

    cos_zenith > 0:
        direction is above the horizon.

    cos_zenith <= 0:
        direction is behind the Moon.

    The blocked region is assigned the supplied
    lunar brightness temperature, currently 0 K.
    """

    visible = cos_zenith > 0.0

    environment = np.where(
        visible,
        sky_temperature,
        moon_temperature,
    )

    return beam, environment


def beam_average(
    sky_temperature,
    beam,
):

    valid = (
        np.isfinite(sky_temperature)
        &
        np.isfinite(beam)
    )

    numerator = np.sum(
        beam[valid]
        * sky_temperature[valid]
    )

    denominator = np.sum(
        beam[valid]
    )

    if denominator <= 0.0:

        raise RuntimeError(
            "Beam normalization is zero."
        )

    return numerator / denominator
