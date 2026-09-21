import numpy as np

from beam import (
    calculate_beam,
)
from lft3_specs import Band
from config import K_B


def effective_area(
    band: Band,
    frequency_mhz,
):

    wavelength = (
        300.0 / frequency_mhz
    )

    Aeff = (
        band.n_elements
        * wavelength**2
        * band.directivity
        / (4.0 * np.pi)
    )

    # Special HF factor
    if band.name == "HF":

        a = 0.48 / 2499.0
        c = 0.02 - a

        backlobe = (
            a * frequency_mhz**2
            + c
        )

        Aeff *= backlobe

    return Aeff


def calculate_sensitivity(
    band,
    frequencies_mhz,
    sky_maps,
    sky_vectors_icrs,
    antenna_j2000,
    zenith_j2000,
):

    n_times = len(antenna_j2000)
    n_freqs = len(frequencies_mhz)

    Tsky = np.zeros(
        (n_times, n_freqs),
        dtype=np.float64,
    )

    Tsys = np.zeros_like(Tsky)

    Ae_Tsys = np.zeros_like(Tsky)

    SEFD = np.zeros_like(Tsky)

    Aeff = np.zeros(
        n_freqs,
        dtype=np.float64,
    )

    # ========================================================
    # GEOMETRY
    #
    # These quantities do not depend on frequency.
    # Therefore calculate them only once.
    # ========================================================

    print("Calculating antenna-to-sky geometry...")

    cos_angle = (
        antenna_j2000
        @ sky_vectors_icrs.T
    )

    cos_angle = np.clip(
        cos_angle,
        -1.0,
        1.0,
    )

    print("Calculating lunar horizon mask...")

    cos_zenith = (
        zenith_j2000
        @ sky_vectors_icrs.T
    )

    visible = (
        cos_zenith > 0.0
    )

    # cos_zenith is no longer required.
    del cos_zenith

    print(
        f"Starting sensitivity calculation "
        f"for {n_freqs} frequencies..."
    )

    # ========================================================
    # FREQUENCY LOOP
    # ========================================================

    for j, freq in enumerate(
        frequencies_mhz
    ):

        print(
            f"  {band.name}: "
            f"{freq:.1f} MHz "
            f"({j + 1}/{n_freqs})"
        )

        sky = np.asarray(
            sky_maps[j],
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # Effective area
        # ----------------------------------------------------

        Aeff[j] = effective_area(
            band,
            freq,
        )

        # ----------------------------------------------------
        # Beam
        #
        # cos_angle is already known for every
        # time and every sky pixel.
        # ----------------------------------------------------

        beam = calculate_beam(
            band,
            freq,
            cos_angle,
        )

        # ----------------------------------------------------
        # Lunar horizon
        #
        # Moon temperature is currently 0 K.
        #
        # Therefore only visible-sky pixels contribute
        # to the numerator.
        #
        # Denominator remains the full beam integral.
        # ----------------------------------------------------

        sky_safe = np.where(
            np.isfinite(sky),
            sky,
            0.0,
        )

        numerator = (
            (beam * visible)
            @ sky_safe
        )

        denominator = np.sum(
            beam,
            axis=1,
        )

        if np.any(
            denominator <= 0.0
        ):
            raise RuntimeError(
                "Beam normalization is zero."
            )

        # ----------------------------------------------------
        # Beam-weighted sky temperature
        # ----------------------------------------------------

        Tsky[:, j] = (
            numerator / denominator
        )

        # ----------------------------------------------------
        # System temperature
        # ----------------------------------------------------

        Tsys[:, j] = (
            Tsky[:, j]
            + band.trcvr_k
        )

        # ----------------------------------------------------
        # A_eff / T_sys
        # ----------------------------------------------------

        Ae_Tsys[:, j] = (
            Aeff[j]
            / Tsys[:, j]
        )

        # ----------------------------------------------------
        # SEFD [Jy]
        # ----------------------------------------------------

        SEFD[:, j] = (
            2.0
            * K_B
            / Ae_Tsys[:, j]
            * 1.0e26
        )

        # Delete this large array before moving to the next frequency.
        del beam

    print(
        f"{band.name} sensitivity calculation complete."
    )

    return {
        "Tsky_K": Tsky,
        "Tsys_K": Tsys,
        "Aeff_m2": Aeff,
        "Ae_Tsys_m2_per_K": Ae_Tsys,
        "SEFD_Jy": SEFD,
    }
