import numpy as np

from beam1 import (
    analytic_directivity,
    beam_directivity,
    hf_backlobe_factor,
    get_beam,
)

from lft3_specs1 import Antenna
from config import (
    K_B,
    ANTENNA_ORIENTATIONS,
)
from receiver import my_receiver


def effective_area(
    antenna,
    frequency_mhz,
    beam_config,
):
    """
    Calculate effective collecting area.

    Analytic models:
        Preserve the original element-based
        effective-area prescription.

    Numerical models:
        Use directivity of the complete formed
        array, assuming a full-sphere power pattern
        and separately specified efficiency.
    """

    freq = float(frequency_mhz)

    wavelength = 299.8 / freq

    beam_type = beam_config["beam_type"]

    if beam_type != "numerical":

        directivity = analytic_directivity(
            antenna.name
        )

        Aeff = (
            antenna.n_elements
            * wavelength**2
            * directivity
            / (4.0 * np.pi)
        )

        if antenna.name == "HF":
            Aeff *= hf_backlobe_factor(freq)

        return Aeff

    # Numerical beam representing the complete array
    directivity = beam_directivity(
        frequency_mhz=freq,
        beam_type="numerical",
        numerical_beam=beam_config["numerical_beam"],
    )

    # Assumed efficiency is 1
    efficiency = 1.0

    return (
        efficiency
        * wavelength**2
        * directivity
        / (4.0 * np.pi)
    )   


def calculate_sensitivity_old(
    antenna,
    antenna_name,
    frequencies_mhz,
    sky_maps,
    sky_vectors_icrs,
    nez_axes_icrs,
    orientation,
    beam_config,
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
            f"  {antenna_name}: "
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
            antenna,
            antenna_name,
            freq,
        )

        # ----------------------------------------------------
        # Beam
        #
        # cos_angle is already known for every
        # time and every sky pixel.
        # ----------------------------------------------------

        beam = get_beam(
            sky_vectors_icrs=sky_vectors_icrs,
            nez_axes_icrs=nez_axes_icrs,
            frequency_mhz=freq,
            beam_type=beam_config["beam_type"],
            numerical_beam=beam_config["numerical_beam"],
            orientation=orientation,
            antenna_orientation=ANTENNA_ORIENTATIONS[antenna.name],
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
        # Receiver temperature
        # ----------------------------------------------------
        
        T_rcvr = my_receiver(
            frequencies_mhz,
            antenna_name,
        )

        # ----------------------------------------------------
        # System temperature
        # ----------------------------------------------------

        Tsys[:, j] = (
            Tsky[:, j]
            + T_rcvr[np.newaxis, j]
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
        f"{antenna_name} sensitivity calculation complete."
    )

    return {
        "Tsky_K": Tsky,
        "Trcvr_K": T_rcvr,
        "Tsys_K": Tsys,
        "Aeff_m2": Aeff,
        "Ae_Tsys_m2_per_K": Ae_Tsys,
        "SEFD_Jy": SEFD,
    }
    
    

def calculate_sensitivity(
    antenna,
    antenna_name,
    frequencies_mhz,
    sky_maps,
    sky_vectors_icrs,
    nez_axes_icrs,
    orientation,
    beam_config,
):

    frequencies_mhz = np.asarray(
        frequencies_mhz,
        dtype=np.float64,
    )

    sky_vectors_icrs = np.asarray(
        sky_vectors_icrs,
        dtype=np.float64,
    )

    nez_axes_icrs = np.asarray(
        nez_axes_icrs,
        dtype=np.float64,
    )

    n_times = len(nez_axes_icrs)
    n_freqs = len(frequencies_mhz)
    n_pix = len(sky_vectors_icrs)

    if nez_axes_icrs.shape != (n_times, 3, 3):
        raise ValueError(
            "nez_axes_icrs must have shape (N_times, 3, 3)"
        )

    if sky_vectors_icrs.shape != (n_pix, 3):
        raise ValueError(
            "sky_vectors_icrs must have shape (N_pix, 3)"
        )

    if len(sky_maps) != n_freqs:
        raise ValueError(
            "Number of sky maps must match frequencies"
        )

    # ========================================================
    # OUTPUT ARRAYS
    # ========================================================

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
    # RECEIVER TEMPERATURE
    # ========================================================

    T_rcvr = np.asarray(
        my_receiver(
            frequencies_mhz,
            antenna.name,
        ),
        dtype=np.float64,
    )

    if T_rcvr.shape != (n_freqs,):
        raise ValueError(
            "Receiver temperature must have shape (N_freqs,)"
        )

    # ========================================================
    # LUNAR HORIZON
    # ========================================================

    print("Calculating lunar horizon mask...")

    # Third column of NEZ matrix is the local zenith.
    zenith_icrs = nez_axes_icrs[:, :, 2]

    cos_zenith = (
        zenith_icrs @ sky_vectors_icrs.T
    )

    visible = cos_zenith > 0.0

    del cos_zenith

    # ========================================================
    # FREQUENCY LOOP
    # ========================================================

    print(
        f"Starting sensitivity calculation "
        f"for {n_freqs} frequencies..."
    )

    for j, freq in enumerate(frequencies_mhz):

        print(
            f"  {antenna_name}: "
            f"{freq:.1f} MHz "
            f"({j + 1}/{n_freqs})"
        )

        sky = np.asarray(
            sky_maps[j],
            dtype=np.float64,
        )

        if sky.shape != (n_pix,):
            raise ValueError(
                f"Sky map at {freq} MHz has "
                f"shape {sky.shape}, expected {(n_pix,)}"
            )

        sky_safe = np.where(
            np.isfinite(sky),
            sky,
            0.0,
        )

        # ----------------------------------------------------
        # Effective area
        # ----------------------------------------------------

        Aeff[j] = effective_area(
            antenna=antenna,
            frequency_mhz=freq,
            beam_config=beam_config,
        )

        # ----------------------------------------------------
        # TIME LOOP
        # ----------------------------------------------------

        for i in range(n_times):

            # Evaluate the beam for this observing time.
            beam = np.asarray(
                get_beam(
                    sky_vectors_icrs=sky_vectors_icrs,
                    nez_axes_icrs=nez_axes_icrs[i],
                    frequency_mhz=freq,
                    beam_type=beam_config["beam_type"],
                    numerical_beam=beam_config["numerical_beam"],
                    orientation=orientation,
                    antenna_orientation=ANTENNA_ORIENTATIONS[antenna.name],
                ),
                dtype=np.float64,
            )

            if beam.shape != (n_pix,):
                raise ValueError(
                    f"Beam shape {beam.shape}, "
                    f"expected {(n_pix,)}"
                )

            if not np.all(np.isfinite(beam)):
                raise ValueError(
                    "Beam contains non-finite values"
                )

            if np.any(beam < 0.0):
                raise ValueError(
                    "Power beam contains negative values"
                )

            # -----------------------------------------------
            # Beam normalization
            # -----------------------------------------------

            denominator = np.sum(beam)

            if denominator <= 0.0:
                raise RuntimeError(
                    "Beam normalization is zero"
                )

            # -----------------------------------------------
            # Sky contribution
            #
            # Only sky above the lunar horizon contributes.
            # The denominator uses the full beam.
            # -----------------------------------------------

            numerator = np.dot(
                beam * visible[i],
                sky_safe,
            )

            Tsky[i, j] = numerator / denominator

        # ----------------------------------------------------
        # SYSTEM TEMPERATURE
        # ----------------------------------------------------

        Tsys[:, j] = (
            Tsky[:, j] + T_rcvr[j]
        )

        if np.any(Tsys[:, j] <= 0.0):
            raise RuntimeError(
                f"Non-positive Tsys at {freq} MHz"
            )

        # ----------------------------------------------------
        # A_eff / T_sys
        # ----------------------------------------------------

        Ae_Tsys[:, j] = (
            Aeff[j] / Tsys[:, j]
        )

        # ----------------------------------------------------
        # SEFD
        # ----------------------------------------------------

        SEFD[:, j] = (
            2.0 * K_B
            / Ae_Tsys[:, j]
            * 1.0e26
        )

    print(
        f"{antenna_name} sensitivity calculation complete."
    )

    return {
        "Tsky_K": Tsky,
        "Trcvr_K": T_rcvr,
        "Tsys_K": Tsys,
        "Aeff_m2": Aeff,
        "Ae_Tsys_m2_per_K": Ae_Tsys,
        "SEFD_Jy": SEFD,
    }

