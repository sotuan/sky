import numpy as np

from config import (
    KERNEL_DIR,
    START_DATE,
    STOP_DATE,
    TIME_STEP_DAYS,
    WORK_NSIDE,
    ANTENNA_ORIENTATIONS,
)

from lft3_specs import make_bands

from sky_model import (
    create_gsm,
    generate_sky_map_with_hf_extrapolation,
    generate_hf_reference_map,
    print_sky_diagnostics,
)

from galactic_to_ICRF import (
    get_icrs_pixel_vectors,
)

from lunar_geometry import (
    load_spice_kernels,
    make_rotation_matrices,
    make_antenna_directions,
    make_zenith_directions,
)

from sensitivity import (
    calculate_sensitivity,
)

from plot_sensitivity import (
    plot_all_sensitivity_ranges,
)

import spiceypy as spice


def make_times():

    start = np.datetime64(
        START_DATE
    )

    stop = np.datetime64(
        STOP_DATE
    )

    step = np.timedelta64(
        TIME_STEP_DAYS,
        "D"
    )

    times = np.arange(
        start,
        stop,
        step
    )

    return np.array([
        np.datetime_as_string(
            t,
            unit="s"
        )
        for t in times
    ])


def main():

    # ========================================================
    # 1. Times
    # ========================================================

    times = make_times()

    print(
        f"Number of times: {len(times)}"
    )

    print(
        "First:",
        times[0]
    )

    print(
        "Last:",
        times[-1]
    )


    # ========================================================
    # 2. SPICE
    # ========================================================

    load_spice_kernels(
        KERNEL_DIR
    )

    rotations = (
        make_rotation_matrices(
            times
        )
    )

    zenith_j2000 = (
        make_zenith_directions(
            rotations
        )
    )


    # ========================================================
    # 3. HEALPix sky directions
    # ========================================================

    sky_vectors_icrs = (
        get_icrs_pixel_vectors(
            WORK_NSIDE
        )
    )


    # ========================================================
    # 4. PyGDSM
    # ========================================================

    gsm = create_gsm()
    
    sky_11mhz = generate_hf_reference_map(
        gsm,
        WORK_NSIDE,
    )


    # ========================================================
    # 5. LFT3 bands
    # ========================================================

    bands = make_bands()

    results = {}


    # ========================================================
    # 6. Each band
    # ========================================================

    for band_name, band in bands.items():

        print()
        print("========================")
        print(band_name)
        print("========================")

        orientation = (
            ANTENNA_ORIENTATIONS[
                band_name
            ]
        )

        # ----------------------------------------------------
        # Antenna/formed-beam direction
        # ----------------------------------------------------

        antenna_j2000 = (
            make_antenna_directions(
                orientation["az_deg"],
                orientation["el_deg"],
                rotations,
            )
        )

        # ----------------------------------------------------
        # Generate GSM maps at every frequency
        # ----------------------------------------------------

        sky_maps = []

        for freq in band.freqs_mhz:

            print(
                f"Generating sky: "
                f"{freq:.1f} MHz"
            )

            sky = generate_sky_map_with_hf_extrapolation(
                gsm,
                freq,
                WORK_NSIDE,
                sky_11mhz,
            )

            sky_maps.append(sky)

        # ----------------------------------------------------
        # Calculate sensitivity
        # ----------------------------------------------------

        print(
            f"Finished generating {band_name} sky maps."
        )

        print(
            f"Starting {band_name} sensitivity calculation..."
        )

        result = calculate_sensitivity(
            band=band,
            frequencies_mhz=band.freqs_mhz,
            sky_maps=sky_maps,
            sky_vectors_icrs=sky_vectors_icrs,
            antenna_j2000=antenna_j2000,
            zenith_j2000=zenith_j2000,
        )

        results[band_name] = result

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        np.savez_compressed(
            f"{band_name}_sensitivity.npz",
            times_utc=times,
            freqs_mhz=band.freqs_mhz,
            Tsky_K=result["Tsky_K"],
            Tsys_K=result["Tsys_K"],
            Aeff_m2=result["Aeff_m2"],
            Ae_Tsys_m2_per_K=
                result["Ae_Tsys_m2_per_K"],
            SEFD_Jy=result["SEFD_Jy"],
        )

        print()
        print(
            f"{band_name}: saved."
        )
    
    
    # ========================================================
    # 7. Generate sensitivity plots
    # ========================================================

    plot_all_sensitivity_ranges(
        results=results,
        bands=bands,
        show_plot=False,
    )


    # ========================================================
    # 8. Clean up SPICE
    # ========================================================


    spice.kclear()

    return results


if __name__ == "__main__":

    results = main()
