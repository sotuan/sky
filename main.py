import numpy as np

from config import (
    KERNEL_DIR,
    START_DATE,
    STOP_DATE,
    TIME_STEP_DAYS,
    WORK_NSIDE,
    ACTIVE_ANTENNAS,
    BEAM_MODELS,
    EXPERIMENT_ORIENTATION,
    ANTENNA_ORIENTATIONS,
    LOW_FREQ_SKY_MODEL,
)

from beam_registry import load_beam_model

from lft3_specs1 import get_antenna_specs

from sky_model import (
    generate_sky_map,
    generate_sky_map_combined,
    create_gsm,
    generate_hf_reference_map,
    print_sky_diagnostics,
)

from galactic_to_ICRF import (
    get_icrs_pixel_vectors,
)

from lunar_geometry1 import (
    load_spice_kernels,
    make_rotation_matrices,
    make_zenith_directions,
    make_nez_axes,
    get_orientation,
)

from sensitivity import (
    calculate_sensitivity,
)

from plot_sensitivity import (
    plot_all_sensitivity_ranges,
)

import spiceypy as spice

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = PROJECT_ROOT / "results" / "sensitivity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


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
    
    nez_axes_icrs = make_nez_axes(
        rotations
    )
    
    old_zenith = make_zenith_directions(rotations)

    assert np.allclose(
        nez_axes_icrs[:, :, 2],
        old_zenith,
        atol=1e-12,
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
    # 4. LFT3 antennas specifications
    # ========================================================

    all_antennas = get_antenna_specs()

    antennas = {
        name: all_antennas[name]
        for name in ACTIVE_ANTENNAS
    }

    # Restrict UHF frequencies for testing
    antennas["UHF"].freqs_mhz = np.arange(
        300.0,
        500.0 + 0.1,
        2.0,
    )

    results = {}
    

    # ========================================================
    # 5. PyGDSM
    # ========================================================

    gsm = create_gsm()
    
    gsm_reference_10mhz = None
    sky_11mhz = None

    if LOW_FREQ_SKY_MODEL == "ulsa":

        gsm_reference_10mhz = generate_sky_map(
            gsm,
            10.0,
            WORK_NSIDE,
    )

    elif LOW_FREQ_SKY_MODEL == "powerlaw":

        sky_11mhz = generate_hf_reference_map(
            gsm,
            WORK_NSIDE,
        )

    else:
        raise ValueError(
            f"Unknown sky model: {LOW_FREQ_SKY_MODEL}"
        )
        

    # ========================================================
    # 6. Calculate sensitivity for each antenna
    # ========================================================

    for antenna_name, antenna in antennas.items():

        selected_beam_model = BEAM_MODELS[antenna_name]
        
        # Identify the sky model and beam model in output filenames
    
        output_suffix = (
            f"_{LOW_FREQ_SKY_MODEL}"
            f"_{selected_beam_model}"
            f"_300-500MHz"
        )
        
        beam_config = load_beam_model(
            antenna_name,
            selected_beam_model,
        )

        print(
            f"{antenna_name}: "
            f"using beam model '{selected_beam_model}'"
        )
    
        # Experiment orientation relative to local lunar NEZ
        
        orientation = get_orientation(
            **EXPERIMENT_ORIENTATION
        )
        
        # ----------------------------------------------------
        # Generate GSM maps at every frequency
        # ----------------------------------------------------

        sky_maps = []

        for freq in antenna.freqs_mhz:

            print(
                f"Generating sky: "
                f"{freq:.1f} MHz"
            )

            sky = generate_sky_map_combined(
                gsm,
                freq,
                WORK_NSIDE,
                sky_model=LOW_FREQ_SKY_MODEL,
                gsm_reference_10mhz=gsm_reference_10mhz,
                reference_map_11mhz=sky_11mhz,
            )
            
            sky_maps.append(sky)

        # ----------------------------------------------------
        # Calculate sensitivity
        # ----------------------------------------------------

        print(
            f"Finished generating {antenna_name} sky maps."
        )

        print(
            f"Starting {antenna_name} sensitivity calculation..."
        )
        
        result = calculate_sensitivity(
            antenna=antenna,
            antenna_name=antenna_name,
            frequencies_mhz=antenna.freqs_mhz,
            sky_maps=sky_maps,
            sky_vectors_icrs=sky_vectors_icrs,
            nez_axes_icrs=nez_axes_icrs,
            orientation=orientation,
            beam_config=beam_config,
        )
        
        results[antenna_name] = result

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        np.savez_compressed(
            RESULTS_DIR / f"{antenna_name}_sensitivity{output_suffix}.npz",
            times_utc=times,
            freqs_mhz=antenna.freqs_mhz,
            sky_model=LOW_FREQ_SKY_MODEL,
            beam_model=selected_beam_model,
            Trcvr_K=result["Trcvr_K"],
            Tsky_K=result["Tsky_K"],
            Tsys_K=result["Tsys_K"],
            Aeff_m2=result["Aeff_m2"],
            Ae_Tsys_m2_per_K=
                result["Ae_Tsys_m2_per_K"],
            SEFD_Jy=result["SEFD_Jy"],
        )

        print()
        print(
            f"{antenna_name}: saved."
        )
    
    
    # ========================================================
    # 7. Generate sensitivity plots
    # ========================================================

    plot_suffix = f"_{LOW_FREQ_SKY_MODEL}"

    plot_all_sensitivity_ranges(
        results=results,
        antennas=antennas,
        show_plot=False,
        output_suffix=plot_suffix,
    )


    # ========================================================
    # 8. Clean up SPICE
    # ========================================================


    spice.kclear()

    return results


if __name__ == "__main__":

    results = main()
