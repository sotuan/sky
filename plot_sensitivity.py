from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_sefd_ranges(
    results,
    bands,
    output_file="SEFD_sensitivity_range.png",
    show_plot=False,
):
    """
    Plot the minimum-to-maximum SEFD range over all observing times.

    Each SEFD array has shape

        (N_times, N_frequencies)

    so min/max/median are calculated over axis=0.
    """

    output_file = Path(output_file)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for band_name, band in bands.items():

        result = results[band_name]

        frequencies = band.freqs_mhz

        sefd = np.asarray(
            result["SEFD_Jy"],
            dtype=float,
        )

        # Statistics over observing time
        sefd_min = np.nanmin(
            sefd,
            axis=0,
        )

        sefd_max = np.nanmax(
            sefd,
            axis=0,
        )

        sefd_median = np.nanmedian(
            sefd,
            axis=0,
        )

        # Min-max envelope
        region = ax.fill_between(
            frequencies,
            sefd_min,
            sefd_max,
            alpha=0.25,
            label=f"{band_name}",
        )

        # Use same automatically chosen colour
        colour = region.get_facecolor()[0]

        # Envelope edges
        ax.plot(
            frequencies,
            sefd_min,
            color=colour,
            linewidth=1.0,
        )

        ax.plot(
            frequencies,
            sefd_max,
            color=colour,
            linewidth=1.0,
        )

        # Median
        ax.plot(
            frequencies,
            sefd_median,
            color=colour,
            linewidth=1.5,
            linestyle="--",
        )

    ax.set_xlabel(
        "Frequency [MHz]"
    )

    ax.set_ylabel(
        "SEFD [Jy]"
    )

    ax.set_title(
        "LFT3 SEFD range over observing period"
    )

    # Double-logarithmic axes
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    ax.legend(
        loc="upper right",
        fontsize="small",
        ncol=2,
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    print(
        f"Saved SEFD plot to: "
        f"{output_file.resolve()}"
    )

    if show_plot:
        plt.show()

    plt.close(fig)


def plot_ae_tsys_ranges(
    results,
    bands,
    output_file="Ae_Tsys_sensitivity_range.png",
    show_plot=False,
):
    """
    Plot the minimum-to-maximum A_eff/T_sys range
    over all observing times.

    Units:

        m^2 / K
    """

    output_file = Path(output_file)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for band_name, band in bands.items():

        result = results[band_name]

        frequencies = band.freqs_mhz

        ae_tsys = np.asarray(
            result["Ae_Tsys_m2_per_K"],
            dtype=float,
        )

        # Statistics over observing time
        ae_tsys_min = np.nanmin(
            ae_tsys,
            axis=0,
        )

        ae_tsys_max = np.nanmax(
            ae_tsys,
            axis=0,
        )

        ae_tsys_median = np.nanmedian(
            ae_tsys,
            axis=0,
        )

        # Min-max envelope
        region = ax.fill_between(
            frequencies,
            ae_tsys_min,
            ae_tsys_max,
            alpha=0.25,
            label=f"{band_name}",
        )

        colour = region.get_facecolor()[0]

        # Envelope edges
        ax.plot(
            frequencies,
            ae_tsys_min,
            color=colour,
            linewidth=1.0,
        )

        ax.plot(
            frequencies,
            ae_tsys_max,
            color=colour,
            linewidth=1.0,
        )

        # Median
        ax.plot(
            frequencies,
            ae_tsys_median,
            color=colour,
            linewidth=1.5,
            linestyle="--",
        )

    ax.set_xlabel(
        "Frequency [MHz]"
    )

    ax.set_ylabel(
        r"$A_{\rm eff}/T_{\rm sys}$ [$m^2$/K]"
    )

    ax.set_title(
        r"LFT3 $A_{\rm eff}/T_{\rm sys}$ "
        "range over observing period"
    )

    # Double-logarithmic axes
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    ax.legend(
        loc="upper left",
        fontsize="small",
        ncol=2,
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
    )

    print(
        f"Saved A_eff/T_sys plot to: "
        f"{output_file.resolve()}"
    )

    if show_plot:
        plt.show()

    plt.close(fig)


def plot_all_sensitivity_ranges(
    results,
    bands,
    show_plot=False,
):
    """
    Function that creates both sensitivity plots.
    """

    plot_sefd_ranges(
        results=results,
        bands=bands,
        output_file="SEFD_sensitivity_range.png",
        show_plot=show_plot,
    )

    plot_ae_tsys_ranges(
        results=results,
        bands=bands,
        output_file="Ae_Tsys_sensitivity_range.png",
        show_plot=show_plot,
    )
