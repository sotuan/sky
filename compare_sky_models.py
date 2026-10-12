from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

POWERLAW_FILE = Path("HF_sensitivity.npz")
ULSA_FILE = Path("HF_sensitivity_ulsa.npz")

OUTPUT_DIR = Path("plots_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD AND COMPARE RESULTS
# ============================================================

def load_results(filename):

    if not filename.exists():
        raise FileNotFoundError(filename)

    with np.load(filename) as data:
        return {
            key: data[key].copy()
            for key in data.files
        }


def calculate_ratio(ulsa, powerlaw, key):

    numerator = np.asarray(
        ulsa[key],
        dtype=float,
    )

    denominator = np.asarray(
        powerlaw[key],
        dtype=float,
    )

    if numerator.shape != denominator.shape:
        raise ValueError(
            f"Different shapes for {key}"
        )

    ratio = np.full_like(
        numerator,
        np.nan,
        dtype=float,
    )

    np.divide(
        numerator,
        denominator,
        out=ratio,
        where=(
            np.isfinite(numerator)
            & np.isfinite(denominator)
            & (denominator > 0)
        ),
    )

    return ratio


# ============================================================
# PLOTTING
# ============================================================

def plot_ratio(
    frequencies,
    ratio,
    ylabel,
    title,
    filename,
):

    ratio_min = np.nanmin(ratio, axis=0)
    ratio_max = np.nanmax(ratio, axis=0)
    ratio_median = np.nanmedian(ratio, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(
        frequencies,
        ratio_min,
        ratio_max,
        alpha=0.25,
        label="Min-max over time",
    )

    ax.plot(
        frequencies,
        ratio_median,
        linewidth=2,
        label="Median",
    )

    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1,
        label="No difference",
    )

    ax.set_xscale("log")

    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )

    ax.legend(loc="best")

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {filename}")


# ============================================================
# MAIN
# ============================================================

def main():

    powerlaw = load_results(POWERLAW_FILE)
    ulsa = load_results(ULSA_FILE)

    # Check that frequency and time grids agree
    frequencies = powerlaw["freqs_mhz"]

    if not np.allclose(
        frequencies,
        ulsa["freqs_mhz"],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError(
            "Frequency grids do not match"
        )

    if not np.array_equal(
        powerlaw["times_utc"],
        ulsa["times_utc"],
    ):
        raise ValueError(
            "Observation times do not match"
        )

    # --------------------------------------------------------
    # SEFD ratio
    # --------------------------------------------------------

    sefd_ratio = calculate_ratio(
        ulsa,
        powerlaw,
        "SEFD_Jy",
    )

    plot_ratio(
        frequencies,
        sefd_ratio,
        ylabel="SEFD(ULSA) / SEFD(power law)",
        title="LFT3 HF: SEFD comparison",
        filename=OUTPUT_DIR / "HF_SEFD_ratio_ulsa_powerlaw.png",
    )

    # --------------------------------------------------------
    # Sky-temperature ratio
    # --------------------------------------------------------

    tsky_ratio = calculate_ratio(
        ulsa,
        powerlaw,
        "Tsky_K",
    )

    plot_ratio(
        frequencies,
        tsky_ratio,
        ylabel="Tsky(ULSA) / Tsky(power law)",
        title="LFT3 HF: Sky temperature comparison",
        filename=OUTPUT_DIR / "HF_Tsky_ratio_ulsa_powerlaw.png",
    )

    # --------------------------------------------------------
    # Save numerical results
    # --------------------------------------------------------

    np.savez_compressed(
        OUTPUT_DIR / "HF_model_comparison.npz",
        times_utc=ulsa["times_utc"],
        freqs_mhz=frequencies,
        SEFD_ratio=sefd_ratio,
        Tsky_ratio=tsky_ratio,
    )

    # --------------------------------------------------------
    # Print diagnostic summary
    # --------------------------------------------------------

    print("\nComparison summary")
    print(
        "Frequency [MHz] | Median SEFD ratio | "
        "Min | Max"
    )

    for i, freq in enumerate(frequencies):

        values = sefd_ratio[:, i]

        print(
            f"{freq:8.2f} | "
            f"{np.nanmedian(values):12.4f} | "
            f"{np.nanmin(values):8.4f} | "
            f"{np.nanmax(values):8.4f}"
        )

    # Check frequencies above 10 MHz
    mask = frequencies > 10.0

    if np.any(mask):

        differences = np.abs(
            sefd_ratio[:, mask] - 1.0
        )

        print(
            "\nMaximum SEFD-ratio deviation "
            "from unity above 10 MHz:",
            np.nanmax(differences),
        )


if __name__ == "__main__":
    main()

