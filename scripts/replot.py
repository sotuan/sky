import numpy as np

from lft3_specs import make_bands
from plot_sensitivity import plot_all_sensitivity_ranges


bands = make_bands()

results = {}

for band_name, band in bands.items():

    filename = f"{band_name}_sensitivity.npz"

    print(f"Loading {filename}")

    data = np.load(filename)

    results[band_name] = {
        "Tsky_K": data["Tsky_K"],
        "Tsys_K": data["Tsys_K"],
        "Aeff_m2": data["Aeff_m2"],
        "Ae_Tsys_m2_per_K":
            data["Ae_Tsys_m2_per_K"],
        "SEFD_Jy": data["SEFD_Jy"],
    }


plot_all_sensitivity_ranges(
    results=results,
    bands=bands,
    show_plot=False,
)

print("Plots regenerated.")
