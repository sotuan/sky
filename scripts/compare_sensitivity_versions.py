from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "sensitivity"

old_file = RESULTS_DIR / "VHF_sensitivity_ulsa.npz"
new_file = RESULTS_DIR / "VHF_sensitivity_ulsa_analytic.npz"

old = np.load(old_file)
new = np.load(new_file)

print("\nOld file:", old_file.name)
print("New file:", new_file.name)

print("\nAvailable arrays:")
print("Old:", old.files)
print("New:", new.files)

# Check coordinate grids
for key in ["times_utc", "freqs_mhz"]:
    assert np.array_equal(old[key], new[key]), (
        f"{key} differs between calculations"
    )

print("\nTime and frequency grids match.")

quantities = [
    "Tsky_K",
    "Tsys_K",
    "Aeff_m2",
    "Ae_Tsys_m2_per_K",
    "SEFD_Jy",
]

print("\nComparison:")
print("-" * 85)

for key in quantities:

    a = np.asarray(old[key], dtype=float)
    b = np.asarray(new[key], dtype=float)

    if a.shape != b.shape:
        print(f"{key}: SHAPE MISMATCH {a.shape} vs {b.shape}")
        continue

    diff = b - a

    abs_diff = np.abs(diff)

    # Relative difference, excluding zero reference values
    relative = np.full_like(abs_diff, np.nan)

    np.divide(
        abs_diff,
        np.abs(a),
        out=relative,
        where=(a != 0),
    )

    max_abs = np.nanmax(abs_diff)

    max_rel = (
        np.nanmax(relative)
        if np.any(np.isfinite(relative))
        else np.nan
    )

    mean_rel = (
        np.nanmean(relative)
        if np.any(np.isfinite(relative))
        else np.nan
    )

    identical = np.array_equal(a, b)

    print(f"\n{key}")
    print(f"  Shape:             {a.shape}")
    print(f"  Exactly identical: {identical}")
    print(f"  Maximum abs diff:  {max_abs:.6e}")
    print(f"  Maximum rel diff:  {max_rel:.6e}")
    print(f"  Mean rel diff:     {mean_rel:.6e}")

print("\nComparison complete.")

