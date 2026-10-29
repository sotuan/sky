from pathlib import Path
import numpy as np


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results" / "sensitivity"

OLD_FILE = RESULTS_DIR / "UHF_sensitivity_ulsa.npz"

NEW_FILE = (
    RESULTS_DIR
    / "UHF_sensitivity_ulsa_analytic_300-500MHz.npz"
)

QUANTITIES = [
    "Tsky_K",
    "Tsys_K",
    "Aeff_m2",
    "Ae_Tsys_m2_per_K",
    "SEFD_Jy",
]


# ============================================================
# Load results
# ============================================================

with np.load(OLD_FILE) as data:
    old = {key: data[key] for key in data.files}

with np.load(NEW_FILE) as data:
    new = {key: data[key] for key in data.files}

print("\nOld file:", OLD_FILE.name)
print("New file:", NEW_FILE.name)

print("\nAvailable arrays:")
print("Old:", list(old))
print("New:", list(new))


# ============================================================
# Check observation times
# ============================================================

old_times = old["times_utc"]
new_times = new["times_utc"]

if not np.array_equal(old_times, new_times):
    raise ValueError(
        "Observation times differ between calculations."
    )

print("\nObservation times match.")
print("Number of times:", len(new_times))


# ============================================================
# Match frequencies
# ============================================================

old_freqs = np.asarray(
    old["freqs_mhz"], dtype=float
)

new_freqs = np.asarray(
    new["freqs_mhz"], dtype=float
)

if not np.all(np.diff(old_freqs) > 0):
    raise ValueError(
        "Old frequencies must be strictly increasing."
    )

indices = np.searchsorted(old_freqs, new_freqs)

if np.any(indices >= len(old_freqs)):
    raise ValueError(
        "Some new frequencies are outside the old grid."
    )

matched_freqs = old_freqs[indices]

if not np.allclose(
    matched_freqs,
    new_freqs,
    rtol=0,
    atol=1e-9,
):
    raise ValueError(
        "Some new frequencies are absent from the old grid."
    )

print("\nFrequency grids matched.")
print("Number of frequencies:", len(new_freqs))
print("Minimum frequency:", new_freqs.min(), "MHz")
print("Maximum frequency:", new_freqs.max(), "MHz")


# ============================================================
# Compare sensitivity quantities
# ============================================================

print("\nComparison:")
print("-" * 85)

for key in QUANTITIES:

    if key not in old or key not in new:
        print(f"\n{key}: missing from one of the files")
        continue

    old_values = np.asarray(
        old[key], dtype=np.float64
    )

    new_values = np.asarray(
        new[key], dtype=np.float64
    )

    # Select matching frequencies from the old results
    if old_values.ndim == 2:
        old_values = old_values[:, indices]

    elif old_values.ndim == 1:
        old_values = old_values[indices]

    else:
        raise ValueError(
            f"Unexpected number of dimensions for {key}"
        )

    if old_values.shape != new_values.shape:
        print(
            f"\n{key}: SHAPE MISMATCH "
            f"{old_values.shape} vs {new_values.shape}"
        )
        continue

    # Absolute differences
    abs_diff = np.abs(
        new_values - old_values
    )

    # Relative differences
    rel_diff = np.full_like(
        abs_diff,
        np.nan,
    )

    np.divide(
        abs_diff,
        np.abs(old_values),
        out=rel_diff,
        where=(old_values != 0),
    )

    max_abs = np.nanmax(abs_diff)

    finite_rel = np.isfinite(rel_diff)

    if np.any(finite_rel):
        max_rel = np.max(rel_diff[finite_rel])
        mean_rel = np.mean(rel_diff[finite_rel])
    else:
        max_rel = np.nan
        mean_rel = np.nan

    identical = np.array_equal(
        old_values,
        new_values,
    )

    close = np.allclose(
        old_values,
        new_values,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )

    print(f"\n{key}")
    print(f"  Shape:             {new_values.shape}")
    print(f"  Exactly identical: {identical}")
    print(f"  Numerically close: {close}")
    print(f"  Maximum abs diff:  {max_abs:.6e}")
    print(f"  Maximum rel diff:  {max_rel:.6e}")
    print(f"  Mean rel diff:     {mean_rel:.6e}")

    # Identify the location of the largest difference
    if np.any(np.isfinite(abs_diff)):

        max_index = np.unravel_index(
            np.nanargmax(abs_diff),
            abs_diff.shape,
        )

        if old_values.ndim == 2:
            i, j = max_index
            print(f"  At time:           {new_times[i]}")
            print(f"  At frequency:      {new_freqs[j]:.1f} MHz")
        else:
            j = max_index[0]
            print(f"  At frequency:      {new_freqs[j]:.1f} MHz")

        print(
            f"  Old value:         "
            f"{old_values[max_index]:.8e}"
        )

        print(
            f"  New value:         "
            f"{new_values[max_index]:.8e}"
        )


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 85)
print("Comparison complete.")
print("=" * 85)

