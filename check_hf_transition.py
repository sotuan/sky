import numpy as np

data = np.load("HF_sensitivity_ulsa.npz")

freqs = data["freqs_mhz"]
Tsky = data["Tsky_K"]
Tsys = data["Tsys_K"]
SEFD = data["SEFD_Jy"]

print(
    "Frequency | Median Tsky [K] | "
    "Median Tsys [K] | Median SEFD [Jy]"
)

for freq in [7, 8, 9, 10, 11, 12, 13, 14]:

    idx = np.where(np.isclose(freqs, freq))[0]

    if len(idx) == 0:
        continue

    i = idx[0]

    print(
        f"{freq:8.1f} | "
        f"{np.median(Tsky[:, i]):15.4e} | "
        f"{np.median(Tsys[:, i]):15.4e} | "
        f"{np.median(SEFD[:, i]):15.4e}"
    )

print("\nEffective sky-temperature spectral indices:")

for f1, f2 in [(8, 9), (9, 10), (10, 11),
               (11, 12), (12, 13)]:

    i1 = np.where(np.isclose(freqs, f1))[0]
    i2 = np.where(np.isclose(freqs, f2))[0]

    if len(i1) == 0 or len(i2) == 0:
        continue

    t1 = np.median(Tsky[:, i1[0]])
    t2 = np.median(Tsky[:, i2[0]])

    beta = np.log(t2 / t1) / np.log(f2 / f1)

    print(f"{f1:2d}-{f2:2d} MHz: beta = {beta:.3f}")

