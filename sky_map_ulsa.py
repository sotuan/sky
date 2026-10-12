import numpy as np
import healpy as hp

from ULSA.sky_map.produce_absorbed_sky_map import absorption_JRZ

# Ask user for frequency
FREQ_MHZ = float(input("Enter frequency in MHz: "))

NSIDE = 32

print(f"Generating ULSA sky map at {FREQ_MHZ:g} MHz...")

model = absorption_JRZ(
    v=FREQ_MHZ,
    nside=NSIDE,
    index_type="constant_index",
    distance=50.0,
    using_raw_diffuse=False,
    using_default_params=True,
    input_spectral_index=None,
    critical_dis=False,
    output_absorp_free_skymap=False,
)

result = np.asarray(model.mpi())

print("Result shape:", result.shape)
print("First rows:", result[:5])

# Include frequency in filename
freq_label = f"{FREQ_MHZ:g}"
filename = f"ulsa_{freq_label}MHz.npy"

np.save(filename, result)

print(f"Saved ULSA map to {filename}")

