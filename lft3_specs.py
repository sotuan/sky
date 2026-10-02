from dataclasses import dataclass
import numpy as np


@dataclass
class Band:

    name: str
    freqs_mhz: np.ndarray

    n_elements: int
    directivity: float
    trcvr_k: float

    beam_type: str

    aperture_diameter_m: float | None = None
    airy_lambda_scale: float | None = None


def make_bands():

    return {

        "HF": Band(
            name="HF",

            # 1-50 MHz, 1 MHz spacing
            freqs_mhz=np.arange(
                1.0, 50.0 + 0.1, 1.0
            ),

            n_elements=1,
            directivity=1.6,
            trcvr_k=300.0,
            beam_type="hf_dipole",
        ),

        "VHF": Band(
            name="VHF",

            # 60-250 MHz, 1 MHz spacing
            freqs_mhz=np.arange(
                60.0, 250.0 + 0.1, 1.0
            ),

            n_elements=1,
            directivity=1.6,
            trcvr_k=100.0,
            beam_type="half_wave_dipole",
        ),

        "UHF": Band(
            name="UHF",

            # 300-2700 MHz, 2 MHz
            freqs_mhz=np.arange(
                300.0, 2700.0 + 0.1, 2.0
            ),

            n_elements=48,
            directivity=3.1,
            trcvr_k=30.0,
            beam_type="airy",
            aperture_diameter_m=3.5,
            airy_lambda_scale=300.0,
        ),
    }
