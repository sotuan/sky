from dataclasses import dataclass
import numpy as np

from config import STRICT_LFT3SOFT_UHF_HI


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

    uhf_hi_lambda_scale = (
        900.0
        if STRICT_LFT3SOFT_UHF_HI
        else 300.0
    )

    return {

        "HF": Band(
            name="HF",

            # lft3soft: 1-50 MHz, 1 MHz spacing
            freqs_mhz=np.arange(
                1.0, 50.0 + 0.1, 1.0
            ),

            n_elements=1,
            directivity=1.6,
            trcvr_k=300.0,
            beam_type="hf_dipole",
        ),

        "VHF_LO": Band(
            name="VHF_LO",

            # 60-110 MHz, 1 MHz
            freqs_mhz=np.arange(
                60.0, 110.0 + 0.1, 1.0
            ),

            n_elements=1,
            directivity=1.6,
            trcvr_k=100.0,
            beam_type="half_wave_dipole",
        ),

        "VHF_HI": Band(
            name="VHF_HI",

            # 150-250 MHz, 1 MHz
            freqs_mhz=np.arange(
                150.0, 250.0 + 0.1, 1.0
            ),

            n_elements=1,
            directivity=1.6,
            trcvr_k=100.0,
            beam_type="half_wave_dipole",
        ),

        "UHF_LO": Band(
            name="UHF_LO",

            # 300-900 MHz, 2 MHz
            freqs_mhz=np.arange(
                300.0, 900.0 + 0.1, 2.0
            ),

            n_elements=48,
            directivity=3.1,
            trcvr_k=30.0,
            beam_type="airy",
            aperture_diameter_m=3.5,
            airy_lambda_scale=300.0,
        ),

        "UHF_HI": Band(
            name="UHF_HI",

            # 900-2700 MHz, 10 MHz
            freqs_mhz=np.arange(
                900.0, 2700.0 + 0.1, 10.0
            ),

            n_elements=8,
            directivity=3.1,
            trcvr_k=35.0,
            beam_type="airy",
            aperture_diameter_m=0.5,
            airy_lambda_scale=uhf_hi_lambda_scale,
        ),
    }
