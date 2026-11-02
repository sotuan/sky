from dataclasses import dataclass
import numpy as np


@dataclass
class Antenna:

    name: str
    freqs_mhz: np.ndarray

    n_elements: int

    aperture_diameter_m: float | None = None


def get_antenna_specs():

    return {

        "HF": Antenna(
            name="HF",

            # 1-50 MHz, 1 MHz spacing
            freqs_mhz=np.arange(
                1.0, 50.0 + 0.1, 1.0
            ),

            n_elements=1,
        ),

        "VHF": Antenna(
            name="VHF",

            # 60-250 MHz, 1 MHz spacing
            freqs_mhz=np.arange(
                60.0, 250.0 + 0.1, 1.0
            ),

            n_elements=1,
        ),

        "UHF": Antenna(
            name="UHF",

            # 300-2700 MHz, 2 MHz spacing
            freqs_mhz=np.arange(
                300.0, 2700.0 + 0.1, 2.0
            ),

            n_elements=48,
            aperture_diameter_m=3.5,
        ),
    }
