from pathlib import Path
from beam1 import NumericalBeam


PROJECT_ROOT = Path(__file__).resolve().parent

BEAM_DIR = PROJECT_ROOT / "data" / "beams"

BEAM_REGISTRY = {
    "HF": {
        "analytic": {
            "type": "hf_dipole",
        },
        "model_A": {
            "type": "numerical",
            "file": BEAM_DIR / "HF" / "model_A.npz",
        },
        "model_B": {
            "type": "numerical",
            "file": BEAM_DIR / "HF" / "model_B.npz",
        },
    },
    "VHF": {
        "analytic": {
            "type": "half_wave_dipole",
        },
        "model_A": {
            "type": "numerical",
            "file": BEAM_DIR / "VHF" / "model_A.npz",
        },
    },
    "UHF": {
        "analytic": {
            "type": "airy",
        },
        "model_A": {
            "type": "numerical",
            "file": BEAM_DIR / "UHF" / "model_A.npz",
        },
        "model_B": {
            "type": "numerical",
            "file": BEAM_DIR / "UHF" / "model_B.npz",
        },
        "model_C": {
            "type": "numerical",
            "file": BEAM_DIR / "UHF" / "model_C.npz",
        },
    },
}


def load_beam_model(antenna_name, model_name):

    if antenna_name not in BEAM_REGISTRY:
        raise ValueError(
            f"Unknown antenna: {antenna_name}"
        )

    models = BEAM_REGISTRY[antenna_name]

    if model_name not in models:
        raise ValueError(
            f"Unknown beam model '{model_name}' "
            f"for antenna '{antenna_name}'"
        )

    definition = models[model_name]

    beam_type = definition["type"]

    if beam_type == "numerical":

        filename = definition["file"]

        if not filename.exists():
            raise FileNotFoundError(filename)

        numerical_beam = NumericalBeam.from_npz(
            filename
        )

        return {
            "beam_type": "numerical",
            "numerical_beam": numerical_beam,
        }

    return {
        "beam_type": beam_type,
        "numerical_beam": None,
    }


