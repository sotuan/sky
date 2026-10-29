import numpy as np

# ============================================================
# Receiver models
# ============================================================

def my_receiver(frequency_mhz, antenna_name):
    """
    Return receiver temperature in kelvin.

    Parameters
    ----------
    frequency_mhz : float or array-like
        Observing frequency in MHz.

    antenna_name : str
        LFT3 antennas: "HF", "VHF", or "UHF".

    Returns
    -------
    float or ndarray
        Receiver temperature in kelvin.

    Notes
    -----
    Replace the expressions below with measured or
    modeled frequency-dependent receiver temperatures.
    """

    freq = np.asarray(frequency_mhz, dtype=float)

    if np.any(~np.isfinite(freq)) or np.any(freq <= 0):
        raise ValueError("Frequency must be finite and positive")

    if antenna_name == "HF":

        # HF: 1-50 MHz
        T_rcvr = np.full_like(freq, 300.0)

    elif antenna_name == "VHF":

        # VHF: 60-250 MHz
        T_rcvr = np.full_like(freq, 100.0)

    elif antenna_name == "UHF":

        # UHF: 300-2700 MHz
        T_rcvr = np.full_like(freq, 30.0)

    else:
        raise ValueError(
            f"Unknown antenna: {antenna_name}"
        )

    if freq.ndim == 0:
        return float(T_rcvr)

    return T_rcvr

