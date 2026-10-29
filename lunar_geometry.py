import numpy as np
import spiceypy as spice

from config import SITE_LAT_DEG, SITE_LON_DEG


def load_spice_kernels(kernel_directory):
    """Load the SPICE kernels needed for lunar-frame transformations."""
    spice.kclear()
    kernels = [
        kernel_directory / "naif0012.tls",
        kernel_directory / "moon_pa_de440_200625.bpc",
        kernel_directory / "moon_de440_250416.tf",
    ]
    for kernel in kernels:
        if not kernel.exists():
            raise FileNotFoundError(f"Kernel not found: {kernel}")
    for kernel in kernels:
        spice.furnsh(str(kernel))
        print(f"Loaded: {kernel.name}")
    print(f"\nSPICE kernels loaded: {spice.ktotal('ALL')}")


def local_basis_moon_me(lat_deg=SITE_LAT_DEG, lon_deg=SITE_LON_DEG):
    """Return local North, East and Zenith unit vectors in MOON_ME."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    north = np.array([
        -np.sin(lat) * np.cos(lon),
        -np.sin(lat) * np.sin(lon),
        np.cos(lat),
    ])
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    zenith = np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])
    return north, east, zenith


def azel_to_moon_me(az_deg, el_deg):
    """Convert azimuth (east of north) and elevation to a MOON_ME vector."""
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    north, east, zenith = local_basis_moon_me()
    direction = (
        np.cos(el) * np.cos(az) * north
        + np.cos(el) * np.sin(az) * east
        + np.sin(el) * zenith
    )
    direction /= np.linalg.norm(direction)
    return direction


def make_rotation_matrices(times_utc):
    """Return MOON_ME-to-J2000 matrices with shape (N_times, 3, 3)."""
    matrices = []
    for utc in times_utc:
        et = spice.str2et(utc)
        matrices.append(spice.pxform("MOON_ME", "J2000", et))
    return np.asarray(matrices, dtype=float).reshape(-1, 3, 3)


def rotate_vector(vector_moon_me, rotation_matrices):
    """Rotate one MOON_ME direction to J2000 at every observing time."""
    vectors_j2000 = np.einsum(
        "tij,j->ti", rotation_matrices, vector_moon_me
    )
    vectors_j2000 /= np.linalg.norm(
        vectors_j2000, axis=1, keepdims=True
    )
    return vectors_j2000


def make_nez_axes(rotation_matrices):
    """Return the local NEZ basis expressed in J2000 at each time.

    Parameters
    ----------
    rotation_matrices : ndarray, shape (N_times, 3, 3)
        MOON_ME-to-J2000 rotation matrices from make_rotation_matrices().

    Returns
    -------
    ndarray, shape (N_times, 3, 3)
        Column 0: North; column 1: East; column 2: Zenith.
        For each time t, v_j2000 = nez_axes[t] @ v_nez.
    """
    rotations = np.asarray(rotation_matrices, dtype=float)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("rotation_matrices must have shape (N_times, 3, 3)")
    if not np.all(np.isfinite(rotations)):
        raise ValueError("rotation_matrices must contain only finite values")

    north, east, zenith = local_basis_moon_me()
    basis_me = np.column_stack((north, east, zenith))
    return rotations @ basis_me


def make_antenna_directions(az_deg, el_deg, rotation_matrices):
    """Legacy pointing-vector interface; retained for compatibility."""
    direction_me = azel_to_moon_me(az_deg, el_deg)
    return rotate_vector(direction_me, rotation_matrices)


def make_zenith_directions(rotation_matrices):
    """Legacy zenith-vector interface; retained for compatibility."""
    _, _, zenith_me = local_basis_moon_me()
    return rotate_vector(zenith_me, rotation_matrices)


def get_orientation(alpha_deg=0.0, beta_deg=0.0, gamma_deg=0.0):
    """Return experiment Euler angles (azimuth, tilt, roll) in degrees.

    Zero angles align experiment X/Y/Z with local North/East/Zenith.
    The associated experiment-to-NEZ rotation is
    Rz(alpha) @ Ry(beta) @ Rz(gamma).
    For nonzero beta, alpha is measured from North toward East.
    """
    angles = np.asarray([alpha_deg, beta_deg, gamma_deg], dtype=float)
    if not np.all(np.isfinite(angles)):
        raise ValueError("Orientation angles must be finite")
    return tuple(float(v) for v in angles)
