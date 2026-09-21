import numpy as np
import spiceypy as spice

from config import (
    SITE_LAT_DEG,
    SITE_LON_DEG,
)

# Load relevant SPICE Kernels

def load_spice_kernels(kernel_directory):

    spice.kclear()

    kernels = [
        kernel_directory / "naif0012.tls",
        kernel_directory / "moon_pa_de440_200625.bpc",
        kernel_directory / "moon_de440_250416.tf",
    ]

    # Check that every kernel exists

    for kernel in kernels:
        if not kernel.exists():
            raise FileNotFoundError(
                f"Kernel not found: {kernel}"
            )

    # Load kernels

    for kernel in kernels:
        spice.furnsh(str(kernel))
        print(f"Loaded: {kernel.name}")


    # Confirm

    print(f"\nSPICE kernels loaded: {spice.ktotal('ALL')}")


def local_basis_moon_me(
    lat_deg=SITE_LAT_DEG,
    lon_deg=SITE_LON_DEG,
):

    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    north = np.array([
        -np.sin(lat) * np.cos(lon),
        -np.sin(lat) * np.sin(lon),
        np.cos(lat),
    ])

    east = np.array([
        -np.sin(lon),
        np.cos(lon),
        0.0,
    ])

    zenith = np.array([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])

    return north, east, zenith


def azel_to_moon_me(
    az_deg,
    el_deg,
):

    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)

    north, east, zenith = (
        local_basis_moon_me()
    )

    direction = (
        np.cos(el)
        * np.cos(az)
        * north
        +
        np.cos(el)
        * np.sin(az)
        * east
        +
        np.sin(el)
        * zenith
    )

    direction /= np.linalg.norm(
        direction
    )

    return direction


def make_rotation_matrices(times_utc):

    matrices = []

    for utc in times_utc:

        et = spice.str2et(utc)

        R = spice.pxform(
            "MOON_ME",
            "J2000",
            et,
        )

        matrices.append(R)

    return np.asarray(matrices)


def rotate_vector(
    vector_moon_me,
    rotation_matrices,
):

    vectors_j2000 = np.einsum(
        "tij,j->ti",
        rotation_matrices,
        vector_moon_me,
    )

    vectors_j2000 /= np.linalg.norm(
        vectors_j2000,
        axis=1,
        keepdims=True,
    )

    return vectors_j2000
    
def make_antenna_directions(
    az_deg,
    el_deg,
    rotation_matrices,
):

    direction_me = azel_to_moon_me(
        az_deg,
        el_deg,
    )

    return rotate_vector(
        direction_me,
        rotation_matrices,
    )


def make_zenith_directions(
    rotation_matrices,
):

    _, _, zenith_me = (
        local_basis_moon_me()
    )

    return rotate_vector(
        zenith_me,
        rotation_matrices,
    )
