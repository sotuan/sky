from pathlib import Path

import numpy as np
import healpy as hp

from astropy.coordinates import SkyCoord
import astropy.units as u


def galactic_to_icrs_matrix():

    gal_basis = SkyCoord(
        l=[0.0, 90.0, 0.0] * u.deg,
        b=[0.0, 0.0, 90.0] * u.deg,
        frame="galactic",
    )

    icrs_basis = gal_basis.icrs

    return icrs_basis.cartesian.xyz.value


def build_icrs_pixel_vectors(nside):

    npix = hp.nside2npix(nside)

    pixels = np.arange(npix)

    x, y, z = hp.pix2vec(
        nside,
        pixels,
        nest=False,
    )

    vectors_gal = np.column_stack(
        (x, y, z)
    )

    R = galactic_to_icrs_matrix()

    vectors_icrs = (
        vectors_gal @ R.T
    )

    vectors_icrs /= np.linalg.norm(
        vectors_icrs,
        axis=1,
        keepdims=True,
    )

    return vectors_icrs


def get_icrs_pixel_vectors(
    nside,
    cache_directory="cache",
):

    cache_directory = Path(
        cache_directory
    )

    cache_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_file = (
        cache_directory
        / f"icrs_vectors_nside{nside}.npy"
    )

    if cache_file.exists():

        print(
            "Loading cached sky vectors:",
            cache_file,
        )

        return np.load(cache_file)

    print(
        "Calculating Galactic -> ICRS sky vectors..."
    )

    vectors = build_icrs_pixel_vectors(
        nside
    )

    np.save(
        cache_file,
        vectors
    )

    return vectors
