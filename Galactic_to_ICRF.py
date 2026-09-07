from pathlib import Path

import numpy as np
import healpy as hp

from astropy.coordinates import SkyCoord
import astropy.units as u


def galactic_to_icrs_matrix():

    basis_gal = SkyCoord(
        l=[0.0, 90.0, 0.0] * u.deg,
        b=[0.0, 0.0, 90.0] * u.deg,
        frame="galactic",
    )

    basis_icrs = basis_gal.icrs

    return basis_icrs.cartesian.xyz.value


def build_icrs_pixel_vectors(nside):
    """
    Return the J2000/ICRS direction of every HEALPix pixel.

    The input HEALPix grid is interpreted as Galactic coordinates
    in RING ordering, matching the output of GSM2016.

    Parameters
    ----------
    nside : int

    Returns
    -------
    vectors_icrs : ndarray, shape (Npix, 3)
        Unit vectors in ICRS/J2000 coordinates.
    """

    npix = hp.nside2npix(nside)

    pixels = np.arange(npix)

    # HEALPix Cartesian vectors.
    #
    # Since the GSM map is Galactic, these x,y,z vectors
    # are Galactic Cartesian vectors.
    x, y, z = hp.pix2vec(
        nside,
        pixels,
        nest=False,       # RING ordering
    )

    vectors_gal = np.column_stack((x, y, z))

    # Galactic -> ICRS transformation
    R = galactic_to_icrs_matrix()

    # For row vectors:
    #
    # v_icrs = v_gal @ R.T
    vectors_icrs = vectors_gal @ R.T

    # Normalize against numerical roundoff.
    vectors_icrs /= np.linalg.norm(
        vectors_icrs,
        axis=1,
        keepdims=True,
    )

    return vectors_icrs


def get_icrs_pixel_vectors(nside, cache_directory="cache"):
    """
    Load pixel vectors from disk if already calculated.
    Otherwise calculate them and save them.

    This transformation depends only on NSIDE, not on
    frequency or observation time.
    """

    cache_directory = Path(cache_directory)

    cache_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    cache_file = (
        cache_directory /
        f"healpix_icrs_vectors_nside{nside}.npy"
    )

    if cache_file.exists():

        print(f"Loading cached sky vectors:")
        print(cache_file)

        return np.load(cache_file)

    print("Calculating Galactic -> ICRS sky vectors...")

    vectors = build_icrs_pixel_vectors(nside)

    np.save(cache_file, vectors)

    print(f"Saved sky vectors to:")
    print(cache_file)

    return vectors
