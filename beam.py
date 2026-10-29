"""LFT3 analytic and tabulated beam models.

Frames: local NEZ axes are X=North, Y=East, Z=Zenith.
Experiment axes coincide with NEZ for orientation=(0,0,0).
Angles are degrees for orientation, radians for beam evaluation.
All beam values are dimensionless power responses, peak-normalized
for analytic models; tabulated beams are used as supplied.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import healpy as hp
from scipy.interpolate import RegularGridInterpolator
from scipy.special import j1


@dataclass(frozen=True)
class NumericalBeam:
    """Tabulated B(frequency, theta, phi) on an angular grid.

    theta_rad must span the desired polar angles; phi_rad must be strictly
    increasing and periodic samples must not duplicate the 2*pi endpoint.
    """
    frequencies_mhz: np.ndarray
    theta_rad: np.ndarray
    phi_rad: np.ndarray
    power: np.ndarray

    def __post_init__(self):
        f = np.asarray(self.frequencies_mhz, dtype=float)
        th = np.asarray(self.theta_rad, dtype=float)
        ph = np.asarray(self.phi_rad, dtype=float)
        power = np.asarray(self.power, dtype=float)
        if any(x.ndim != 1 or len(x) < 2 or np.any(np.diff(x) <= 0)
               for x in (f, th, ph)):
            raise ValueError('Numerical beam axes must be increasing 1D arrays with >=2 samples')
        if power.shape != (len(f), len(th), len(ph)):
            raise ValueError('power must have shape (Nfreq, Ntheta, Nphi)')
        if np.any(~np.isfinite(power)) or np.any(power < 0):
            raise ValueError('power must be finite and nonnegative')
        if ph[0] < 0 or ph[-1] >= 2 * np.pi:
            raise ValueError('phi grid must lie in [0, 2*pi) without repeated endpoint')
        for name, value in [('frequencies_mhz', f), ('theta_rad', th),
                            ('phi_rad', ph), ('power', power)]:
            object.__setattr__(self, name, value)

    @classmethod
    def from_npz(cls, filename):
        """Load arrays named frequencies_mhz, theta_rad, phi_rad, power."""
        with np.load(Path(filename)) as data:
            return cls(*(data[k].copy() for k in
                         ('frequencies_mhz', 'theta_rad', 'phi_rad', 'power')))

    def evaluate(self, theta, phi, frequency_mhz):
        theta, phi = np.broadcast_arrays(np.asarray(theta, float),
                                         np.asarray(phi, float))
        f = float(frequency_mhz)
        if f < self.frequencies_mhz[0] or f > self.frequencies_mhz[-1]:
            raise ValueError('Frequency outside numerical beam grid')
        if np.any(theta < self.theta_rad[0]) or np.any(theta > self.theta_rad[-1]):
            raise ValueError('Theta outside numerical beam grid')
        # Periodic extension of the azimuthal dimension.
        extended_phi = np.r_[self.phi_rad[-1] - 2*np.pi,
                             self.phi_rad, self.phi_rad[0] + 2*np.pi]
        extended_power = np.concatenate((self.power[..., -1:], self.power,
                                         self.power[..., :1]), axis=-1)
        interpolator = RegularGridInterpolator(
            (self.frequencies_mhz, self.theta_rad, extended_phi),
            extended_power, bounds_error=True)
        pts = np.column_stack((np.full(theta.size, f), theta.ravel(),
                               np.mod(phi.ravel(), 2*np.pi)))
        return interpolator(pts).reshape(theta.shape)


def orientation_matrix(orientation=(0.0, 0.0, 0.0)):
    """Rotation from experiment coordinates into local NEZ coordinates."""
    alpha, beta, gamma = np.deg2rad(orientation)
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    rz_a = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    ry_b = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz_g = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    return rz_a @ ry_b @ rz_g


def antenna_frame_matrix(az_deg, el_deg):
    """
    Return a rotation matrix mapping antenna-frame
    vectors into experiment-frame coordinates.

    The antenna Z axis is the dipole axis or
    formed-beam boresight.

    The antenna X axis is chosen by projecting
    experiment Z onto the plane perpendicular
    to the antenna axis.

    For a vertical antenna axis, experiment X
    is used as the reference instead.
    """

    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)

    ez = np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])

    ez /= np.linalg.norm(ez)

    reference = np.array([0.0, 0.0, 1.0])

    if abs(np.dot(reference, ez)) > 0.999:
        reference = np.array([1.0, 0.0, 0.0])

    ex = reference - np.dot(reference, ez) * ez
    ex /= np.linalg.norm(ex)

    ey = np.cross(ez, ex)

    return np.column_stack((ex, ey, ez))


def beam_model(
    theta,
    phi,
    frequency_mhz,
    beam_type,
    numerical_beam=None,
    aperture_diameter_m=3.5,
):
    """
    Evaluate scalar power beam B(theta, phi, frequency).

    Parameters
    ----------
    theta : array-like
        Polar angle from the experiment Z axis [rad].

    phi : array-like
        Azimuth from experiment X toward Y [rad].

    frequency_mhz : float
        Observing frequency in MHz.

    beam_type : str
        "hf_dipole", "half_wave_dipole",
        "airy", or "numerical".

    numerical_beam : NumericalBeam, optional
        Loaded numerical beam model.

    aperture_diameter_m : float
        Aperture diameter for the analytic Airy model.

    Notes
    -----
    Analytic models reproduce the original LFT3
    beam prescriptions.

    The analytic beams are axisymmetric, so phi
    is not used.

    Numerical beams can depend on both theta and phi.
    """

    theta, phi = np.broadcast_arrays(
        np.asarray(theta, dtype=float),
        np.asarray(phi, dtype=float),
    )

    cos_theta = np.cos(theta)

    # ========================================================
    # HF: frequency-dependent dipole interpolation
    # ========================================================

    if beam_type == "hf_dipole":

        sin_theta = np.sqrt(
            np.maximum(
                0.0,
                1.0 - cos_theta**2
            )
        )

        small_field = sin_theta

        half_field = np.zeros_like(theta)
        full_field = np.zeros_like(theta)

        valid = sin_theta > 1e-12

        half_field[valid] = (
            np.cos(
                0.5 * np.pi * cos_theta[valid]
            )
            / sin_theta[valid]
        )

        full_field[valid] = (
            np.cos(
                np.pi * cos_theta[valid]
            ) + 1.0
        ) / (
            2.0 * sin_theta[valid]
        )

        freq = float(frequency_mhz)

        if freq < 10.0:

            field = small_field

        elif freq < 25.0:

            w = (freq - 10.0) / 15.0

            field = (
                (1.0 - w) * small_field
                + w * half_field
            )

        else:

            w = (freq - 25.0) / 25.0

            field = (
                (1.0 - w) * half_field
                + w * full_field
            )

        return field**2

    # ========================================================
    # VHF: half-wave dipole
    # ========================================================

    if beam_type == "half_wave_dipole":

        sin_theta = np.sqrt(
            np.maximum(
                0.0,
                1.0 - cos_theta**2
            )
        )

        field = np.zeros_like(theta)

        valid = sin_theta > 1e-12

        field[valid] = (
            np.cos(
                0.5 * np.pi * cos_theta[valid]
            )
            / sin_theta[valid]
        )

        return field**2

    # ========================================================
    # UHF: analytic Airy formed beam
    # ========================================================

    if beam_type == "airy":

        if aperture_diameter_m <= 0:
            raise ValueError(
                "Airy aperture diameter must be positive"
            )

        wavelength = 299.8 / float(frequency_mhz)

        sin_theta = np.sqrt(
            np.maximum(
                0.0,
                1.0 - cos_theta**2
            )
        )

        x = (
            np.pi
            * aperture_diameter_m
            / wavelength
            * sin_theta
        )

        beam = np.ones_like(x)

        valid = np.abs(x) > 1e-12

        beam[valid] = (
            2.0 * j1(x[valid]) / x[valid]
        )**2

        # Forward hemisphere only
        beam[cos_theta < 0.0] = 0.0

        return beam

    # ========================================================
    # Numerical beam
    # ========================================================

    if beam_type == "numerical":

        if numerical_beam is None:
            raise ValueError(
                "numerical_beam is required"
            )

        return numerical_beam.evaluate(
            theta,
            phi,
            frequency_mhz,
        )

    raise ValueError(
        f"Unknown beam type: {beam_type}"
    )



def get_beam(
    sky_vectors_icrs,
    nez_axes_icrs,
    frequency_mhz,
    beam_type,
    orientation=(0.0, 0.0, 0.0),
    antenna_orientation=None,
    **beam_kwargs,
):
    """
    Evaluate antenna power beam on ICRS sky directions.

    Parameters
    ----------
    sky_vectors_icrs : ndarray, shape (Npix, 3)
        Unit sky directions in ICRS/J2000.

    nez_axes_icrs : ndarray, shape (3, 3)
        Local North, East, Zenith unit vectors expressed
        in ICRS, stored as columns.

    frequency_mhz : float
        Observing frequency in MHz.

    beam_type : str
        Analytic or numerical beam type.

    orientation : tuple
        Experiment orientation (alpha, beta, gamma)
        in degrees relative to local NEZ.

    antenna_orientation : dict, optional
        Antenna axis orientation relative to the
        experiment frame:

            {
                "az_deg": ...,
                "el_deg": ...
            }

        If None, antenna Z is aligned with experiment Z.

    **beam_kwargs
        Additional arguments passed to beam_model().

    Returns
    -------
    ndarray, shape (Npix,)
        Scalar power beam response.
    """

    sky = np.asarray(
        sky_vectors_icrs,
        dtype=float,
    )

    nez = np.asarray(
        nez_axes_icrs,
        dtype=float,
    )

    if sky.ndim != 2 or sky.shape[1] != 3:
        raise ValueError(
            "sky_vectors_icrs must have shape (Npix, 3)"
        )

    if nez.shape != (3, 3):
        raise ValueError(
            "nez_axes_icrs must have shape (3, 3)"
        )

    if not np.all(np.isfinite(sky)):
        raise ValueError(
            "Sky directions must be finite"
        )

    if not np.all(np.isfinite(nez)):
        raise ValueError(
            "NEZ basis must be finite"
        )

    # ========================================================
    # 1. ICRS -> local lunar NEZ
    # ========================================================

    sky_nez = sky @ nez

    # ========================================================
    # 2. NEZ -> experiment frame
    # ========================================================

    R_exp = orientation_matrix(
        orientation
    )

    sky_exp = sky_nez @ R_exp

    # ========================================================
    # 3. Experiment -> antenna frame
    # ========================================================

    if antenna_orientation is None:

        sky_ant = sky_exp

    else:

        R_ant = antenna_frame_matrix(
            az_deg=antenna_orientation["az_deg"],
            el_deg=antenna_orientation["el_deg"],
        )

        sky_ant = sky_exp @ R_ant

    # ========================================================
    # 4. Antenna-frame angular coordinates
    # ========================================================

    theta = np.arccos(
        np.clip(
            sky_ant[:, 2],
            -1.0,
            1.0,
        )
    )

    phi = np.mod(
        np.arctan2(
            sky_ant[:, 1],
            sky_ant[:, 0],
        ),
        2.0 * np.pi,
    )

    # ========================================================
    # 5. Evaluate intrinsic beam model
    # ========================================================

    return beam_model(
        theta=theta,
        phi=phi,
        frequency_mhz=frequency_mhz,
        beam_type=beam_type,
        **beam_kwargs,
    )


def beam_directivity(
    frequency_mhz,
    beam_type,
    numerical_beam=None,
    integration_nside=128,
):
    """
    Calculate maximum directivity from the power beam.

    Parameters
    ----------
    frequency_mhz : float
        Frequency in MHz.

    beam_type : str
        Selected analytic or numerical beam type.

    numerical_beam : NumericalBeam or None
        Loaded numerical beam model.

    integration_nside : int
        HEALPix NSIDE for angular integration.

    Returns
    -------
    float
        Maximum beam directivity (dimensionless).
    """

    npix = hp.nside2npix(integration_nside)

    theta, phi = hp.pix2ang(
        integration_nside,
        np.arange(npix),
        nest=False,
    )

    beam = np.asarray(
        beam_model(
            theta,
            phi,
            frequency_mhz,
            beam_type=beam_type,
            numerical_beam=numerical_beam,
        ),
        dtype=float,
    )

    if beam.shape != (npix,):
        raise ValueError("Unexpected beam shape")

    if not np.all(np.isfinite(beam)):
        raise ValueError("Beam contains non-finite values")

    if np.any(beam < 0):
        raise ValueError("Power beam must be nonnegative")

    beam_integral = (
        4.0 * np.pi * np.mean(beam)
    )

    if beam_integral <= 0:
        raise ValueError("Beam integral must be positive")

    return (
        4.0 * np.pi
        * np.max(beam)
        / beam_integral
    )
    
    
def analytic_directivity(antenna_name):
    """
    Prescribed element directivity used in the
    original LFT3 analytic sensitivity model.
    """

    directivities = {
        "HF": 1.6,
        "VHF": 1.6,
        "UHF": 3.1,
    }

    try:
        return directivities[antenna_name]
    except KeyError:
        raise ValueError(
            f"Unknown antenna: {antenna_name}"
        )
    

def hf_backlobe_factor(frequency_mhz):
    """
    Empirical HF effective-area correction used in
    the original sensitivity model.

    Physical interpretation and numerical coefficients
    require validation against an antenna simulation.
    """
    a = 0.48 / 2499.0
    return 0.02 + a * (frequency_mhz**2 - 1.0)
