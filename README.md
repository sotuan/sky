# LFT3 Sky Temperature and Sensitivity Model

This project calculates the expected sky contribution to the LFT3 system temperature and derives two sensitivity measures as functions of frequency and time:

- effective area over system temperature, $A_{\rm eff}/T_{\rm sys}$, in $\text{m}^2$/K;
- system equivalent flux density (SEFD), in Jy.

The calculation uses the preliminary LFT3 beam models across HF, VHF and UHF bands, the GSM2016 sky temperature maps obtained with PyGDSM, and the lunar reference frame imported from SPICE.

## 1. Assumed LFT3 specifications

| Band | Frequency range | Step | Elements | Directivity | Receiver temperature | Beam model |
|---|---:|---:|---:|---:|---:|---|
| HF | 1-50 MHz | 1 MHz | 1 | 1.6 | 300 K | frequency-dependent dipole |
| VHF-LO | 60-110 MHz | 1 MHz | 1 | 1.6 | 100 K | half-wave dipole |
| VHF-HI | 150-250 MHz | 1 MHz | 1 | 1.6 | 100 K | half-wave dipole |
| UHF-LO | 300-900 MHz | 2 MHz | 48 | 3.1 | 30 K | Airy-like formed beam |
| UHF-HI | 900-2700 MHz | 10 MHz | 8 | 3.1 | 35 K | Airy-like formed beam |

The band definitions are in `lft3_specs.py`.

## 2. Time grid

The default calculation uses one sample per day over the half-year interval:

- start: `2028-07-01T00:00:00`
- stop: `2029-01-01T00:00:00`
- cadence: 1 day

The observing-time settings are in `config.py`.

## 3. Lunar site and local coordinates

The site latitude and longitude are set in `config.py`.

`lunar_geometry.py` constructs local North, East, and Zenith unit vectors in the Moon-fixed `MOON_ME` frame.

The azimuth/elevation convention is:

- azimuth = 0 deg: north;
- azimuth = 90 deg: east;
- elevation = 0 deg: local horizon;
- elevation = 90 deg: local zenith.

A local direction is converted to a Moon-fixed Cartesian vector, and SPICE is then used to rotate the fixed lunar frame direction into J2000 at every observing time.

The relevant SPICE kernels are:

- `naif0012.tls`
- `moon_pa_de440_200625.bpc`
- `moon_de440_250416.tf`

They are stored in `kernels/` unless `config.py` specifies another location.

## 4. Antenna orientations

Antenna orientations are specified in `config.py` in `ANTENNA_ORIENTATIONS`.

For HF and VHF, azimuth/elevation describe the dipole axis.

For UHF, azimuth/elevation describe the boresight of the simplified effective formed beam.

## 5. Beam models

The beam models are implemented in `beam.py`.

### HF

HF uses the simplified frequency-dependent dipole model.

### VHF

VHF-LO and VHF-HI use the same half-wave-dipole power pattern,

$$
B(\theta)=
\left[
\frac{\cos\left(\frac{\pi}{2}\cos\theta\right)}
{\sin\theta}
\right]^2,
$$

where $\theta$ is measured from the dipole axis.

### UHF

The UHF bands use a simplified effective Airy-like formed beam model,

$$
B(\alpha)=
\left[\frac{2J_1(x)}{x}\right]^2,
$$

with

$$
x=\frac{\pi D}{\lambda}\sin\alpha,
$$

where $\alpha$ is angular separation from the formed-beam boresight, and $D$ is the directivity.

## 6. Sky model

`sky_model.py` creates the GSM2016 sky model with PyGDSM.

The current setup uses:

- MHz as the frequency unit;
- Rayleigh-Jeans temperature (`TRJ`);
- PCHIP interpolation;
- CMB included.

For frequencies at or below 10 MHz, the code extrapolates from 11 MHz with spectral index -2.7:

$$
T(\nu)=T(11\,\mathrm{MHz})
\left(\frac{\nu}{11\,\mathrm{MHz}}\right)^{-2.7}.
$$

## 7. ICRS-coordinates of sky map pixels

`galactic_to_ICRF.py` builds a unit vector for every HEALPix pixel, converts the Galactic pixel directions into ICRS/J2000-compatible Cartesian coordinates, and caches the result in `cache/`.

For HEALPix,

$$
N_{\rm pix}=12 \mathrm{NSIDE}^2.
$$

Examples:

- NSIDE 32 -> 12,288 pixels
- NSIDE 64 -> 49,152 pixels
- NSIDE 128 -> 196,608 pixels

The working NSIDE is set in `config.py`.

## 8. Lunar horizon

For each observing time, the local zenith vector is transformed into J2000 using SPICE. A sky direction is considered visible if

$$
\hat{z}(t)\cdot\hat{s}>0.
$$

Directions with

$$
\hat{z}(t)\cdot\hat{s}\le0
$$

are blocked by the Moon and are currently assigned 0 K.

## 9. Sensitivity calculation

`sensitivity.py` performs the main numerical calculation.

For each band it:

1. calculates antenna-to-sky geometry;
2. creates the lunar-horizon mask;
3. evaluates the beam at every sky pixel;
4. integrates the beam-weighted sky temperature;
5. adds receiver temperature;
6. calculates effective area;
7. calculates $A_{\rm eff}/T_{\rm sys}$;
8. calculates SEFD.

## 10. Output files

`main.py` saves one compressed NumPy file for each band:

```text
HF_sensitivity.npz
VHF_LO_sensitivity.npz
VHF_HI_sensitivity.npz
UHF_LO_sensitivity.npz
UHF_HI_sensitivity.npz
```

Each file contains:

- `times_utc`
- `freqs_mhz`
- `Tsky_K`
- `Tsys_K`
- `Aeff_m2`
- `Ae_Tsys_m2_per_K`
- `SEFD_Jy`

The output arrays have shape

```text
(N_times, N_frequencies)
```

so, for example,

```python
result["SEFD_Jy"][i, j]
```

is the SEFD at time index `i` and frequency index `j`.

## 11. Sensitivity plots

`plot_sensitivity.py` generates two sensitivity plots.

### `SEFD_sensitivity_range.png`

For every frequency it shows:

- minimum SEFD over the observing period;
- maximum SEFD;
- the filled min-max envelope;
- median SEFD.

Both axes are logarithmic.

### `Ae_Tsys_sensitivity_range.png`

This shows the corresponding time envelope for

$$
A_{\rm eff}/T_{\rm sys}.
$$

Both axes are logarithmic.

## 12. Project structure

```text
SkyT/
|
|-- main.py
|-- config.py
|-- lft3_specs.py
|-- sky_model.py
|-- galactic_to_ICRF.py.py
|-- lunar_geometry.py
|-- beam.py
|-- sensitivity.py
|-- plot_sensitivity.py
|-- replot.py
|
|-- kernels/
|   |-- naif0012.tls
|   |-- moon_pa_de440_200625.bpc
|   `-- moon_de440_250416.tf
|
|-- cache/
|   `-- icrs_vectors_nside*.npy
|
|-- *_sensitivity.npz
|-- SEFD_sensitivity_range.png
`-- Ae_Tsys_sensitivity_range.png
```

### `config.py`
Site, time, NSIDE, antenna orientation, kernel, and model settings.

### `lft3_specs.py`
Frequency grids and simplified instrument parameters for the five bands.

### `sky_model.py`
GSM2016 sky generation and HF extrapolation.

### `galactic_to_ICRF.py`
HEALPix sky-vector generation, Galactic-to-ICRS conversion, and caching.

### `lunar_geometry.py`
Local lunar coordinates and SPICE frame transformations.

### `beam.py`
HF, VHF, and UHF beam models.

### `sensitivity.py`
Beam-weighted sky temperature, system temperature, effective area, $A_{\rm eff}/T_{\rm sys}$, and SEFD.

### `plot_sensitivity.py`
Plot min-max sensitivity envelopes over the full observing interval.

### `main.py`
Runs the full scientific pipeline.

### `replot.py`
Regenerates figures from saved numerical results.

## 13. Main dependencies

The project uses:

- NumPy
- SciPy
- Matplotlib
- Healpy
- Astropy
- PyGDSM
- SpiceyPy

## 14. Modeling assumptions

Current assumptions include:

1. The Moon is locally spherical (horizon masks exactly half of the sky).
2. The lunar surface is assumed to have zero radio brightness temperature.
3. Antenna orientations are fixed local azimuth/elevation values.
4. GSM2016 represents the diffuse radio sky, while the sky at <=10 MHz is extrapolated from 11 MHz with spectral index -2.7.
5. HF/VHF use idealized dipole patterns.
6. UHF uses an Airy-like effective formed beam rather than a full phased-array electromagnetic model.
7. Solar radio emission is not included.
