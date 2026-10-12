# LFT3 Sky Temperature and Sensitivity Model

This project calculates the expected sky contribution to the LFT3 system temperature and derives two sensitivity measures as functions of frequency and time:

- effective area over system temperature, in $\text{m}^2$/K;
- system equivalent flux density (SEFD), in Jy.

The calculation uses the preliminary LFT3 beam models across HF, VHF and UHF bands, the GSM2016 sky temperature maps obtained with PyGDSM, the low-frequency sky maps obtained with ULSA, and the lunar reference frame imported from SPICE.

## 0. Main formulae

For every observing time $t$ and frequency $\nu$ within a certain band, the program evaluates the beam-weighted sky temperature

$$
T_{\rm sky}(t,\nu)=
\frac{\int B(\hat{s},t,\nu)T_{\rm env}(\hat{s},t,\nu)d\Omega}
{\int B(\hat{s},t,\nu)d\Omega} ,
$$

where  $B(\hat{s},t,\nu)$ is the beam function that depends on the antenna orientations/beam pointing.

The environment temperature is currently modeled as

$$
T_{\rm env}=
\begin{cases}
T_{\rm GSM}, & \text{above the lunar horizon at time $t$},\\
0, & \text{otherwise}.
\end{cases}
$$

The system temperature is obtained by adding the receiver temperature

$$
T_{\rm sys}(t,\nu)=T_{\rm sky}(t,\nu)+T_{\rm rcvr}(\nu) ,
$$

from which one obtains the sensitivity measures:

$$
\frac{A_{\rm eff}}{T_{\rm sys}}
$$

and

$$
{\rm SEFD}=
\frac{2k_B (T_{\rm sys}/ \text{K})}{A_{\rm eff}/\text{m}^2}10^{26}\ {\text{Jy}}.
$$

## 1. Assumed LFT3 specifications

| Band | Frequency range | Step | Elements | Directivity | Receiver temperature | Beam model |
|---|---:|---:|---:|---:|---:|---|
| HF | 1-50 MHz | 1 MHz | 1 | 1.6 | 300 K | frequency-dependent dipole |
| VHF | 60-250 MHz | 1 MHz | 1 | 1.6 | 100 K | half-wave dipole |
| UHF | 300-2700 MHz | 2 MHz | 48 | 3.1 | 30 K | Airy-like formed beam |

The band definitions are in `lft3_specs.py`.

## 2. Time grid

The default calculation uses one sample per day over the half-year interval:

- start: `2028-07-01T00:00:00`
- stop: `2029-01-01T00:00:00`
- step: 1 day

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

They are stored in `SPICE_kernels/` unless `config.py` specifies another location.

## 4. Antenna orientations

Antenna orientations are specified in `config.py` in `ANTENNA_ORIENTATIONS`.

For HF and VHF, azimuth/elevation describe the dipole axis.

For UHF, azimuth/elevation describe the boresight of the simplified effective formed beam.

## 5. Beam models

The beam models are implemented in `beam.py`.

### HF

HF uses the simplified frequency-dependent dipole model.

### VHF

VHF uses the half-wave-dipole power pattern,

$$
B(\theta)=
\left[
\frac{\cos\left(\frac{\pi}{2}\cos\theta\right)}
{\sin\theta}
\right]^2,
$$

where $\theta$ is measured from the dipole axis.

### UHF

The UHF band uses a simplified effective Airy-like formed beam model,

$$
B(\alpha)=
\left[\frac{2J_1(x)}{x}\right]^2,
$$

with

$$
x=\frac{\pi D}{\lambda}\sin\alpha,
$$

where $\alpha$ is angular separation from the formed-beam boresight, and $D$ is the effective aperture diameter.

## 6. Sky model

### 6.1 GSM2016 at 10 MHz and above

`sky_model.py` creates `pygdsm.GlobalSkyModel16` with the settings:

```python
GlobalSkyModel16(
    freq_unit="MHz",
    data_unit="TRJ",
    resolution="low",
    interpolation="pchip",
    include_cmb=True,
)
```

The generated map is converted to the working HEALPix `NSIDE` when necessary. Sky temperatures are in Rayleigh–Jeans kelvin. The maps and derived sky vectors are interpreted in Galactic HEALPix RING ordering.

### 6.2 ULSA maps from 1 to 10 MHz

The [Ultra-long Wavelength Sky Model with Absorption (ULSA)](https://github.com/Yanping-Cong/ULSA), described by [Cong et al. (2021)](https://arxiv.org/abs/2104.03170), is used to model the low-frequency spectral and spatial effects of Galactic free–free absorption. This improves on extrapolating to lower frequencies via a single synchrotron power law.

The repository contains ten precomputed raw ULSA sky-temperature maps:

```text
data/ulsa_maps/
    ulsa_1MHz_nside64.npy
    ulsa_2MHz_nside64.npy
    ...
    ulsa_10MHz_nside64.npy
```

Each `.npy` file is a one-dimensional array of $12(64)^2=49\,152$ temperatures in kelvin, with Galactic coordinates and HEALPix RING ordering. The two-column output returned by the original ULSA `mpi()` method (pixel index and temperature) was converted to this one-dimensional format when caching the maps.

The maps were generated using the following ULSA parameters:

```python
absorption_JRZ(
    v=frequency_mhz,
    nside=64,
    index_type="constant_index",
    distance=50.0,             # kpc
    using_raw_diffuse=False,
    using_default_params=True,
    input_spectral_index=None,
    critical_dis=False,
    output_absorp_free_skymap=False,
)
```

Here `constant_index` refers to the model's intrinsic synchrotron spectral-index option. ULSA map generation requires its external code, observational input maps (including Haslam 408 MHz), and the NE2001 Galactic electron-density calculation.

### 6.3 Matching ULSA to GSM2016 at 10 MHz

The default hybrid model is selected by setting

```python
LOW_FREQ_SKY_MODEL = "ulsa"
```

in `config.py`. For frequency $1\leq \nu \leq 10$ MHz and sky pixel $p$, `ulsa_sky.py` uses

$$
\boxed{
T_{\rm hybrid}(\nu,p)=T_{\rm CMB}+
\big[T_{\rm GSM}(10 \mathrm{MHz},p)-T_{\rm CMB}\big]
\frac{T_{\rm ULSA}(\nu,p)}{T_{\rm ULSA}(10 \mathrm{MHz},p)}
}
$$

with $T_{\rm CMB}=2.7255$ K. For $\nu\geq10$ MHz, the pipeline uses GSM2016 directly.

The cached ULSA maps lie on a 1-MHz grid. For noninteger frequencies between 1 and 10 MHz, `ulsa_sky.py` implements per-pixel interpolation in logarithmic temperature versus logarithmic frequency. The standard HF grid currently uses integer MHz values.

### 6.4 Power-law comparison

The previous model extrapolated the 11-MHz GSM2016 map down to lower frequencies with a fixed power law:

$$
T_{\rm old}(\nu,p)=T_{\rm GSM}(11 \mathrm{MHz},p)
\left(\frac{\nu}{11 \mathrm{MHz}}\right)^{-2.7},
\qquad \nu\leq10 \mathrm{MHz}.
$$

This option remains available with

```python
LOW_FREQ_SKY_MODEL = "powerlaw"
```

## 7. ICRS-coordinates of sky map pixels

`galactic_to_ICRF.py` builds a unit vector for every HEALPix pixel, converts the Galactic pixel directions into ICRS/J2000-compatible Cartesian coordinates, and caches the result in `cache/`.

For HEALPix,

$$
N_{\rm pix}=12\cdot \mathrm{NSIDE}^2.
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

`main.py` saves one compressed NumPy file for each band in results/sensitivity:

```text
HF_sensitivity_ulsa.npz
VHF_sensitivity_ulsa.npz
UHF_sensitivity_ulsa.npz
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
sky/
|
|-- main.py
|-- config.py
|-- lft3_specs.py
|-- sky_model.py
|-- ulsa_sky
|-- galactic_to_ICRF.py
|-- lunar_geometry.py
|-- beam.py
|-- sensitivity.py
|-- plot_sensitivity.py
|
|-- SPICE_kernels/
|   |-- naif0012.tls
|   |-- moon_pa_de440_200625.bpc
|   `-- moon_de440_250416.tf
|
|-- data/ulsa_maps/
    |-- ulsa_1MHz_nside64.npy
    |-- ulsa_2MHz_nside64.npy
    ...
    |-- ulsa_10MHz_nside64.npy
|
|-- scripts/
    |-- generate_ulsa_maps.py
    |-- plot_ulsa.py
    |-- plot_gsm_sky.py
    |-- compare_sky_models.py
    |-- check_hf_transition.py
    |-- replot.py
```

**Module responsibilities:** `main.py` orchestrates the run; `config.py` holds run parameters; `lft3_specs.py` defines bands and instrument parameters; `sky_model.py` selects the GSM2016/ULSA/power-law input; `ulsa_sky.py` loads and interpolates cached ULSA maps; `galactic_to_ICRF.py` and `lunar_geometry.py` perform the sky and lunar geometry; `beam.py` specifies the simplified beam models; `sensitivity.py` calculates temperatures and sensitivity; and `plot_sensitivity.py` plots results.

The `scripts/` folder contains standalone diagnostics, data-generation, comparison, and replotting tools. These are not imported by the main scientific pipeline.

## 13. Main dependencies

The project uses:

- NumPy
- SciPy
- Matplotlib
- Healpy
- Astropy
- PyGDSM
- SpiceyPy
- ULSA

## 14. Modeling assumptions

Current assumptions include:

1. The Moon is locally spherical (horizon masks exactly half of the sky).
2. The lunar surface is assumed to have zero radio brightness temperature.
3. Antenna orientations are fixed local azimuth/elevation values.
4. GSM2016 represents the diffuse radio sky above 10 MHz, while the sky temperature at frequencies below 10 MHz is calculated using ULSA, with the account for free-free absorption.
5. HF/VHF use idealized dipole patterns.
6. UHF uses an Airy-like effective formed beam rather than a full phased-array electromagnetic model.
7. Solar radio emission is not included.
