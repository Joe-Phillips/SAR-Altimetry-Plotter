import os
import sys
import numpy as np
from scipy import interpolate
from netCDF4 import Dataset
import pyproj


_required_keys = {
    "S3": {
        "l2": [
            "lat_20_ku", "lon_20_ku", "lat_cor_20_ku", "lon_cor_20_ku",
            "waveform_20_ku", "alt_20_ku", "elevation_ice_sheet_20_ku",
            "tracker_range_20_ku", "time_01", "time_20_ku", "surf_class_20_ku",
            "cog_cor_01", "mod_dry_tropo_cor_meas_altitude_01",
            "mod_wet_tropo_cor_meas_altitude_01", "iono_cor_gim_01_ku",
            "inv_bar_cor_01", "load_tide_sol2_01", "solid_earth_tide_01",
            "pole_tide_01",
        ],
        "l1": [],
    },
    "CS2": {
        "l2": [
            "lat_20_ku", "lon_20_ku", "lat_poca_20_ku", "lon_poca_20_ku",
            "alt_20_ku", "height_1_20_ku",
        ],
        "l1": [
            "pwr_waveform_20_ku", "window_del_20_ku", "ind_meas_1hz_20_ku",
            "mod_dry_tropo_cor_01", "mod_wet_tropo_cor_01", "iono_cor_gim_01",
            "hf_fluct_total_cor_01", "ocean_tide_01", "ocean_tide_eq_01",
            "load_tide_01", "solid_earth_tide_01", "pole_tide_01", "surf_type_01",
        ],
    },
}


def validate_epsg(epsg_string):
    """
    Check that an EPSG string is valid and uses metre units. Prints and exits on failure.

    Args:
        epsg_string (str): Projection string in EPSG format, e.g. 'epsg:3031'.
    """
    try:
        crs = pyproj.CRS.from_user_input(epsg_string)
    except Exception:
        print(f"Error: Could not parse projection: '{epsg_string}'. Expected format e.g. 'epsg:3031'.")
        sys.exit(1)

    units = crs.axis_info[0].unit_name
    if "metre" not in units and "meter" not in units:
        print(f"Error: Projection '{epsg_string}' uses units '{units}' — must be in metres.")
        sys.exit(1)


def validate_nc(path, satellite, level):
    """
    Check that a NetCDF file exists, is readable, and contains required keys.

    Args:
        path (str): Path to the NetCDF file.
        satellite (str): "S3" or "CS2".
        level (str): "l1" or "l2".
    """
    if not os.path.isfile(path):
        print(f"Error: NetCDF file not found: {path}")
        sys.exit(1)
    if not path.lower().endswith(".nc"):
        print(f"Error: File does not appear to be a NetCDF (.nc): {path}")
        sys.exit(1)
    try:
        data = Dataset(path)
    except Exception as e:
        print(f"Error: Could not open file as NetCDF: {e}")
        sys.exit(1)

    required = _required_keys[satellite][level]
    missing = [k for k in required if k not in data.variables]
    if missing:
        print(f"Error: NetCDF file is missing required variables: {missing}")
        sys.exit(1)


def get_and_correct_range(data, record_range, satellite):
    """
    Get and correct range data from satellite measurements.

    Args:
        data (dict): Satellite track data.
        record_range (tuple): (start_record, end_record)
        satellite (str): "S3" or "CS2".

    Returns:
        numpy.ndarray: Corrected range data.
    """
    record_range_1hz = [record_range[0] // 20, max(1, int(np.ceil(record_range[1] / 20)))]

    if satellite == "S3":
        z = data["tracker_range_20_ku"][record_range[0]:record_range[1]]
        t_1hz = data["time_01"][record_range_1hz[0]:record_range_1hz[1]]
        t_20hz = data["time_20_ku"][record_range[0]:record_range[1]]
        surface_class = data["surf_class_20_ku"][record_range[0]:record_range[1]]

        cog = data["cog_cor_01"][record_range_1hz[0]:record_range_1hz[1]]
        dry_tropo = data["mod_dry_tropo_cor_meas_altitude_01"][record_range_1hz[0]:record_range_1hz[1]]
        wet_tropo = data["mod_wet_tropo_cor_meas_altitude_01"][record_range_1hz[0]:record_range_1hz[1]]
        iono = data["iono_cor_gim_01_ku"][record_range_1hz[0]:record_range_1hz[1]]
        ibe = data["inv_bar_cor_01"][record_range_1hz[0]:record_range_1hz[1]]

        try:
            hf_ibe = data["hf_fluct_cor_01"][record_range_1hz[0]:record_range_1hz[1]]
        except Exception:
            hf_ibe = np.zeros_like(ibe)

        try:
            ocean_tide = data["ocean_tide_sol2_01"][record_range_1hz[0]:record_range_1hz[1]]
        except Exception:
            ocean_tide = np.zeros_like(ibe)

        load_tide = data["load_tide_sol2_01"][record_range_1hz[0]:record_range_1hz[1]]
        solid_earth_tide = data["solid_earth_tide_01"][record_range_1hz[0]:record_range_1hz[1]]
        pole_tide = data["pole_tide_01"][record_range_1hz[0]:record_range_1hz[1]]

        def resample(values):
            return interpolate.interp1d(t_1hz, values, kind="linear", fill_value="extrapolate")(t_20hz.data)

        cog_20hz          = resample(cog)
        dry_tropo_20hz    = resample(dry_tropo)
        wet_tropo_20hz    = resample(wet_tropo)
        iono_20hz         = resample(iono)
        ibe_20hz          = resample(ibe)
        hf_ibe_20hz       = resample(hf_ibe)
        ocean_tide_20hz   = resample(ocean_tide)
        load_tide_20hz    = resample(load_tide)
        solid_earth_20hz  = resample(solid_earth_tide)
        pole_tide_20hz    = resample(pole_tide)

        geo_land = dry_tropo_20hz + wet_tropo_20hz + iono_20hz + load_tide_20hz + solid_earth_20hz + pole_tide_20hz
        geo_float = geo_land + ibe_20hz + hf_ibe_20hz + ocean_tide_20hz

        corrected_z_land  = z + cog_20hz + geo_land
        corrected_z_float = z + cog_20hz + geo_float

        corrected_z = np.zeros(len(t_20hz))
        for i in range(len(t_20hz)):
            # open_ocean, land, continental_water, aquatic_vegetation, continental_ice_snow, floating_ice, salted_basin
            if surface_class[i] in (1, 4, 5):
                corrected_z[i] = corrected_z_land[i]
            else:
                corrected_z[i] = corrected_z_float[i]

    elif satellite == "CS2":
        c = 299792458  # speed of light (m/s)
        z = 0.5 * c * data["window_del_20_ku"][record_range[0]:record_range[1]]
        inds = data["ind_meas_1hz_20_ku"][record_range[0]:record_range[1]]

        dry_tropo       = data["mod_dry_tropo_cor_01"][inds]
        wet_tropo       = data["mod_wet_tropo_cor_01"][inds]
        iono            = data["iono_cor_gim_01"][inds]
        dac             = data["hf_fluct_total_cor_01"][inds]
        ocean_tide      = data["ocean_tide_01"][inds]
        ocean_eq_tide   = data["ocean_tide_eq_01"][inds]
        load_tide       = data["load_tide_01"][inds]
        solid_earth     = data["solid_earth_tide_01"][inds]
        pole_tide       = data["pole_tide_01"][inds]
        surface_class   = data["surf_type_01"][inds]

        geo_land  = load_tide + solid_earth + pole_tide + dry_tropo + wet_tropo + iono
        geo_float = geo_land + ocean_tide + dac + ocean_eq_tide

        corrected_z_land  = z + geo_land
        corrected_z_float = z + geo_float

        corrected_z = np.zeros(len(inds))
        for i in range(len(inds)):
            # ocean, lake_enclosed_sea, ice, land
            if surface_class[i] == 3:
                corrected_z[i] = corrected_z_land[i]
            else:
                corrected_z[i] = corrected_z_float[i]

    return corrected_z


def get_leading_edge(
    waveform,
    tracker_range,
    reference_bin_index,
    smoothing_window_width,
    range_bin_size,
    wf_oversampling_factor,
):
    """
    Detect the leading edge in a waveform and compute its range.

    Args:
        waveform (numpy.ndarray): Input waveform data.
        tracker_range (float): Tracker range value.
        reference_bin_index (int): Index of the reference bin.
        smoothing_window_width (int): Width of the smoothing window.
        range_bin_size (float): Distance between each range bin (m).
        wf_oversampling_factor (int): Waveform oversampling factor.

    Returns:
        tuple: (le_index_start, le_index_end, le_range_start, le_range_end)
    """
    noise_threshold  = 0.3   # reject waveform if mean noise exceeds this
    le_threshold_id  = 0.05  # power must exceed noise by this to be a leading edge candidate
    le_threshold_dp  = 0.2   # minimum normalised amplitude change to accept a leading edge

    le_index_start = np.nan
    le_index_end   = np.nan
    le_range_start = np.nan
    le_range_end   = np.nan

    while True:
        wf_norm = waveform / max(waveform)

        # Pseudo-Gaussian smoothing (3 passes of sliding-average)
        kernel = np.ones(smoothing_window_width) / smoothing_window_width
        wf_smooth = np.convolve(np.convolve(np.convolve(wf_norm, kernel, mode="same"), kernel, mode="same"), kernel, mode="same")

        wf_sorted = np.sort(wf_norm)
        wf_noise_mean = np.mean(wf_sorted[:6])

        if wf_noise_mean > noise_threshold:
            break

        # Oversample via spline
        oversampling_interval = 1 / wf_oversampling_factor
        bin_indices = np.arange(0, len(waveform), oversampling_interval)
        wf_interp = interpolate.splev(bin_indices, interpolate.splrep(range(len(waveform)), wf_smooth))
        wf_interp_d1 = np.gradient(wf_interp, oversampling_interval)

        le_index_prev = 0
        le_dp = 0

        while le_index_prev < len(bin_indices) - wf_oversampling_factor:
            candidates = np.where(
                (wf_interp > (wf_noise_mean + le_threshold_id)) &
                (bin_indices > bin_indices[le_index_prev + wf_oversampling_factor])
            )

            if np.size(candidates) == 0:
                break
            le_idx = candidates[0][0]

            peaks = np.where((wf_interp_d1 <= 0) & (bin_indices > bin_indices[le_idx]))
            if np.size(peaks) == 0:
                break
            first_peak = peaks[0][0]

            le_dp = wf_interp[first_peak] - wf_interp[le_idx]
            le_index_prev = first_peak

            if le_dp > le_threshold_dp:
                le_index_start = le_idx / wf_oversampling_factor
                le_index_end   = first_peak / wf_oversampling_factor
                break

        if not np.isnan(le_index_start):
            le_range_start = tracker_range - (reference_bin_index - le_index_start) * range_bin_size
            le_range_end   = tracker_range + (le_index_end - reference_bin_index) * range_bin_size
        break

    return le_index_start, le_index_end, le_range_start, le_range_end


def get_heading(nadir_x, nadir_y):
    """
    Compute satellite track headings from nadir coordinates.

    Args:
        nadir_x (numpy.ndarray): X-coordinates of nadir points.
        nadir_y (numpy.ndarray): Y-coordinates of nadir points.

    Returns:
        numpy.ndarray: Heading angles in degrees for each record.
    """
    num_records = len(nadir_x)
    heading = np.full(num_records, np.nan)

    for i in range(num_records):
        if i == num_records - 1:
            dx = nadir_x[i] - nadir_x[i - 1]
            dy = nadir_y[i] - nadir_y[i - 1]
        else:
            dx = nadir_x[i + 1] - nadir_x[i]
            dy = nadir_y[i + 1] - nadir_y[i]

        inner = np.rad2deg(np.arctan(dx / dy))
        if dy < 0:
            heading[i] = inner + 180
        elif dy > 0 and dx < 0:
            heading[i] = inner + 360
        else:
            heading[i] = inner

    return heading