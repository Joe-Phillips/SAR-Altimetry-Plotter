# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

import numpy as np
from netCDF4 import Dataset
import pyproj as proj
import matplotlib as mpl
from scipy import interpolate
import multiprocessing as mp
from tifffile import TiffFile, imread
from scipy.spatial.transform import Rotation
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import argparse

# ----------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------


def get_dem_segment(RASTER_DIR, segment_bounds, GRID_XY=True, FLATTEN=False):

    """
    Read in and subset raster data to given bounds.

    Args:
        RASTER_DIR (str): The path to the raster data file.
        segment_bounds (list): A list containing two tuples representing the bounds:
            - [(minx, maxx), (miny, maxy)] where minx, maxx, miny, and maxy are numeric values.
        GRID_XY (bool, optional): If True, the x and y coordinates will be returned as grids.
            If False, they will be returned as vectors. Defaults to True.
        FLATTEN (bool, optional): If True, the raster data will be flattened. If False,
            the raster data will be returned as is. Defaults to False.

    Returns:
        tuple: A tuple containing the following elements:
            - x (numpy.ndarray): The x-coordinates corresponding to the subset of raster data.
            - y (numpy.ndarray): The y-coordinates corresponding to the subset of raster data.
            - raster_data (numpy.ndarray): The subset of raster data within the specified bounds.
            - RESOLUTION (float): The resolution of the raster data.

    Note:
        - The function reads in the raster data from the specified path and subsets it
          based on the provided segment bounds.
        - If GRID_XY is True, x and y will be 2D grids; otherwise, they will be 1D vectors.
        - If FLATTEN is True, the raster data will be flattened; otherwise, it will be returned
          as a multi-dimensional array.
    """

    # ----------------------------------------------------------------------
    # Get raster and associated information
    # ----------------------------------------------------------------------

    raster = TiffFile(RASTER_DIR).pages[0].tags

    NUM_X = raster["ImageWidth"].value
    NUM_Y = raster["ImageLength"].value
    RESOLUTION = raster["ModelPixelScaleTag"].value[0]
    NUM_BANDS = len(TiffFile(RASTER_DIR).pages)
    X_MIN = raster["ModelTiepointTag"].value[3]
    Y_MAX = raster["ModelTiepointTag"].value[4]
    X_MAX = X_MIN + NUM_X * RESOLUTION
    Y_MIN = Y_MAX - NUM_Y * RESOLUTION

    # ----------------------------------------------------------------------
    # Obtain raster coordinates
    # ----------------------------------------------------------------------

    x = np.linspace(X_MIN, X_MAX, NUM_X, endpoint=False)
    y = np.linspace(Y_MIN, Y_MAX, NUM_Y, endpoint=False)
    y = np.flip(y)

    # ----------------------------------------------------------------------
    # Get segment bound coordinates as indices
    # ----------------------------------------------------------------------

    X_MIN_IND = (np.absolute(segment_bounds[0][0] - x - RESOLUTION)).argmin()
    X_MAX_IND = (np.absolute(segment_bounds[0][1] - x + RESOLUTION)).argmin()
    Y_MIN_IND = (np.absolute(segment_bounds[1][0] - y - RESOLUTION)).argmin()
    Y_MAX_IND = (np.absolute(segment_bounds[1][1] - y + RESOLUTION)).argmin()

    # ----------------------------------------------------------------------
    # Crop full raster coords to segment bounds
    # ----------------------------------------------------------------------

    x = x[X_MIN_IND:X_MAX_IND]
    y = y[Y_MAX_IND:Y_MIN_IND]

    raster = np.full((len(y), len(x), NUM_BANDS), np.nan, dtype=np.float32)
    for BAND in range(NUM_BANDS):
        raster_current_band = imread(RASTER_DIR, key=BAND)
        raster[:, :, BAND] = raster_current_band[
            Y_MAX_IND:Y_MIN_IND, X_MIN_IND:X_MAX_IND
        ]

    # ----------------------------------------------------------------------
    # Convert x,y vectors to grids if requested
    # ----------------------------------------------------------------------

    if GRID_XY == True:
        x, y = np.meshgrid(x, y)

    # ----------------------------------------------------------------------
    # Return, flattened if requested
    # ----------------------------------------------------------------------

    if FLATTEN == False:
        return x, y, raster.squeeze(), RESOLUTION
    else:
        raster = np.array(
            [
                raster[:, :, BAND_INDEX].flatten()
                for BAND_INDEX in range(np.shape(raster)[-1])
            ]
        )
        return x.flatten(), y.flatten(), raster.squeeze(), RESOLUTION


def get_dem_in_fp(
    dem_x,
    dem_y,
    dem_z,
    ACROSS_TRACK_WIDTH,
    ALONG_TRACK_WIDTH,
    heading,
    nadir_x,
    nadir_y,
    altitude,
    range_to_window_top,
    range_to_window_bottom,
    range_to_window_top_le,
    range_to_window_bottom_le,
):

    """
    Extract DEM (Digital Elevation Model) data within specified footprints.

    Args:
        dem_x (numpy.ndarray): Array of x-coordinates of DEM data points.
        dem_y (numpy.ndarray): Array of y-coordinates of DEM data points.
        dem_z (numpy.ndarray): Array of elevation values (DEM data).
        ACROSS_TRACK_WIDTH (float): Width of the footprint across the track.
        ALONG_TRACK_WIDTH (float): Width of the footprint along the track.
        heading (float): Heading angle in degrees.
        nadir_x (float): X-coordinate of the nadir point.
        nadir_y (float): Y-coordinate of the nadir point.
        altitude (float): Altitude of the satellite.
        range_to_window_top (float): Range distance to the top of the window.
        range_to_window_bottom (float): Range distance to the bottom of the window.
        range_to_window_top_le (float or None): Range distance to the top of the leading edge window.
        range_to_window_bottom_le (float or None): Range distance to the bottom of the leading edge window.

    Returns:
        list: A list containing three elements:
            - dem_in_fp_xy (tuple): A tuple of arrays (x, y, z) representing DEM data points
              within the specified footprint. Contains NaN if no points are within the footprint.
            - dem_in_fp_xyz (tuple): A tuple of arrays (x, y, z) representing DEM data points
              within the specified range window. Contains NaN if no points are within the window.
            - dem_in_fp_xyz_LE (tuple): A tuple of arrays (x, y, z) representing DEM data points
              within the specified leading edge window. Contains NaN if no points are within the window.

    Notes:
        - This function extracts DEM data points within specified footprints based on provided
          parameters such as footprint dimensions, orientation, satellite location, and range windows.
        - The function returns tuples of arrays, with NaN values indicating empty results for
          each specified condition.
    """

    # ----------------------------------------------------------------------
    # Initialise outputs
    # ----------------------------------------------------------------------

    dem_in_fp_xy = (np.full(1, np.nan), np.full(1, np.nan), np.full(1, np.nan))
    dem_in_fp_xyz = (np.full(1, np.nan), np.full(1, np.nan), np.full(1, np.nan))
    dem_in_fp_xyz_LE = (np.full(1, np.nan), np.full(1, np.nan), np.full(1, np.nan))

    # ----------------------------------------------------------------------
    # Create xy beam footprint
    # ----------------------------------------------------------------------

    # Intialise beam footprint in xy with centre at 0,0 as mpl path object
    fp_xy = mpl.path.Path(
        [
            (-ALONG_TRACK_WIDTH / 2, -ACROSS_TRACK_WIDTH / 2),
            (-ALONG_TRACK_WIDTH / 2, ACROSS_TRACK_WIDTH / 2),
            (ALONG_TRACK_WIDTH / 2, ACROSS_TRACK_WIDTH / 2),
            (ALONG_TRACK_WIDTH / 2, -ACROSS_TRACK_WIDTH / 2),
        ]
    )

    fp_xy = fp_xy.transformed(
        mpl.transforms.Affine2D().rotate_deg(90 - heading)
    )  # rotate to heading
    fp_xy = fp_xy.transformed(
        mpl.transforms.Affine2D().translate(nadir_x, nadir_y)
    )  # translate to nadir

    # ----------------------------------------------------------------------
    # Subset dem_x and dem_y to record area for faster running of contains_points()
    # ----------------------------------------------------------------------

    fp_xy_verts = fp_xy.vertices.copy()  # get footprint vertices

    # get fp vertex bounds
    fp_X_MAX = np.max(fp_xy_verts[:, 0])
    fp_X_MIN = np.min(fp_xy_verts[:, 0])
    fp_Y_MAX = np.max(fp_xy_verts[:, 1])
    fp_Y_MIN = np.min(fp_xy_verts[:, 1])

    # find index of points in DEM that are inside of fp bounds
    inds_in_fp_bounds = np.where(
        np.logical_and(dem_x >= fp_X_MIN, dem_x <= fp_X_MAX)
        & np.logical_and(dem_y >= fp_Y_MIN, dem_y <= fp_Y_MAX)
    )

    # subset
    dem_x = dem_x[inds_in_fp_bounds]
    dem_y = dem_y[inds_in_fp_bounds]
    dem_z = dem_z[inds_in_fp_bounds]

    # ----------------------------------------------------------------------
    # Identify indices of dem segment coords that are within xy beam footprint (top down)
    # ----------------------------------------------------------------------

    in_fp_mask = fp_xy.contains_points(
        np.column_stack((dem_x, dem_y))
    )  # Get boolean array checking whether dem coords are within the footprint
    inds_in_fp = np.nonzero(
        in_fp_mask == 1
    )  # get indices of coords that are within xy footprint

    # ----------------------------------------------------------------------
    # Extract DEM within xy footprint bounds
    # ----------------------------------------------------------------------

    dem_x_in_fp = dem_x[inds_in_fp]
    dem_y_in_fp = dem_y[inds_in_fp]
    dem_z_in_fp = dem_z[inds_in_fp]

    # Check whether there are any points within the xy footprint
    if len(dem_z_in_fp) == 0:
        return [dem_in_fp_xy, dem_in_fp_xyz, dem_in_fp_xyz_LE]
    else:
        dem_in_fp_xy = [dem_x_in_fp, dem_y_in_fp, dem_z_in_fp]

    # ----------------------------------------------------------------------
    # Find distance of points to sat
    # ----------------------------------------------------------------------

    dem_in_fp = np.column_stack(
        [dem_x_in_fp, dem_y_in_fp, dem_z_in_fp]
    )  # zip dem coords
    satellite_location = np.column_stack(
        [nadir_x, nadir_y, altitude]
    )  # zip sat locatation coords
    point_to_sat_vec = (
        dem_in_fp - satellite_location
    )  # get vector from sat to each dem point
    point_to_sat_dist = np.linalg.norm(
        point_to_sat_vec, axis=-1
    )  # convert to distances

    # ----------------------------------------------------------------------
    # Identify indices of dem segment coords that are within the range window and leading edge
    # ----------------------------------------------------------------------

    inds_in_fp_rw = np.where(
        np.logical_and(
            point_to_sat_dist >= range_to_window_top,
            point_to_sat_dist <= range_to_window_bottom,
        )
    )

    # check leading edge found
    if np.isnan(range_to_window_top_le).all():
        inds_in_fp_le = []
    else:
        inds_in_fp_le = np.where(
            np.logical_and(
                point_to_sat_dist >= range_to_window_top_le,
                point_to_sat_dist <= range_to_window_bottom_le,
            )
        )

    # ----------------------------------------------------------------------
    # Extract DEM within range window and leading edge
    # ----------------------------------------------------------------------

    if np.size(inds_in_fp_rw) > 0:
        dem_x_in_fp_xyz = dem_x_in_fp[inds_in_fp_rw]
        dem_y_in_fp_xyz = dem_y_in_fp[inds_in_fp_rw]
        dem_z_in_fp_xyz = dem_z_in_fp[inds_in_fp_rw]

        dem_in_fp_xyz = [dem_x_in_fp_xyz, dem_y_in_fp_xyz, dem_z_in_fp_xyz]

    if np.size(inds_in_fp_le) > 0:
        dem_x_in_fp_xyz_LE = dem_x_in_fp[inds_in_fp_le]
        dem_y_in_fp_xyz_LE = dem_y_in_fp[inds_in_fp_le]
        dem_z_in_fp_xyz_LE = dem_z_in_fp[inds_in_fp_le]

        dem_in_fp_xyz_LE = [dem_x_in_fp_xyz_LE, dem_y_in_fp_xyz_LE, dem_z_in_fp_xyz_LE]

    # ----------------------------------------------------------------------
    # Return results
    # ----------------------------------------------------------------------

    return [dem_in_fp_xy, dem_in_fp_xyz, dem_in_fp_xyz_LE]


def get_and_correct_range(data, record_range, SATELLITE):

    """
    Get and correct range data from satellite measurements.

    Args:
        data (dict): Satellite track data.
        record_range (list): A list containing two integers representing the range of records to process.
        SATELLITE (str): The satellite type, either "S3" or "CS2".

    Returns:
        numpy.ndarray: An array of corrected range data based on satellite measurements.

    Notes:
        - This function takes satellite measurement data, including corrections, and extracts and
          corrects the range data.
        - The correction methods and satellite-specific processing differ between S3 and CS2,
          and the corrected range is returned accordingly.
    """

    record_range_1hz = np.array(record_range) // 20

    if SATELLITE == "S3":

        # ----------------------------------------------------------------------
        # Read in data
        # ----------------------------------------------------------------------

        z = data["tracker_range_20_ku"][record_range[0] : record_range[1]]
        t0000_1hz = data["time_01"][record_range_1hz[0] : record_range_1hz[1]]
        t0000 = data["time_20_ku"][record_range[0] : record_range[1]]
        surface_class = data["surf_class_20_ku"][record_range[0] : record_range[1]]
        COG_correction = data["cog_cor_01"][record_range_1hz[0] : record_range_1hz[1]]
        dry_tropospheric_correction = data["mod_dry_tropo_cor_meas_altitude_01"][
            record_range_1hz[0] : record_range_1hz[1]
        ]
        wet_tropospheric_correction = data["mod_wet_tropo_cor_meas_altitude_01"][
            record_range_1hz[0] : record_range_1hz[1]
        ]
        ionospheric_correction = data["iono_cor_gim_01_ku"][
            record_range_1hz[0] : record_range_1hz[1]
        ]
        IBE_correction = data["inv_bar_cor_01"][
            record_range_1hz[0] : record_range_1hz[1]
        ]
        HF_IBE_correction = data["hf_fluct_cor_01"][
            record_range_1hz[0] : record_range_1hz[1]
        ]
        ocean_tide = data["ocean_tide_sol2_01"][
            record_range_1hz[0] : record_range_1hz[1]
        ]
        load_tide = data["load_tide_sol2_01"][record_range_1hz[0] : record_range_1hz[1]]
        solid_earth_tide = data["solid_earth_tide_01"][
            record_range_1hz[0] : record_range_1hz[1]
        ]
        pole_tide = data["pole_tide_01"][record_range_1hz[0] : record_range_1hz[1]]

        # ----------------------------------------------------------------------
        # Resample 1 Hz data to 20 Hz
        # ----------------------------------------------------------------------

        COG_correction_20hz = interpolate.interp1d(
            t0000_1hz, COG_correction, kind="linear", fill_value="extrapolate"
        )(t0000.data)
        dry_tropospheric_correction_20hz = interpolate.interp1d(
            t0000_1hz,
            dry_tropospheric_correction,
            kind="linear",
            fill_value="extrapolate",
        )(t0000.data)
        wet_tropospheric_correction_20hz = interpolate.interp1d(
            t0000_1hz,
            wet_tropospheric_correction,
            kind="linear",
            fill_value="extrapolate",
        )(t0000.data)
        ionospheric_correction_20hz = interpolate.interp1d(
            t0000_1hz, ionospheric_correction, kind="linear", fill_value="extrapolate"
        )(t0000.data)
        IBE_correction_20hz = interpolate.interp1d(
            t0000_1hz, IBE_correction, kind="linear", fill_value="extrapolate"
        )(t0000.data)
        HF_IBE_correction_20hz = interpolate.interp1d(
            t0000_1hz, HF_IBE_correction, kind="linear", fill_value="extrapolate"
        )(t0000.data)
        ocean_tide_20hz = interpolate.interp1d(
            t0000_1hz, ocean_tide, kind="linear", fill_value="extrapolate"
        )(t0000.data)
        load_tide_20hz = interpolate.interp1d(
            t0000_1hz, load_tide, kind="linear", fill_value="extrapolate"
        )(t0000.data)
        solid_earth_tide_20hz = interpolate.interp1d(
            t0000_1hz, solid_earth_tide, kind="linear", fill_value="extrapolate"
        )(t0000.data)
        pole_tide_20hz = interpolate.interp1d(
            t0000_1hz, pole_tide, kind="linear", fill_value="extrapolate"
        )(t0000.data)

        # ----------------------------------------------------------------------
        # Compute total geophysical correction
        # ----------------------------------------------------------------------

        geophysical_correction_land_20hz = (
            dry_tropospheric_correction_20hz
            + wet_tropospheric_correction_20hz
            + ionospheric_correction_20hz
            + load_tide_20hz
            + solid_earth_tide_20hz
            + pole_tide_20hz
        )  # default includes land ice corrections only
        geophysical_correction_float_20hz = (
            geophysical_correction_land_20hz
            + IBE_correction_20hz
            + HF_IBE_correction_20hz
            + ocean_tide_20hz
        )  # compute version for floating ice including tide and ibe

        # ----------------------------------------------------------------------
        # Apply corrections
        # ----------------------------------------------------------------------

        corrected_z_land = (
            z + COG_correction_20hz + geophysical_correction_land_20hz
        )  # add corrections to range for land ice
        corrected_z_float = (
            z + COG_correction_20hz + geophysical_correction_float_20hz
        )  # compute version for floating ice including tide and ibe corrections

        # ----------------------------------------------------------------------
        # Merge land and float z values
        # ----------------------------------------------------------------------

        corrected_z = np.zeros(len(t0000))
        for i in range(len(t0000)):

            # open_ocean, land, continental_water, aquatic_vegetation, continental_ice_snow, floating_ice, salted_basin
            if surface_class[i] == 1 or surface_class[i] == 4 or surface_class[i] == 5:
                corrected_z[i] = corrected_z_land[i]
            else:
                corrected_z[i] = corrected_z_float[i]

    elif SATELLITE == "CS2":

        # ----------------------------------------------------------------------
        # Read in data
        # ----------------------------------------------------------------------
        C = 299792458  # speed of light (m/s)
        z = 0.5 * C * data["window_del_20_ku"][record_range[0] : record_range[1]]
        inds_1hz_to_20hz = data["ind_meas_1hz_20_ku"][record_range[0] : record_range[1]]
        dry_tropospheric_correction = data["mod_dry_tropo_cor_01"]
        wet_tropospheric_correction = data["mod_wet_tropo_cor_01"]
        ionospheric_correction = data["iono_cor_gim_01"]
        DAC = data["hf_fluct_total_cor_01"]
        ocean_tide = data["ocean_tide_01"]
        ocean_eq_tide = data["ocean_tide_eq_01"]
        load_tide = data["load_tide_01"]
        solid_earth_tide = data["solid_earth_tide_01"]
        pole_tide = data["pole_tide_01"]
        surface_class = data["surf_type_01"]

        # ----------------------------------------------------------------------
        # Resample 1 Hz data to 20 Hz
        # ----------------------------------------------------------------------

        dry_tropospheric_correction_20hz = dry_tropospheric_correction[inds_1hz_to_20hz]
        wet_tropospheric_correction_20hz = wet_tropospheric_correction[inds_1hz_to_20hz]
        ionospheric_correction_20hz = ionospheric_correction[inds_1hz_to_20hz]
        DAC_20hz = DAC[inds_1hz_to_20hz]
        ocean_tide_20hz = ocean_tide[inds_1hz_to_20hz]
        ocean_eq_tide_20hz = ocean_eq_tide[inds_1hz_to_20hz]
        load_tide_20hz = load_tide[inds_1hz_to_20hz]
        solid_earth_tide_20hz = solid_earth_tide[inds_1hz_to_20hz]
        pole_tide_20hz = pole_tide[inds_1hz_to_20hz]
        surface_class_20hz = surface_class[
            inds_1hz_to_20hz
        ]  # this is very low resolution (~6km)

        # ----------------------------------------------------------------------
        # Compute total geophysical corrections
        # ----------------------------------------------------------------------

        geophysical_correction_land_20hz = (
            load_tide_20hz
            + solid_earth_tide_20hz
            + pole_tide_20hz
            + dry_tropospheric_correction_20hz
            + wet_tropospheric_correction_20hz
            + ionospheric_correction_20hz
        )
        geophysical_correction_float_20hz = (
            geophysical_correction_land_20hz
            + ocean_tide_20hz
            + DAC_20hz
            + ocean_eq_tide_20hz
        )

        # ----------------------------------------------------------------------
        # Apply corrections
        # ----------------------------------------------------------------------

        corrected_z_land = (
            z + geophysical_correction_land_20hz
        )  # add corrections to range for land
        corrected_z_float = (
            z + geophysical_correction_float_20hz
        )  # compute version for floating ice and water

        # ----------------------------------------------------------------------
        # Merge land and float z values
        # ----------------------------------------------------------------------

        corrected_z = np.zeros(len(inds_1hz_to_20hz))
        for i in range(len(inds_1hz_to_20hz)):

            # ocean, lake_enclosed_sea, ice, land
            if surface_class_20hz[i] == 3:
                corrected_z[i] = corrected_z_land[i]
            else:
                corrected_z[i] = corrected_z_float[i]

    return corrected_z


def get_leading_edge(
    waveform,
    tracker_range,
    REFERENCE_BIN_INDEX,
    SMOOTHING_WINDOW_WIDTH,
    RANGE_BIN_SIZE,
    WF_OVERSAMPLING_FACTOR,
):

    """
    Get the leading edge in a waveform and compute its range.

    Args:
        waveform (numpy.ndarray): The input waveform data.
        tracker_range (float): The tracker range value.
        REFERENCE_BIN_INDEX (int): The index of the reference bin.
        SMOOTHING_WINDOW_WIDTH (int): Width of the smoothing window.
        RANGE_BIN_SIZE (float): The size of the range bin.
        WF_OVERSAMPLING_FACTOR (int): Oversampling factor for waveform processing.

    Returns:
        tuple: A tuple containing the following elements:
            - LE_INDEX_START (float): Index where the leading edge starts.
            - LE_INDEX_END (float): Index where the leading edge ends.
            - LE_RANGE_START (float): Range corresponding to the start of the leading edge.
            - LE_RANGE_END (float): Range corresponding to the end of the leading edge.

    Notes:
        - This function processes an input waveform to detect the leading edge and
          computes the corresponding range based on provided parameters.
        - The leading edge is identified based on specific amplitude and gradient criteria.
        - The returned indices are oversampled indices, and range values are computed accordingly.
    """

    # ----------------------------------------------------------------------
    # Define variables
    # ----------------------------------------------------------------------

    NOISE_THRESHOLD = (
        0.3  # if mean amplitude in noise bins exceeds threshold then reject waveform
    )
    LE_THRESHOLD_ID = 0.05  # power must be this much greater than thermal noise to be identified as leading edge
    LE_THRESHOLD_DP = 0.2  # define threshold on normalised amplitude change which is required to be accepted as lead edge

    # Initialise output variables
    LE_INDEX_END = np.nan
    LE_INDEX_START = np.nan
    LE_RANGE_START = np.nan
    LE_RANGE_END = np.nan
    WF_NORM_SMOOTH_INTERPOLATED = np.nan
    WF_NORM_SMOOTH_INTERPOLATED_D1 = np.nan
    WF_NOISE_MEAN = np.nan

    # Infinite loop to allow for conditional breaking and avoid bloated return statements - maybe a dumb idea?
    while True:

        # ----------------------------------------------------------------------
        # Normalise waveform
        # ----------------------------------------------------------------------

        wf_norm = waveform / max(waveform)  # normalise waveform

        # ----------------------------------------------------------------------
        # Smooth waveform
        # ----------------------------------------------------------------------

        # pseudo-Gaussian smoothing (3 passes of sliding-average smoothing)
        wf_norm_smooth = np.convolve(
            np.convolve(
                np.convolve(
                    wf_norm,
                    np.ones(SMOOTHING_WINDOW_WIDTH) / SMOOTHING_WINDOW_WIDTH,
                    mode="same",
                ),
                np.ones(SMOOTHING_WINDOW_WIDTH) / SMOOTHING_WINDOW_WIDTH,
                mode="same",
            ),
            np.ones(SMOOTHING_WINDOW_WIDTH) / SMOOTHING_WINDOW_WIDTH,
            mode="same",
        )
        np.insert(wf_norm_smooth, 0, np.nan)  # set end values as nan
        np.insert(wf_norm_smooth, len(wf_norm), np.nan)

        # ----------------------------------------------------------------------
        # Compute thermal noise
        # ----------------------------------------------------------------------

        wf_sorted = np.sort(
            wf_norm
        )  # sort power values of unsmoothed waveform in ascending order
        WF_NOISE_MEAN = np.mean(
            wf_sorted[0:6]
        )  # estimate noise based on lowest 6 samples

        # ----------------------------------------------------------------------
        # Quality check 1 - check if mean noise above predefined threshold
        # ----------------------------------------------------------------------

        if WF_NOISE_MEAN > NOISE_THRESHOLD:
            break

        # ----------------------------------------------------------------------
        # Oversample using spline
        # ----------------------------------------------------------------------

        WF_OVERSAMPLING_INTERVAL = (
            1 / WF_OVERSAMPLING_FACTOR
        )  # compute bin interval for oversampled waveform
        wf_bin_number_indices = np.arange(
            0, len(waveform), WF_OVERSAMPLING_INTERVAL
        )  # create oversampled waveform bin indices
        WF_NORM_SMOOTH_INTERPOLATED = interpolate.splev(
            wf_bin_number_indices,
            interpolate.splrep(range(len(waveform)), wf_norm_smooth),
        )  # compute spline and interpolated values of smoothed waveform at bin numbers

        # ----------------------------------------------------------------------
        # Compute derivatives
        # ----------------------------------------------------------------------

        WF_NORM_SMOOTH_INTERPOLATED_D1 = np.gradient(
            WF_NORM_SMOOTH_INTERPOLATED, WF_OVERSAMPLING_INTERVAL
        )  # compute first derivative of smoothed waveform

        # ----------------------------------------------------------------------
        # Loop through indices until no more peak candidates found or index a bin width away from end is reached
        # ----------------------------------------------------------------------

        LE_INDEX_PREVIOUS = 0  # previous leading edge index
        LE_DP = 0  # normalised amplitude change

        while LE_INDEX_PREVIOUS < len(wf_bin_number_indices) - WF_OVERSAMPLING_FACTOR:

            le_indices = np.where(
                (WF_NORM_SMOOTH_INTERPOLATED > (WF_NOISE_MEAN + LE_THRESHOLD_ID))
                & (
                    wf_bin_number_indices
                    > wf_bin_number_indices[LE_INDEX_PREVIOUS + WF_OVERSAMPLING_FACTOR]
                )
            )  # find next leading edge candidates (at least 1 original bin width from last) which are above the threshold

            # ----------------------------------------------------------------------
            # Quality check 2 - check if no samples are sufficiently above the noise or large enough leading edge
            # ----------------------------------------------------------------------

            if np.size(le_indices) == 0:
                break

            else:  # else take the first index found
                LE_INDEX = le_indices[0][0]

            # ----------------------------------------------------------------------
            # If leading edge exists find position
            # ----------------------------------------------------------------------

            # find stationary points on leading edge where gradient first becomes negative again
            peak_indices = np.where(
                (WF_NORM_SMOOTH_INTERPOLATED_D1 <= 0)
                & (wf_bin_number_indices > wf_bin_number_indices[LE_INDEX])
            )

            # ----------------------------------------------------------------------
            # Quality check 3 - check if a waveform peak can be identified after the start of the leading edge
            # ----------------------------------------------------------------------

            if np.size(peak_indices) == 0:
                break

            else:  # else take the first index found
                FIRST_PEAK_INDEX = peak_indices[0][0]

            # ----------------------------------------------------------------------
            # Calculate amplitude of peak above the noise floor threshold
            # ----------------------------------------------------------------------

            LE_DP = (
                WF_NORM_SMOOTH_INTERPOLATED[FIRST_PEAK_INDEX]
                - WF_NORM_SMOOTH_INTERPOLATED[LE_INDEX]
            )
            LE_INDEX_PREVIOUS = FIRST_PEAK_INDEX  # update previous leading edge to current one in case the amplitude change threshold is not met

            # ----------------------------------------------------------------------
            # Take peak if it exceeds minimum amplitude threshold
            # ----------------------------------------------------------------------

            if LE_DP > LE_THRESHOLD_DP:
                LE_INDEX_START = LE_INDEX / WF_OVERSAMPLING_FACTOR
                LE_INDEX_END = FIRST_PEAK_INDEX / WF_OVERSAMPLING_FACTOR
                break

        # ----------------------------------------------------------------------
        # Get the range from the sat to the leading edge
        # ----------------------------------------------------------------------

        if np.isnan(LE_INDEX_START) == False:
            LE_RANGE_START = (
                tracker_range - (REFERENCE_BIN_INDEX - LE_INDEX_START) * RANGE_BIN_SIZE
            )
            LE_RANGE_END = (
                tracker_range + (LE_INDEX_END - REFERENCE_BIN_INDEX) * RANGE_BIN_SIZE
            )
        break

    return (LE_INDEX_START, LE_INDEX_END, LE_RANGE_START, LE_RANGE_END)


def get_heading(nadir_x, nadir_y):

    """
    Get satellite track headings based on nadir coordinates.

    Args:
        nadir_x (numpy.ndarray): Array of x-coordinates of nadir points.
        nadir_y (numpy.ndarray): Array of y-coordinates of nadir points.

    Returns:
        numpy.ndarray: An array of heading angles in degrees corresponding to each record.

    Notes:
        - This function calculates the satellite track headings based on the nadir
          coordinates provided as input.
        - The heading angle is computed as the angle between the nadir vector and the
          north direction, with adjustments for quadrant orientation.
        - Requires at least 2 records for calculation.
    """

    # ----------------------------------------------------------------------
    # Define variables
    # ----------------------------------------------------------------------

    NUM_RECORDS = len(nadir_x)
    heading = np.full(NUM_RECORDS, np.nan)

    # ----------------------------------------------------------------------
    # Calculate heading for each record
    # ----------------------------------------------------------------------

    for record in range(NUM_RECORDS):

        # if it's the final record, copy the preceding dx and dy values
        if record == NUM_RECORDS - 1:
            dx = nadir_x[record] - nadir_x[record - 1]
            dy = nadir_y[record] - nadir_y[record - 1]

        else:  # else, compute delta x and delta y to next record
            dx = nadir_x[record + 1] - nadir_x[record]
            dy = nadir_y[record + 1] - nadir_y[record]

        # compute heading inner angle
        heading_inner = np.rad2deg(np.arctan(dx / dy))

        # if heading vector orientated into Q2 or Q3 then add 180 degrees
        if dy < 0:
            heading[record] = heading_inner + 180

        # if heading vector orientated into Q4 then add 360 degrees to make positive
        elif dy > 0 and dx < 0:
            heading[record] = heading_inner + 360

        else:
            heading[record] = heading_inner

    return heading


def plot(
    SATELLITE,
    record_range,
    DEM_PATH,
    L2_TRACK_PATH,
    L1_TRACK_PATH=None,
    DEM_PROJECTION="epsg:3031",
):

    """
    Main function to plot altimetry data alongside DEM data.

    Args:
        SATELLITE (str): The satellite type ("S3" or "CS2").
        record_range (tuple): A tuple containing the range of records to plot.
            It should be in the form (start_record, end_record).
        DEM_PATH (str): The path to the DEM (Digital Elevation Model) file.
        L2_TRACK_PATH (str): The path containing Level-2 track data.
        L1_TRACK_PATH (str, optional): The path containing Level-1 track data.
        DEM_PROJECTION (str, optional): The projection of the DEM data in EPSG format.
            Needs to be in units of meters. Defaults to "epsg:3031".
            Recommended that DEM resolution is less than size of the satellite footprint along-track.

    Returns:
        None: This function does not return a value. It generates and displays plots.

    Note:
        This function reads altimetry track data, processes it, and creates various plots
        to visualize altimetry data alongside a DEM. It handles different satellite types
        (S3 and CS2) and supports customizing the plot by specifying various parameters.
    """

    # ----------------------------------------------------------------------
    # Set up
    # ----------------------------------------------------------------------

    # read in track data
    data_L2 = Dataset(L2_TRACK_PATH)
    if L1_TRACK_PATH is not None:
        data_L1 = Dataset(L1_TRACK_PATH)
    else:
        if SATELLITE == "CS2":
            print("CS2 requires L1b data. Exiting...")
            return
        data_L1 = data_L2

    if len(data_L2["lat_20_ku"]) < record_range[1]:
        print("Record range outside of track range. Exiting...")
        return

    # define satellite-specific info
    if SATELLITE == "S3":

        nadir_lat = data_L2["lat_20_ku"][record_range[0] : record_range[1]]
        nadir_lon = data_L2["lon_20_ku"][record_range[0] : record_range[1]]
        poca_lat = data_L2["lat_cor_20_ku"][record_range[0] : record_range[1]]
        poca_lon = data_L2["lon_cor_20_ku"][record_range[0] : record_range[1]]
        waveform = data_L2["waveform_20_ku"][record_range[0] : record_range[1]]
        altitude = data_L2["alt_20_ku"][record_range[0] : record_range[1]]
        elevation = data_L2["elevation_ice_sheet_20_ku"][
            record_range[0] : record_range[1]
        ]

        elevation = elevation.filled(np.nan)
        elevation[elevation > 9999] = np.nan

        ACROSS_TRACK_WIDTH = 18200
        ALONG_TRACK_WIDTH = 300

        C = 299792458  # speed of light (m/s)
        B = 320000000  # chirp bandwidth used (Hz) from Donlon 2012 table 6
        RANGE_BIN_SIZE = C / (2 * B)  # compute distance between each bin in meters
        NUM_BINS = len(waveform[0])  # number of bins in waveform
        REFERENCE_BIN_INDEX = 43  # index on waveform that range is referenced to
        DIST_REFERENCE_BIN_INDEX_TO_START_WF = (
            REFERENCE_BIN_INDEX * RANGE_BIN_SIZE
        )  # distance in meters from reference index to start of waveform
        DIST_REFERENCE_BIN_INDEX_TO_END_WF = (
            NUM_BINS - REFERENCE_BIN_INDEX
        ) * RANGE_BIN_SIZE  # distance in meters from reference index to end of waveform

        LE_SMOOTHING_WINDOW_WIDTH = (
            3  # width of smoothing window for waveform when calculating leading edge
        )
        LE_WF_OVERSAMPLING_FACTOR = (
            10  # waveform oversampling factor for when calculating leading edge
        )

    elif SATELLITE == "CS2":

        nadir_lat = data_L2["lat_20_ku"][record_range[0] : record_range[1]]
        nadir_lon = data_L2["lon_20_ku"][record_range[0] : record_range[1]]
        poca_lat = data_L2["lat_poca_20_ku"][record_range[0] : record_range[1]]
        poca_lon = data_L2["lon_poca_20_ku"][record_range[0] : record_range[1]]
        waveform = data_L1["pwr_waveform_20_ku"][record_range[0] : record_range[1]]
        altitude = data_L2["alt_20_ku"][record_range[0] : record_range[1]]
        elevation = data_L2["height_1_20_ku"][record_range[0] : record_range[1]]

        ACROSS_TRACK_WIDTH = 15000
        ALONG_TRACK_WIDTH = 380

        C = 299792458  # speed of light (m/s)
        B = 320000000  # chirp bandwidth used (Hz) from Donlon 2012 table 6
        RANGE_BIN_SIZE = C / (4 * B)  # compute distance between each bin in meters
        NUM_BINS = len(waveform[0])  # number of bins in waveform
        REFERENCE_BIN_INDEX = 512  # index on waveform that range is referenced to
        DIST_REFERENCE_BIN_INDEX_TO_START_WF = (
            REFERENCE_BIN_INDEX * RANGE_BIN_SIZE
        )  # distance in meters from reference index to start of waveform
        DIST_REFERENCE_BIN_INDEX_TO_END_WF = (
            REFERENCE_BIN_INDEX * RANGE_BIN_SIZE
        )  # distance in meters from reference index to end of waveform

        LE_SMOOTHING_WINDOW_WIDTH = (
            11  # width of smoothing window for waveform when calculating leading edge
        )
        LE_WF_OVERSAMPLING_FACTOR = (
            1  # waveform oversampling factor for when calculating leading edge
        )

    else:
        print(SATELLITE + " not supported. Exiting...")
        return

    # get and correct tracker range
    tracker_range = get_and_correct_range(data_L1, record_range, SATELLITE)

    # transform to DEM crs
    transformer = proj.Transformer.from_crs("epsg:4326", DEM_PROJECTION)
    nadir_x, nadir_y = transformer.transform(nadir_lat, nadir_lon)
    poca_x, poca_y = transformer.transform(poca_lat, poca_lon)
    poca_x[np.isinf(poca_x)] = np.nan
    poca_y[np.isinf(poca_y)] = np.nan

    # get track heading
    heading = get_heading(nadir_x, nadir_y)

    NUM_RECORDS = record_range[1] - record_range[0]
    heading = np.full(NUM_RECORDS, heading)

    # get range from top to bottom of range window
    range_to_window_top = tracker_range - DIST_REFERENCE_BIN_INDEX_TO_START_WF
    range_to_window_bottom = tracker_range + DIST_REFERENCE_BIN_INDEX_TO_END_WF

    # get leading edge information
    NUM_RECORDS = record_range[1] - record_range[0]
    pool = mp.Pool()
    mp_output = pool.starmap(
        get_leading_edge,
        [
            (
                waveform[record],
                tracker_range[record],
                REFERENCE_BIN_INDEX,
                LE_SMOOTHING_WINDOW_WIDTH,
                RANGE_BIN_SIZE,
                LE_WF_OVERSAMPLING_FACTOR,
            )
            for record in range(NUM_RECORDS)
        ],
    )
    pool.close

    mp_output = np.asarray(mp_output, dtype="object")

    le_index_start = mp_output[:, 0]
    le_index_end = mp_output[:, 1]
    range_to_window_top_le = mp_output[:, 2]
    range_to_window_bottom_le = mp_output[:, 3]

    # define the footprint edges
    heading_perpendicular = heading - 270
    heading_perpendicular[heading_perpendicular < 0] = heading_perpendicular + 360
    dx_perpendicular = (ACROSS_TRACK_WIDTH / 2) * np.sin(
        np.deg2rad(heading_perpendicular)
    )
    dy_perpendicular = (ACROSS_TRACK_WIDTH / 2) * np.cos(
        np.deg2rad(heading_perpendicular)
    )
    fp_edge_1 = np.column_stack(
        [
            nadir_x + dx_perpendicular,
            nadir_y + dy_perpendicular,
            np.full(NUM_RECORDS, np.nan),
        ]
    )
    fp_edge_2 = np.column_stack(
        [
            nadir_x - dx_perpendicular,
            nadir_y - dy_perpendicular,
            np.full(NUM_RECORDS, np.nan),
        ]
    )

    # get bounds of DEM segment required from max and min of data
    X_MIN = np.min(
        [
            np.nanmin(poca_x),
            np.nanmin(nadir_x),
            np.nanmin(fp_edge_1[:, 0]),
            np.nanmin(fp_edge_2[:, 0]),
        ]
    )
    X_MAX = np.max(
        [
            np.nanmax(poca_x),
            np.nanmax(nadir_x),
            np.nanmax(fp_edge_1[:, 0]),
            np.nanmax(fp_edge_2[:, 0]),
        ]
    )
    Y_MIN = np.min(
        [
            np.nanmin(poca_y),
            np.nanmin(nadir_y),
            np.nanmin(fp_edge_1[:, 1]),
            np.nanmin(fp_edge_2[:, 1]),
        ]
    )
    Y_MAX = np.max(
        [
            np.nanmax(poca_y),
            np.nanmax(nadir_y),
            np.nanmax(fp_edge_1[:, 1]),
            np.nanmax(fp_edge_2[:, 1]),
        ]
    )

    # read in DEM and subset to track
    dem_x, dem_y, dem_z, DEM_RESOLUTION = get_dem_segment(
        DEM_PATH, [(X_MIN, X_MAX), (Y_MIN, Y_MAX)], GRID_XY=False, FLATTEN=False
    )
    dem_z[dem_z < -9998] = np.nan  # catch for when nans commonly presented as -9999

    # get max and min of z
    Z_MIN = np.min([np.nanmin(elevation), np.nanmin(dem_z)])
    Z_MAX = np.max([np.nanmax(elevation), np.nanmax(dem_z)])

    # check if area contains any DEM data
    if np.isnan(dem_z).all():
        print("No DEM data found within record range. Exiting...")
        return

    # get gridded and flattened versions of DEM data
    dem_x_gridded, dem_y_gridded = np.meshgrid(dem_x, dem_y)
    dem_x_gridded = dem_x_gridded.flatten()
    dem_y_gridded = dem_y_gridded.flatten()
    dem_z_gridded = dem_z.flatten()

    # get DEM in footprint for each record
    pool = mp.Pool()
    mp_output = pool.starmap(
        get_dem_in_fp,
        [
            (
                dem_x_gridded,
                dem_y_gridded,
                dem_z_gridded,
                ACROSS_TRACK_WIDTH,
                ALONG_TRACK_WIDTH,
                heading[record],
                nadir_x[record],
                nadir_y[record],
                altitude[record],
                range_to_window_top[record],
                range_to_window_bottom[record],
                range_to_window_top_le[record],
                range_to_window_bottom_le[record],
            )
            for record in range(NUM_RECORDS)
        ],
    )
    pool.close

    mp_output = np.asarray(mp_output, dtype="object")

    dem_in_fp = mp_output[:, 0]
    dem_in_fp_rw = mp_output[:, 1]
    dem_in_fp_le = mp_output[:, 2]

    # set up interpolation for generating z values for data with x and y
    interp = interpolate.NearestNDInterpolator(
        list(
            zip(
                dem_x_gridded[~np.isnan(dem_z_gridded)],
                dem_y_gridded[~np.isnan(dem_z_gridded)],
            )
        ),
        dem_z_gridded[~np.isnan(dem_z_gridded)],
    )

    # interpolate nadir z
    nadir_z = interp(nadir_x, nadir_y)

    # interpolate footprint edge z
    fp_edge_1[:, 2] = interp(fp_edge_1[:, 0], fp_edge_1[:, 1])
    fp_edge_2[:, 2] = interp(fp_edge_2[:, 0], fp_edge_2[:, 1])

    # set POCA z to elevation, interpolate where elevation is nan
    poca_z = elevation
    poca_z_interp = np.full(len(poca_x), np.nan)
    poca_nan = np.isnan(poca_x)
    poca_z_interp[~poca_nan] = interp(poca_x[~poca_nan], poca_y[~poca_nan])
    poca_z_interp[
        ~np.isnan(poca_z)
    ] = (
        np.nan
    )  # interpolated poca z only appears when non-interpolated poca z has no data

    # get range and leading edge windows in 3D for each record
    range_windows_3D = np.full((NUM_RECORDS, 2, 35, 3), np.nan)
    leading_edge_windows_3D = np.full((NUM_RECORDS, 2, 35, 3), np.nan)
    for record in range(NUM_RECORDS):

        # loop through x,y,z and leading edge x,y,z ranges (range to top, range to bottom)
        window_3D = []
        for window_range in [
            [range_to_window_top[record], range_to_window_bottom[record]],
            [range_to_window_top_le[record], range_to_window_bottom_le[record]],
        ]:

            # if current z ranges are all nan or of size 0, append nan
            if np.isnan(window_range).all() == True or np.size(window_range) == 0:
                window_3D.append(np.full((2, 35, 3), np.nan))
            else:
                # get x coords, taking footprint about x,y = 0
                footprint_trace_x = np.linspace(
                    -ACROSS_TRACK_WIDTH / 2, ACROSS_TRACK_WIDTH / 2, 17
                )

                # Get z coords of top and bottom of footprint
                footprint_trace_z1 = altitude[record] - np.sqrt(
                    window_range[0] ** 2 - np.square(footprint_trace_x)
                )
                footprint_trace_z2 = altitude[record] - np.sqrt(
                    window_range[1] ** 2 - np.square(footprint_trace_x)
                )

                # connect x and z traces so they create their own closed loops
                footprint_trace_x = np.append(
                    np.concatenate((footprint_trace_x, footprint_trace_x[::-1])),
                    -ACROSS_TRACK_WIDTH / 2,
                )
                footprint_trace_z = np.append(
                    np.concatenate((footprint_trace_z1, footprint_trace_z2[::-1])),
                    footprint_trace_z1[0],
                )

                # concatenate x and z coords together
                footprint_trace_xz = np.c_[footprint_trace_x, footprint_trace_z]

                # seperate footprint into two indentical footprints, along-track width width apart
                footprint_trace_xyz_1 = np.insert(
                    footprint_trace_xz, 1, -ALONG_TRACK_WIDTH / 2, axis=1
                )
                footprint_trace_xyz_2 = np.insert(
                    footprint_trace_xz, 1, ALONG_TRACK_WIDTH / 2, axis=1
                )

                # calculate rotation matrix for heading
                rotMat = Rotation.from_euler("z", -heading[record], degrees=True)

                # rotate footprints to correct rotation
                footprint_trace_xyz_1 = rotMat.apply(footprint_trace_xyz_1)
                footprint_trace_xyz_2 = rotMat.apply(footprint_trace_xyz_2)

                # translate footprints to correct location and append
                footprint_trace_xyz_1 = footprint_trace_xyz_1 + np.array(
                    [nadir_x[record], nadir_y[record], 0], dtype="object"
                )
                footprint_trace_xyz_2 = footprint_trace_xyz_2 + np.array(
                    [nadir_x[record], nadir_y[record], 0], dtype="object"
                )
                window_3D.append(
                    np.array([footprint_trace_xyz_1, footprint_trace_xyz_2])
                )

        range_windows_3D[record] = window_3D[0]
        leading_edge_windows_3D[record] = window_3D[1]

    # define plot-specific parameters
    surface_colourscale = [
        [0.0, "rgb(255,255,255)"],
        [1.0, "rgb(255,255,255)"],
    ]  # surface colourscale - set to white to white for ice
    lighting_effects = dict(
        ambient=0.7, diffuse=0.5, roughness=0.9, specular=0.1, fresnel=0.5
    )  # light effects for surface plot - specified for ice
    LIGHT_X_DIRECTION = 100
    LIGHT_Y_DIRECTION = 10
    LIGHT_Z_DIRECTION = 100000

    # get aspect ratios for surface view
    aspect_ratio = ((X_MAX - X_MIN) / (Y_MAX - Y_MIN), 1, 0.1)

    # set center of camera to nadir location for each record
    nadir_x_cam = (((nadir_x - X_MIN) / (X_MAX - X_MIN)) - 0.5) * aspect_ratio[0]
    nadir_y_cam = (((nadir_y - Y_MIN) / (Y_MAX - Y_MIN)) - 0.5) * aspect_ratio[1]
    nadir_z_cam = (((nadir_z - Z_MIN) / (Z_MAX - Z_MIN)) - 0.5) * aspect_ratio[2]

    cam_centre = np.column_stack([nadir_x_cam, nadir_y_cam, nadir_z_cam])

    # set camera position backwards along track
    Z_VIEW_ANGLE = 70
    CAM_NUM_RECORDS_BACK = (
        20  # number of records back the surface plot camera is displaced
    )
    backward_vector = np.column_stack(
        [
            np.sin(np.deg2rad(Z_VIEW_ANGLE)) * np.cos(np.deg2rad(270 - heading)),
            np.sin(np.deg2rad(Z_VIEW_ANGLE)) * np.sin(np.deg2rad(270 - heading)),
            np.full(NUM_RECORDS, np.cos(np.deg2rad(Z_VIEW_ANGLE))),
        ]
    )  # spherical -> cartesian
    cam_dist_between_records = np.linalg.norm(
        [nadir_x_cam[1] - nadir_x_cam[0], nadir_y_cam[1] - nadir_y_cam[0]]
    )
    CAM_EYE_DIST = cam_dist_between_records * CAM_NUM_RECORDS_BACK
    cam_eye = cam_centre + (backward_vector * CAM_EYE_DIST)

    # get data ratios to scale range window view properly
    fp_range_x = np.max(range_windows_3D[NUM_RECORDS // 2, 0, :, 0]) - np.min(
        range_windows_3D[NUM_RECORDS // 2, 0, :, 0]
    )
    fp_range_y = np.max(range_windows_3D[NUM_RECORDS // 2, 0, :, 1]) - np.min(
        range_windows_3D[NUM_RECORDS // 2, 0, :, 1]
    )
    aspect_ratio_2 = (
        np.ceil((fp_range_x / (fp_range_x + fp_range_y)) * 10),
        np.ceil((fp_range_y / (fp_range_x + fp_range_y)) * 10),
        3,
    )

    # default camera look is at center of window in range window view, change eye vector coords so camera faces along track
    cam_eye_dist_2 = 5
    cam_eye_2 = (
        np.column_stack(
            [
                np.cos(np.deg2rad(270 - heading)),
                np.sin(np.deg2rad(270 - heading)),
                np.full(NUM_RECORDS, 0),
            ]
        )
        * cam_eye_dist_2
    )

    # define plot colours
    nadir_colour = "#DE3C4B"
    poca_colour = "#00cee1"
    poca_interp_colour = "#307fa6"
    footprint_colour = "#39ac39"
    dem_colour = "#8c40b8"
    dem_in_rw_colour = "#b555b2"
    dem_in_le_colour = "#d4aed2"
    waveform_colour = "#8f510a"
    le_colour = "#DEB887"

    # ----------------------------------------------------------------------
    # Plot
    # ----------------------------------------------------------------------

    # Initialize figure with subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=["Surface View", "Waveform", "Range Window View"],
        column_widths=[0.5, 0.5],
        specs=[
            [{"type": "scene", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "scene"}],
        ],
        vertical_spacing=0.125,
        horizontal_spacing=0.05,
    )

    # row 1, col 1-2
    fig.add_trace(
        go.Surface(
            name=str(int(DEM_RESOLUTION)) + "m Resolution DEM<br>(Scaled)",
            x=dem_x,
            y=dem_y,
            z=dem_z,
            colorscale=surface_colourscale,
            opacity=0.8,
            showscale=False,
            showlegend=True,
            lighting=lighting_effects,
            lightposition=dict(
                x=LIGHT_X_DIRECTION, y=LIGHT_Y_DIRECTION, z=LIGHT_Z_DIRECTION
            ),
            contours=go.surface.Contours(
                x=go.surface.contours.X(highlight=False),
                y=go.surface.contours.Y(highlight=False),
                z=go.surface.contours.Z(highlight=False),
            ),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=nadir_x,
            y=nadir_y,
            z=nadir_z,
            mode="lines+markers",
            name="Nadir",
            marker=dict(color=nadir_colour, size=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=poca_x,
            y=poca_y,
            z=poca_z,
            mode="lines+markers",
            name="POCA (L2 Elevation)",
            marker=dict(color=poca_colour, size=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=poca_x,
            y=poca_y,
            z=poca_z_interp,
            mode="lines+markers",
            name="POCA (Interpolated Elevation)",
            marker=dict(color=poca_interp_colour, size=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=fp_edge_1[:, 0],
            y=fp_edge_1[:, 1],
            z=fp_edge_1[:, 2],
            mode="lines",
            name="Footprint Edge",
            marker=dict(color=footprint_colour),
            opacity=0.7,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=fp_edge_2[:, 0],
            y=fp_edge_2[:, 1],
            z=fp_edge_2[:, 2],
            mode="lines",
            name="Footprint Edge",
            showlegend=False,
            marker=dict(color=footprint_colour),
            opacity=0.7,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=dem_in_fp[0][0],
            y=dem_in_fp[0][1],
            z=np.array(dem_in_fp[0][2]),
            marker=dict(color=dem_colour, size=3),
            mode="markers",
            name="DEM in Footprint",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X", range=[X_MIN, X_MAX], showgrid=False),
            yaxis=dict(title="Y", range=[Y_MIN, Y_MAX], showgrid=False),
            zaxis=dict(title="Elevation", showgrid=False),
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="manual",
            aspectratio=dict(x=aspect_ratio[0], y=aspect_ratio[1], z=aspect_ratio[2]),
            bgcolor="#87CEEB",
            camera=dict(
                eye=dict(x=cam_eye[0, 0], y=cam_eye[0, 1], z=cam_eye[0, 2]),
                center=dict(x=cam_centre[0, 0], y=cam_centre[0, 1], z=cam_centre[0, 2]),
            ),
        )
    )

    # row 2 col 1
    gate_number = np.array(range(len(waveform[0])))
    fig.add_trace(
        go.Scatter(
            x=gate_number,
            y=waveform[0],
            name="Waveform",
            mode="lines",
            marker=dict(color=waveform_colour),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[le_index_start[0], le_index_start[0]],
            y=[min(waveform[0]), max(waveform[0])],
            name="Leading Edge",
            mode="lines",
            line=dict(dash="dash"),
            marker=dict(color=le_colour),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[le_index_end[0], le_index_end[0]],
            y=[min(waveform[0]), max(waveform[0])],
            name="Leading Edge",
            mode="lines",
            showlegend=False,
            line=dict(dash="dash"),
            marker=dict(color=le_colour),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        xaxis=dict(title="Gate Number"),
        yaxis=dict(
            title="Power", autorange=False, range=[0, 1.1 * np.max(waveform[0])]
        ),
    )

    # row 2, col 2
    fig.add_trace(
        go.Scatter3d(
            x=dem_in_fp[0][0],
            y=dem_in_fp[0][1],
            z=dem_in_fp[0][2],
            marker=dict(color=dem_colour, size=2),
            mode="markers",
            name="DEM in Footprint",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=dem_in_fp_rw[0][0],
            y=dem_in_fp_rw[0][1],
            z=dem_in_fp_rw[0][2],
            name="DEM in Range Window",
            mode="markers",
            marker=dict(color=dem_in_rw_colour, size=2),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=dem_in_fp_le[0][0],
            y=dem_in_fp_le[0][1],
            z=dem_in_fp_le[0][2],
            name="DEM in Leading Edge",
            mode="markers",
            marker=dict(color=dem_in_le_colour, size=2),
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter3d(
            x=range_windows_3D[0][0][:, 0],
            y=range_windows_3D[0][0][:, 1],
            z=range_windows_3D[0][0][:, 2],
            marker=dict(color=footprint_colour),
            mode="lines",
            name="Range Window",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=range_windows_3D[0][1][:, 0],
            y=range_windows_3D[0][1][:, 1],
            z=range_windows_3D[0][1][:, 2],
            marker=dict(color=footprint_colour),
            mode="lines",
            showlegend=False,
            name="Range Window",
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter3d(
            x=leading_edge_windows_3D[0][0][:, 0],
            y=leading_edge_windows_3D[0][0][:, 1],
            z=leading_edge_windows_3D[0][0][:, 2],
            marker=dict(color=footprint_colour),
            mode="lines",
            showlegend=False,
            name="Leading Edge",
            line=dict(dash="dash"),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=leading_edge_windows_3D[0][1][:, 0],
            y=leading_edge_windows_3D[0][1][:, 1],
            z=leading_edge_windows_3D[0][1][:, 2],
            marker=dict(color=footprint_colour),
            mode="lines",
            showlegend=False,
            name="Leading Edge",
            line=dict(dash="dash"),
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter3d(
            x=[poca_x[0]],
            y=[poca_y[0]],
            z=[poca_z[0]],
            mode="markers",
            marker=dict(color=poca_colour, size=4, symbol="x"),
            showlegend=False,
            name="POCA (L2 Elevation)",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter3d(
            x=[poca_x[0]],
            y=[poca_y[0]],
            z=[poca_z_interp[0]],
            mode="markers",
            marker=dict(color=poca_interp_colour, size=4, symbol="x"),
            showlegend=False,
            name="POCA (Interpolated Elevation)",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        scene2=dict(
            xaxis=dict(title="X", showgrid=False),
            yaxis=dict(title="Y", showgrid=False),
            zaxis=dict(
                title="Elevation (Scaled)", tickformat=".0f", dtick=40, showgrid=False
            ),
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="manual",
            aspectratio=dict(
                x=aspect_ratio_2[0],
                y=aspect_ratio_2[1],
                z=aspect_ratio_2[2],
            ),
            camera=dict(
                eye=dict(x=cam_eye_2[0, 0], y=cam_eye_2[0, 1], z=cam_eye_2[0, 2])
            ),
        )
    )

    # make frames
    frames = [
        dict(
            name=k + record_range[0],
            data=[  # row 1 col 1-2
                go.Scatter3d(x=dem_in_fp[k][0], y=dem_in_fp[k][1], z=dem_in_fp[k][2]),
                # row 2 col 1
                go.Scatter(x=gate_number, y=waveform[k]),
                go.Scatter(
                    x=[le_index_start[k], le_index_start[k]],
                    y=[min(waveform[k]), max(waveform[k])],
                ),
                go.Scatter(
                    x=[le_index_end[k], le_index_end[k]],
                    y=[min(waveform[k]), max(waveform[k])],
                ),
                # row 2 col 2
                go.Scatter3d(x=dem_in_fp[k][0], y=dem_in_fp[k][1], z=dem_in_fp[k][2]),
                go.Scatter3d(
                    x=dem_in_fp_rw[k][0], y=dem_in_fp_rw[k][1], z=dem_in_fp_rw[k][2]
                ),
                go.Scatter3d(
                    x=dem_in_fp_le[k][0], y=dem_in_fp_le[k][1], z=dem_in_fp_le[k][2]
                ),
                go.Scatter3d(
                    x=range_windows_3D[k][0][:, 0],
                    y=range_windows_3D[k][0][:, 1],
                    z=range_windows_3D[k][0][:, 2],
                ),
                go.Scatter3d(
                    x=range_windows_3D[k][1][:, 0],
                    y=range_windows_3D[k][1][:, 1],
                    z=range_windows_3D[k][1][:, 2],
                ),
                go.Scatter3d(
                    x=leading_edge_windows_3D[k][0][:, 0],
                    y=leading_edge_windows_3D[k][0][:, 1],
                    z=leading_edge_windows_3D[k][0][:, 2],
                ),
                go.Scatter3d(
                    x=leading_edge_windows_3D[k][1][:, 0],
                    y=leading_edge_windows_3D[k][1][:, 1],
                    z=leading_edge_windows_3D[k][1][:, 2],
                ),
                go.Scatter3d(x=[poca_x[k]], y=[poca_y[k]], z=[poca_z[k]]),
                go.Scatter3d(x=[poca_x[k]], y=[poca_y[k]], z=[poca_z_interp[k]]),
            ],
            layout=dict(
                scene=dict(
                    camera=dict(
                        eye=dict(x=cam_eye[k, 0], y=cam_eye[k, 1], z=cam_eye[k, 2]),
                        center=dict(
                            x=cam_centre[k, 0], y=cam_centre[k, 1], z=cam_centre[k, 2]
                        ),
                    )
                ),
                scene2=dict(
                    camera=dict(
                        eye=dict(
                            x=cam_eye_2[k, 0], y=cam_eye_2[k, 1], z=cam_eye_2[k, 2]
                        )
                    )
                ),
                yaxis=dict(range=[0, 1.1 * np.max(waveform[k])]),
            ),
            traces=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        )
        for k in range(NUM_RECORDS)
    ]

    # make sliders and buttons
    updatemenus = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 500, "redraw": False},
                            "fromcurrent": True,
                            "transition": {"duration": 1},
                            "mode": "next",
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 85},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }
    ]

    sliders = [
        {
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Record: ",
                "xanchor": "right",
            },
            "transition": {"duration": 0},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [
                        [k],
                        {
                            "frame": {"duration": 500, "redraw": False},
                            "fromcurrent": True,
                            "transition": {"duration": 1},
                            "mode": "next",
                        },
                    ],
                    "label": k,
                    "method": "animate",
                }
                for k in list(range(record_range[0], record_range[1]))
            ],
        }
    ]

    # Finalise
    fig.frames = frames
    fig.update_layout(
        updatemenus=updatemenus,
        sliders=sliders,
        title=SATELLITE
        + " SAR Altimetry Plot for Records "
        + str(record_range[0])
        + "-"
        + str(record_range[1])
        + " in<br>"
        + L2_TRACK_PATH,
        template="plotly_dark",
        title_x=0.5,
        title_y=0.965,
        legend_itemclick=False,
        legend_itemdoubleclick=False,
        legend=dict(font=dict(size=10)),
        # autosize=False,
        # width=1366,
        # height=784,
    )

    fig.write_html(
        SATELLITE
        + "_SAR_Altimetry_Plot_Records_"
        + str(record_range[0])
        + "-"
        + str(record_range[-1])
        + ".html",
        auto_play=False,
        # config=dict(responsive=False),
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":

    # Read in command-line variables
    parser = argparse.ArgumentParser(
        description="Create an interactive plot visualising SAR satellite altimetry alongside DEM data."
    )

    parser.add_argument("SAT", choices=["S3", "CS2"], help="The satellite (S3 or CS2).")
    parser.add_argument(
        "RANGE",
        type=lambda s: tuple(map(int, s.split(","))),
        help="A tuple containing the range of records to plot (start_record, end_record).",
    )
    parser.add_argument("DEM_PATH", help="The path to the DEM file. Ideally, the DEM should have resolution below the size of the satellite footprint along-track.")
    parser.add_argument("L2_PATH", help="The path to the Level-2 track data. Required for Cryosat-2.")
    parser.add_argument(
        "--L1_PATH",
        default=None,
        help="The path to the Level-1 track data.",
    )
    parser.add_argument(
        "--DEM_PROJ",
        default="epsg:3031",
        help="The projection of the DEM data in EPSG format (default is EPSG:3031). This should be in meters.",
    )

    args = parser.parse_args()

    # Call the plot function with the provided command-line arguments
    plot(
        args.SAT,
        args.RANGE,
        args.DEM_PATH,
        args.L2_PATH,
        args.L1_PATH,
        args.DEM_PROJ,
    )