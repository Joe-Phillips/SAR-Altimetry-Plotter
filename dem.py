import os
import sys
import numpy as np
import matplotlib as mpl
from tifffile import TiffFile, imread


def validate_dem(dem_path):
    """Basic sanity checks on a DEM GeoTIFF. Prints and exits on failure."""
    if not os.path.isfile(dem_path):
        print(f"Error: DEM file not found: {dem_path}")
        sys.exit(1)
    if not dem_path.lower().endswith((".tif", ".tiff")):
        print(f"Error: DEM does not appear to be a GeoTIFF: {dem_path}")
        sys.exit(1)
    try:
        tags = TiffFile(dem_path).pages[0].tags
    except Exception as e:
        print(f"Error: Could not read DEM as a TIFF: {e}")
        sys.exit(1)
    for tag in ("ImageWidth", "ImageLength", "ModelPixelScaleTag", "ModelTiepointTag"):
        if tag not in tags:
            print(f"Error: DEM is missing expected GeoTIFF tag: {tag}")
            sys.exit(1)
    if tags["ModelPixelScaleTag"].value[0] <= 0:
        print("Error: DEM has an invalid pixel scale (must be > 0).")
        sys.exit(1)


def get_dem_segment(raster_dir, segment_bounds, grid_xy=True, flatten=False, band_name=None):
    """
    Read in and subset raster data to given bounds.

    Args:
        raster_dir (str): The path to the raster data file.
        segment_bounds (list): [(minx, maxx), (miny, maxy)]
        grid_xy (bool): If True, x and y are returned as grids. Defaults to True.
        flatten (bool): If True, raster data is flattened. Defaults to False.
        band_name (str, optional): Name/description of the band to use. If None, the first
            band is used. If the file has multiple bands and this is None, a warning is
            printed showing the name of the band that was selected.

    Returns:
        tuple: (x, y, raster_data, resolution)
    """
    tif = TiffFile(raster_dir)
    raster_tags = tif.pages[0].tags

    num_x = raster_tags["ImageWidth"].value
    num_y = raster_tags["ImageLength"].value
    resolution = raster_tags["ModelPixelScaleTag"].value[0]
    num_bands = len(tif.pages)
    x_min = raster_tags["ModelTiepointTag"].value[3]
    y_max = raster_tags["ModelTiepointTag"].value[4]
    x_max = x_min + num_x * resolution
    y_min = y_max - num_y * resolution

    # Resolve which band index to use
    band_names = [tif.pages[i].description or f"band_{i}" for i in range(num_bands)]

    if band_name is None:
        band_index = 0
        if num_bands > 1:
            print(f"Warning: DEM has {num_bands} bands. Using first band: '{band_names[0]}'.")
    else:
        if band_name not in band_names:
            print(f"Error: Band '{band_name}' not found in DEM. Available bands: {band_names}")
            sys.exit(1)
        band_index = band_names.index(band_name)

    x = np.linspace(x_min, x_max, num_x, endpoint=False)
    y = np.linspace(y_min, y_max, num_y, endpoint=False)
    y = np.flip(y)

    x_min_ind = (np.absolute(segment_bounds[0][0] - x - resolution)).argmin()
    x_max_ind = (np.absolute(segment_bounds[0][1] - x + resolution)).argmin()
    y_min_ind = (np.absolute(segment_bounds[1][0] - y - resolution)).argmin()
    y_max_ind = (np.absolute(segment_bounds[1][1] - y + resolution)).argmin()

    x = x[x_min_ind:x_max_ind]
    y = y[y_max_ind:y_min_ind]

    raster = np.full((len(y), len(x), 1), np.nan, dtype=np.float32)
    raster_band = imread(raster_dir, key=band_index)
    raster[:, :, 0] = raster_band[y_max_ind:y_min_ind, x_min_ind:x_max_ind]

    if grid_xy:
        x, y = np.meshgrid(x, y)

    if not flatten:
        return x, y, raster.squeeze(), resolution

    raster = np.array(
        [raster[:, :, i].flatten() for i in range(np.shape(raster)[-1])]
    )
    return x.flatten(), y.flatten(), raster.squeeze(), resolution


def get_dem_in_fp(
    dem_x,
    dem_y,
    dem_z,
    across_track_width,
    along_track_width,
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
    Extract DEM data within specified footprints.

    Args:
        dem_x, dem_y, dem_z (numpy.ndarray): DEM coordinate and elevation arrays.
        across_track_width (float): Footprint width across track.
        along_track_width (float): Footprint width along track.
        heading (float): Heading angle in degrees.
        nadir_x, nadir_y (float): Nadir point coordinates.
        altitude (float): Satellite altitude.
        range_to_window_top (float): Range to top of window.
        range_to_window_bottom (float): Range to bottom of window.
        range_to_window_top_le (float, optional): Range to top of leading edge window.
        range_to_window_bottom_le (float, optional): Range to bottom of leading edge window.

    Returns:
        list: [dem_in_fp_xy, dem_in_fp_xyz, dem_in_fp_xyz_le]
            Each is a tuple of (x, y, z) arrays, or NaN arrays if no points found.
    """
    nan_result = (np.full(1, np.nan), np.full(1, np.nan), np.full(1, np.nan))
    dem_in_fp_xy = nan_result
    dem_in_fp_xyz = nan_result
    dem_in_fp_xyz_le = nan_result

    # Build and orient the xy footprint
    fp_xy = mpl.path.Path([
        (-along_track_width / 2, -across_track_width / 2),
        (-along_track_width / 2,  across_track_width / 2),
        ( along_track_width / 2,  across_track_width / 2),
        ( along_track_width / 2, -across_track_width / 2),
    ])
    fp_xy = fp_xy.transformed(mpl.transforms.Affine2D().rotate_deg(90 - heading))
    fp_xy = fp_xy.transformed(mpl.transforms.Affine2D().translate(nadir_x, nadir_y))

    # Subset DEM to bounding box of footprint for speed
    verts = fp_xy.vertices
    in_bounds = np.where(
        (dem_x >= verts[:, 0].min()) & (dem_x <= verts[:, 0].max()) &
        (dem_y >= verts[:, 1].min()) & (dem_y <= verts[:, 1].max())
    )
    dem_x = dem_x[in_bounds]
    dem_y = dem_y[in_bounds]
    dem_z = dem_z[in_bounds]

    # Find DEM points within footprint
    in_fp = fp_xy.contains_points(np.column_stack((dem_x, dem_y)))
    inds_in_fp = np.nonzero(in_fp)

    dem_x_in_fp = dem_x[inds_in_fp]
    dem_y_in_fp = dem_y[inds_in_fp]
    dem_z_in_fp = dem_z[inds_in_fp]

    if len(dem_z_in_fp) == 0:
        return [dem_in_fp_xy, dem_in_fp_xyz, dem_in_fp_xyz_le]

    dem_in_fp_xy = [dem_x_in_fp, dem_y_in_fp, dem_z_in_fp]

    # Compute distances from each DEM point to the satellite
    dem_pts = np.column_stack([dem_x_in_fp, dem_y_in_fp, dem_z_in_fp])
    sat_pos = np.column_stack([nadir_x, nadir_y, altitude])
    dists = np.linalg.norm(dem_pts - sat_pos, axis=-1)

    # Range window subset
    inds_rw = np.where((dists >= range_to_window_top) & (dists <= range_to_window_bottom))
    if np.size(inds_rw) > 0:
        dem_in_fp_xyz = [dem_x_in_fp[inds_rw], dem_y_in_fp[inds_rw], dem_z_in_fp[inds_rw]]

    # Leading edge subset
    if np.isnan(range_to_window_top_le).all():
        inds_le = []
    else:
        inds_le = np.where(
            (dists >= range_to_window_top_le) & (dists <= range_to_window_bottom_le)
        )

    if np.size(inds_le) > 0:
        dem_in_fp_xyz_le = [dem_x_in_fp[inds_le], dem_y_in_fp[inds_le], dem_z_in_fp[inds_le]]

    return [dem_in_fp_xy, dem_in_fp_xyz, dem_in_fp_xyz_le]