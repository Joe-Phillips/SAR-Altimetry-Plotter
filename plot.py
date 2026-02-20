import os
import sys
import tempfile
import numpy as np
import pyproj as proj
import multiprocessing as mp
from tqdm import tqdm
from netCDF4 import Dataset
from scipy import interpolate
from scipy.spatial.transform import Rotation
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from dem import get_dem_segment, get_dem_in_fp, validate_dem
from altimetry import get_and_correct_range, get_leading_edge, get_heading, validate_nc, validate_epsg


def build_dem_colourscale(colours, bins, z_min, z_max):
    """
    Convert a list of colours and bin-edge values into a normalised Plotly colourscale
    with sharp (step) transitions.

    Args:
        colours (list): N colour names or hex strings.
        bins (list): N-1 bin-edge values in metres.
        z_min (float): Minimum z value in the DEM.
        z_max (float): Maximum z value in the DEM.

    Returns:
        list: Plotly colourscale as [[normalised_value, colour], ...] pairs.
    """
    if len(colours) != len(bins) + 1:
        print(f"Error: Expected {len(bins) + 1} colours for {len(bins)} bin edges, got {len(colours)}.")
        sys.exit(1)

    z_range = z_max - z_min or 1.0

    def normalise(v):
        return float(np.clip((v - z_min) / z_range, 0.0, 1.0))

    scale = [[0.0, colours[0]]]
    for i, edge in enumerate(bins):
        n = normalise(edge)
        scale.append([n, colours[i]])
        scale.append([n, colours[i + 1]])
    scale.append([1.0, colours[-1]])

    return scale


def plot(
    satellite,
    record_range,
    dem_path,
    l2_track_path,
    l1_track_path=None,
    dem_projection="epsg:3031",
    dem_band_name=None,
    dem_colours=None,
    dem_colour_bins=None,
    generate_html=True,
    generate_video=True,
    video_fps=10,
    video_resolution=(1280, 720),
    output_name=None,
):
    """
    Plot altimetry data alongside DEM data.

    Args:
        satellite (str): "S3" or "CS2".
        record_range (tuple): (start_record, end_record).
        dem_path (str): Path to the DEM GeoTIFF file.
        l2_track_path (str): Path to the Level-2 NetCDF track data.
        l1_track_path (str, optional): Path to the Level-1 NetCDF track data.
            Required for CS2.
        dem_projection (str, optional): DEM projection in EPSG format (must be in metres).
            Defaults to "epsg:3031".
        dem_band_name (str, optional): Name of the DEM band to use. If None, the first
            band is used and a warning is printed if multiple bands exist.
        dem_colours (list, optional): N colours (named or hex) for DEM shading.
            If None, the surface is rendered flat white.
        dem_colour_bins (list, optional): N-1 bin edges in metres that separate
            dem_colours. Must have exactly one fewer entry than dem_colours.
        generate_html (bool, optional): Whether to write an interactive HTML file.
            Defaults to True.
        generate_video (bool, optional): Whether to render an MP4 video by exporting
            each animation frame as a static image and combining them.
            Requires kaleido==0.2.1 and imageio[ffmpeg]. Defaults to True.
        video_fps (int, optional): Frames (records) per second in the output video.
            Defaults to 10.
        video_resolution (tuple, optional): Output video resolution as (width, height)
            in pixels. Defaults to (1280, 720).
        output_name (str, optional): Base name for output files, without extension.
            Defaults to None, which auto-generates a name from the satellite and
            record range.
    """
    # -------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------

    validate_dem(dem_path)
    validate_nc(l2_track_path, satellite, "l2")
    if l1_track_path is not None:
        validate_nc(l1_track_path, satellite, "l1")
    validate_epsg(dem_projection)

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------

    data_l2 = Dataset(l2_track_path)
    data_l1 = Dataset(l1_track_path) if l1_track_path is not None else data_l2

    if satellite == "CS2" and l1_track_path is None:
        print("Error: CS2 requires L1b data. Please provide --l1_path.")
        sys.exit(1)

    if len(data_l2["lat_20_ku"]) < record_range[1]:
        print(f"Error: Record range {record_range} exceeds track length {len(data_l2['lat_20_ku'])}.")
        sys.exit(1)

    # -------------------------------------------------------------------
    # Satellite-specific parameters
    # -------------------------------------------------------------------

    c = 299792458  # speed of light (m/s)
    b = 320000000  # chirp bandwidth (Hz), Donlon 2012 table 6

    if satellite == "S3":
        nadir_lat  = data_l2["lat_20_ku"][record_range[0]:record_range[1]]
        nadir_lon  = data_l2["lon_20_ku"][record_range[0]:record_range[1]]
        poca_lat   = data_l2["lat_cor_20_ku"][record_range[0]:record_range[1]]
        poca_lon   = data_l2["lon_cor_20_ku"][record_range[0]:record_range[1]]
        waveform   = data_l2["waveform_20_ku"][record_range[0]:record_range[1]]
        altitude   = data_l2["alt_20_ku"][record_range[0]:record_range[1]]
        elevation  = data_l2["elevation_ice_sheet_20_ku"][record_range[0]:record_range[1]]

        elevation = elevation.filled(np.nan)
        elevation[elevation >= 9999] = np.nan

        across_track_width     = 18200
        along_track_width      = 300
        range_bin_size         = c / (2 * b)
        num_bins               = len(waveform[0])
        reference_bin_index    = 43
        le_smoothing_width     = 3
        le_oversampling_factor = 10

    elif satellite == "CS2":
        nadir_lat  = data_l2["lat_20_ku"][record_range[0]:record_range[1]]
        nadir_lon  = data_l2["lon_20_ku"][record_range[0]:record_range[1]]
        poca_lat   = data_l2["lat_poca_20_ku"][record_range[0]:record_range[1]]
        poca_lon   = data_l2["lon_poca_20_ku"][record_range[0]:record_range[1]]
        waveform   = data_l1["pwr_waveform_20_ku"][record_range[0]:record_range[1]]
        altitude   = data_l2["alt_20_ku"][record_range[0]:record_range[1]]
        elevation  = data_l2["height_1_20_ku"][record_range[0]:record_range[1]]

        across_track_width     = 15000
        along_track_width      = 380
        range_bin_size         = c / (4 * b)
        num_bins               = len(waveform[0])
        reference_bin_index    = 512
        le_smoothing_width     = 11
        le_oversampling_factor = 1

    else:
        print(f"Error: Satellite '{satellite}' is not supported. Choose 'S3' or 'CS2'.")
        sys.exit(1)

    dist_ref_to_start = reference_bin_index * range_bin_size
    dist_ref_to_end   = (num_bins - reference_bin_index) * range_bin_size if satellite == "S3" \
                        else reference_bin_index * range_bin_size

    # -------------------------------------------------------------------
    # Range corrections, projection, heading
    # -------------------------------------------------------------------

    tracker_range = get_and_correct_range(data_l1, record_range, satellite)

    transformer = proj.Transformer.from_crs("epsg:4326", dem_projection)
    nadir_x, nadir_y = transformer.transform(nadir_lat, nadir_lon)
    poca_x, poca_y   = transformer.transform(poca_lat, poca_lon)
    poca_x[np.isinf(poca_x)] = np.nan
    poca_y[np.isinf(poca_y)] = np.nan

    heading = get_heading(nadir_x, nadir_y)

    range_to_window_top    = tracker_range - dist_ref_to_start
    range_to_window_bottom = tracker_range + dist_ref_to_end

    # -------------------------------------------------------------------
    # Leading edge detection
    # -------------------------------------------------------------------

    num_records = record_range[1] - record_range[0]

    with mp.Pool() as pool:
        le_results = pool.starmap(
            get_leading_edge,
            [
                (waveform[k], tracker_range[k], reference_bin_index,
                 le_smoothing_width, range_bin_size, le_oversampling_factor)
                for k in range(num_records)
            ],
        )

    le_results                = np.asarray(le_results, dtype="object")
    le_index_start            = le_results[:, 0]
    le_index_end              = le_results[:, 1]
    range_to_window_top_le    = le_results[:, 2]
    range_to_window_bottom_le = le_results[:, 3]

    # -------------------------------------------------------------------
    # Footprint edges
    # -------------------------------------------------------------------

    heading_perp = heading - 270
    heading_perp[heading_perp < 0] += 360

    dx_perp   = (across_track_width / 2) * np.sin(np.deg2rad(heading_perp))
    dy_perp   = (across_track_width / 2) * np.cos(np.deg2rad(heading_perp))
    fp_edge_1 = np.column_stack([nadir_x + dx_perp, nadir_y + dy_perp, np.full(num_records, np.nan)])
    fp_edge_2 = np.column_stack([nadir_x - dx_perp, nadir_y - dy_perp, np.full(num_records, np.nan)])

    # -------------------------------------------------------------------
    # Load DEM
    # -------------------------------------------------------------------

    x_min = np.nanmin([np.nanmin(poca_x), np.nanmin(nadir_x), np.nanmin(fp_edge_1[:, 0]), np.nanmin(fp_edge_2[:, 0])])
    x_max = np.nanmax([np.nanmax(poca_x), np.nanmax(nadir_x), np.nanmax(fp_edge_1[:, 0]), np.nanmax(fp_edge_2[:, 0])])
    y_min = np.nanmin([np.nanmin(poca_y), np.nanmin(nadir_y), np.nanmin(fp_edge_1[:, 1]), np.nanmin(fp_edge_2[:, 1])])
    y_max = np.nanmax([np.nanmax(poca_y), np.nanmax(nadir_y), np.nanmax(fp_edge_1[:, 1]), np.nanmax(fp_edge_2[:, 1])])

    dem_x, dem_y, dem_z, dem_resolution = get_dem_segment(
        dem_path, [(x_min, x_max), (y_min, y_max)],
        grid_xy=False, flatten=False, band_name=dem_band_name,
    )
    dem_z[dem_z <= -9999] = np.nan

    if np.isnan(dem_z).all():
        print("Error: No DEM data found within the record range. Check that the DEM covers the track.")
        sys.exit(1)

    z_min = np.nanmin([np.nanmin(elevation), np.nanmin(dem_z)])
    z_max = np.nanmax([np.nanmax(elevation), np.nanmax(dem_z)])

    dem_x_flat, dem_y_flat = np.meshgrid(dem_x, dem_y)
    dem_x_flat = dem_x_flat.flatten()
    dem_y_flat = dem_y_flat.flatten()
    dem_z_flat = dem_z.flatten()

    # -------------------------------------------------------------------
    # DEM within footprint for each record
    # -------------------------------------------------------------------

    with mp.Pool() as pool:
        fp_results = pool.starmap(
            get_dem_in_fp,
            [
                (dem_x_flat, dem_y_flat, dem_z_flat,
                 across_track_width, along_track_width,
                 heading[k], nadir_x[k], nadir_y[k], altitude[k],
                 range_to_window_top[k], range_to_window_bottom[k],
                 range_to_window_top_le[k], range_to_window_bottom_le[k])
                for k in range(num_records)
            ],
        )

    fp_results   = np.asarray(fp_results, dtype="object")
    dem_in_fp    = fp_results[:, 0]
    dem_in_fp_rw = fp_results[:, 1]
    dem_in_fp_le = fp_results[:, 2]

    # -------------------------------------------------------------------
    # Elevation interpolation
    # -------------------------------------------------------------------

    valid = ~np.isnan(dem_z_flat)
    interp = interpolate.NearestNDInterpolator(
        list(zip(dem_x_flat[valid], dem_y_flat[valid])),
        dem_z_flat[valid],
    )

    nadir_z = interp(nadir_x, nadir_y)
    fp_edge_1[:, 2] = interp(fp_edge_1[:, 0], fp_edge_1[:, 1])
    fp_edge_2[:, 2] = interp(fp_edge_2[:, 0], fp_edge_2[:, 1])

    poca_z        = elevation
    poca_z_interp = np.full(len(poca_x), np.nan)
    poca_nan      = np.isnan(poca_x)
    poca_z_interp[~poca_nan] = interp(poca_x[~poca_nan], poca_y[~poca_nan])
    poca_z_interp[~np.isnan(poca_z)] = np.nan

    # -------------------------------------------------------------------
    # Range window and leading edge 3D shapes
    # -------------------------------------------------------------------

    range_windows_3d        = np.full((num_records, 2, 35, 3), np.nan)
    leading_edge_windows_3d = np.full((num_records, 2, 35, 3), np.nan)

    for k in range(num_records):
        window_3d = []
        for window_range in [
            [range_to_window_top[k], range_to_window_bottom[k]],
            [range_to_window_top_le[k], range_to_window_bottom_le[k]],
        ]:
            if np.isnan(window_range).all() or np.size(window_range) == 0:
                window_3d.append(np.full((2, 35, 3), np.nan))
                continue

            trace_x  = np.linspace(-across_track_width / 2, across_track_width / 2, 17)
            trace_z1 = altitude[k] - np.sqrt(window_range[0] ** 2 - np.square(trace_x))
            trace_z2 = altitude[k] - np.sqrt(window_range[1] ** 2 - np.square(trace_x))

            trace_x  = np.append(np.concatenate((trace_x, trace_x[::-1])), -across_track_width / 2)
            trace_z  = np.append(np.concatenate((trace_z1, trace_z2[::-1])), trace_z1[0])
            trace_xz = np.c_[trace_x, trace_z]

            side1 = np.insert(trace_xz, 1, -along_track_width / 2, axis=1)
            side2 = np.insert(trace_xz, 1,  along_track_width / 2, axis=1)

            rot   = Rotation.from_euler("z", -heading[k], degrees=True)
            side1 = rot.apply(side1) + np.array([nadir_x[k], nadir_y[k], 0], dtype="object")
            side2 = rot.apply(side2) + np.array([nadir_x[k], nadir_y[k], 0], dtype="object")

            window_3d.append(np.array([side1, side2]))

        range_windows_3d[k]        = window_3d[0]
        leading_edge_windows_3d[k] = window_3d[1]

    # -------------------------------------------------------------------
    # Visual / camera setup
    # -------------------------------------------------------------------

    z_data_min, z_data_max = np.nanmin(dem_z), np.nanmax(dem_z)

    if dem_colours is not None:
        surface_colourscale = build_dem_colourscale(dem_colours, dem_colour_bins or [], z_data_min, z_data_max)
    else:
        surface_colourscale = [[0.0, "white"], [1.0, "white"]]

    lighting  = dict(ambient=0.7, diffuse=0.5, roughness=0.9, specular=0.1, fresnel=0.5)
    light_pos = dict(x=100, y=10, z=100000)

    aspect_ratio = ((x_max - x_min) / (y_max - y_min), 1, 0.1)

    nadir_x_cam = (((nadir_x - x_min) / (x_max - x_min)) - 0.5) * aspect_ratio[0]
    nadir_y_cam = (((nadir_y - y_min) / (y_max - y_min)) - 0.5) * aspect_ratio[1]
    nadir_z_cam = (((nadir_z - z_min) / (z_max - z_min)) - 0.5) * aspect_ratio[2]
    cam_centre  = np.column_stack([nadir_x_cam, nadir_y_cam, nadir_z_cam])

    z_view_angle     = 70
    cam_records_back = 20
    backward_vec = np.column_stack([
        np.sin(np.deg2rad(z_view_angle)) * np.cos(np.deg2rad(270 - heading)),
        np.sin(np.deg2rad(z_view_angle)) * np.sin(np.deg2rad(270 - heading)),
        np.full(num_records, np.cos(np.deg2rad(z_view_angle))),
    ])
    cam_dist_per_record = np.linalg.norm([nadir_x_cam[1] - nadir_x_cam[0], nadir_y_cam[1] - nadir_y_cam[0]])
    cam_eye = cam_centre + backward_vec * cam_dist_per_record * cam_records_back

    mid        = num_records // 2
    fp_range_x = np.max(range_windows_3d[mid, 0, :, 0]) - np.min(range_windows_3d[mid, 0, :, 0])
    fp_range_y = np.max(range_windows_3d[mid, 0, :, 1]) - np.min(range_windows_3d[mid, 0, :, 1])
    total      = fp_range_x + fp_range_y
    aspect_ratio_2 = (np.ceil(fp_range_x / total * 10), np.ceil(fp_range_y / total * 10), 3)

    cam_eye_2 = np.column_stack([
        np.cos(np.deg2rad(270 - heading)),
        np.sin(np.deg2rad(270 - heading)),
        np.full(num_records, 0),
    ]) * 5

    nadir_colour    = "#DE3C4B"
    poca_colour     = "#00cee1"
    poca_interp_col = "#307fa6"
    fp_colour       = "#39ac39"
    dem_colour      = "#8c40b8"
    dem_rw_colour   = "#b555b2"
    dem_le_colour   = "#d4aed2"
    wf_colour       = "#8f510a"
    le_colour       = "#DEB887"

    # -------------------------------------------------------------------
    # Build figure
    # -------------------------------------------------------------------

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["Surface View", "Waveform", "Range Window View"],
        column_widths=[0.5, 0.5],
        specs=[
            [{"type": "scene", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "scene"}],
        ],
        vertical_spacing=0.125,
        horizontal_spacing=0.05,
    )

    # Row 1 — surface view
    fig.add_trace(go.Surface(
        name=f"{int(dem_resolution)}m Resolution DEM<br>(Scaled)",
        x=dem_x, y=dem_y, z=dem_z,
        colorscale=surface_colourscale, opacity=0.8,
        showscale=False, showlegend=True,
        lighting=lighting, lightposition=light_pos,
        contours=go.surface.Contours(
            x=go.surface.contours.X(highlight=False),
            y=go.surface.contours.Y(highlight=False),
            z=go.surface.contours.Z(highlight=False),
        ),
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=nadir_x, y=nadir_y, z=nadir_z,
        mode="lines+markers", name="Nadir",
        marker=dict(color=nadir_colour, size=1.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=poca_x, y=poca_y, z=poca_z,
        mode="lines+markers", name="POCA (Level-2 Elevation)",
        marker=dict(color=poca_colour, size=1.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=poca_x, y=poca_y, z=poca_z_interp,
        mode="lines+markers", name="POCA (Interpolated Elevation)",
        marker=dict(color=poca_interp_col, size=1.5),
    ), row=1, col=1)

    for edge, show in [(fp_edge_1, True), (fp_edge_2, False)]:
        fig.add_trace(go.Scatter3d(
            x=edge[:, 0], y=edge[:, 1], z=edge[:, 2],
            mode="lines", name="Footprint Edge",
            marker=dict(color=fp_colour), opacity=0.7,
            showlegend=show,
        ), row=1, col=1)

    fig.add_trace(go.Scatter3d(
        x=dem_in_fp[0][0], y=dem_in_fp[0][1], z=np.array(dem_in_fp[0][2]),
        mode="markers", name="DEM in Footprint", showlegend=False,
        marker=dict(color=dem_colour, size=3),
    ), row=1, col=1)

    fig.update_layout(scene=dict(
        xaxis=dict(title="X", range=[x_min, x_max], showgrid=False),
        yaxis=dict(title="Y", range=[y_min, y_max], showgrid=False),
        zaxis=dict(title="Elevation", showgrid=False),
        xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
        aspectmode="manual",
        aspectratio=dict(x=aspect_ratio[0], y=aspect_ratio[1], z=aspect_ratio[2]),
        bgcolor="#87CEEB",
        camera=dict(
            eye=dict(x=cam_eye[0, 0], y=cam_eye[0, 1], z=cam_eye[0, 2]),
            center=dict(x=cam_centre[0, 0], y=cam_centre[0, 1], z=cam_centre[0, 2]),
        ),
    ))

    # Row 2, col 1 — waveform
    gate_number = np.arange(len(waveform[0]))

    fig.add_trace(go.Scatter(
        x=gate_number, y=waveform[0],
        name="Waveform", mode="lines",
        marker=dict(color=wf_colour),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=[le_index_start[0], le_index_start[0], None, le_index_end[0], le_index_end[0]],
        y=[min(waveform[0]), max(waveform[0]), None, min(waveform[0]), max(waveform[0])],
        name="Leading Edge", mode="lines",
        line=dict(dash="dash"), marker=dict(color=le_colour),
    ), row=2, col=1)

    fig.update_layout(
        xaxis=dict(title="Gate Number"),
        yaxis=dict(title="Power", autorange=False, range=[0, 1.1 * np.max(waveform[0])]),
    )

    # Row 2, col 2 — range window view
    fig.add_trace(go.Scatter3d(
        x=dem_in_fp[0][0], y=dem_in_fp[0][1], z=dem_in_fp[0][2],
        mode="markers", name="DEM in Footprint",
        marker=dict(color=dem_colour, size=2),
    ), row=2, col=2)

    fig.add_trace(go.Scatter3d(
        x=dem_in_fp_rw[0][0], y=dem_in_fp_rw[0][1], z=dem_in_fp_rw[0][2],
        mode="markers", name="DEM in Range Window",
        marker=dict(color=dem_rw_colour, size=2),
    ), row=2, col=2)

    fig.add_trace(go.Scatter3d(
        x=dem_in_fp_le[0][0], y=dem_in_fp_le[0][1], z=dem_in_fp_le[0][2],
        mode="markers", name="DEM in Leading Edge",
        marker=dict(color=dem_le_colour, size=2),
    ), row=2, col=2)

    fig.add_trace(go.Scatter3d(
        x=np.concatenate([range_windows_3d[0][0][:, 0], [None], range_windows_3d[0][1][:, 0]]),
        y=np.concatenate([range_windows_3d[0][0][:, 1], [None], range_windows_3d[0][1][:, 1]]),
        z=np.concatenate([range_windows_3d[0][0][:, 2], [None], range_windows_3d[0][1][:, 2]]),
        mode="lines", name="Range Window",
        marker=dict(color=fp_colour),
    ), row=2, col=2)

    fig.add_trace(go.Scatter3d(
        x=np.concatenate([leading_edge_windows_3d[0][0][:, 0], [None], leading_edge_windows_3d[0][1][:, 0]]),
        y=np.concatenate([leading_edge_windows_3d[0][0][:, 1], [None], leading_edge_windows_3d[0][1][:, 1]]),
        z=np.concatenate([leading_edge_windows_3d[0][0][:, 2], [None], leading_edge_windows_3d[0][1][:, 2]]),
        mode="lines", name="Leading Edge Window",
        marker=dict(color=fp_colour), line=dict(dash="dash"),
    ), row=2, col=2)

    for colour, z_val, name in [
        (poca_colour,     poca_z[0],        "POCA (L2 Elevation)"),
        (poca_interp_col, poca_z_interp[0], "POCA (Interpolated Elevation)"),
    ]:
        fig.add_trace(go.Scatter3d(
            x=[poca_x[0]], y=[poca_y[0]], z=[z_val],
            mode="markers", name=name, showlegend=False,
            marker=dict(color=colour, size=4, symbol="x"),
        ), row=2, col=2)

    fig.update_layout(scene2=dict(
        xaxis=dict(title="X", showgrid=False),
        yaxis=dict(title="Y", showgrid=False),
        zaxis=dict(title="Elevation (Scaled)", tickformat=".0f", dtick=40, showgrid=False),
        xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
        aspectmode="manual",
        aspectratio=dict(x=aspect_ratio_2[0], y=aspect_ratio_2[1], z=aspect_ratio_2[2]),
        camera=dict(eye=dict(x=cam_eye_2[0, 0], y=cam_eye_2[0, 1], z=cam_eye_2[0, 2])),
    ))

    # -------------------------------------------------------------------
    # Animation frames
    # -------------------------------------------------------------------

    frames = [
        dict(
            name=k + record_range[0],
            data=[
                go.Scatter3d(x=dem_in_fp[k][0], y=dem_in_fp[k][1], z=dem_in_fp[k][2]),
                go.Scatter(x=gate_number, y=waveform[k]),
                go.Scatter(
                    x=[le_index_start[k], le_index_start[k], None, le_index_end[k], le_index_end[k]],
                    y=[min(waveform[k]), max(waveform[k]), None, min(waveform[k]), max(waveform[k])],
                ),
                go.Scatter3d(x=dem_in_fp[k][0],    y=dem_in_fp[k][1],    z=dem_in_fp[k][2]),
                go.Scatter3d(x=dem_in_fp_rw[k][0], y=dem_in_fp_rw[k][1], z=dem_in_fp_rw[k][2]),
                go.Scatter3d(x=dem_in_fp_le[k][0], y=dem_in_fp_le[k][1], z=dem_in_fp_le[k][2]),
                go.Scatter3d(
                    x=np.concatenate([range_windows_3d[k][0][:, 0], [None], range_windows_3d[k][1][:, 0]]),
                    y=np.concatenate([range_windows_3d[k][0][:, 1], [None], range_windows_3d[k][1][:, 1]]),
                    z=np.concatenate([range_windows_3d[k][0][:, 2], [None], range_windows_3d[k][1][:, 2]]),
                ),
                go.Scatter3d(
                    x=np.concatenate([leading_edge_windows_3d[k][0][:, 0], [None], leading_edge_windows_3d[k][1][:, 0]]),
                    y=np.concatenate([leading_edge_windows_3d[k][0][:, 1], [None], leading_edge_windows_3d[k][1][:, 1]]),
                    z=np.concatenate([leading_edge_windows_3d[k][0][:, 2], [None], leading_edge_windows_3d[k][1][:, 2]]),
                ),
                go.Scatter3d(x=[poca_x[k]], y=[poca_y[k]], z=[poca_z[k]]),
                go.Scatter3d(x=[poca_x[k]], y=[poca_y[k]], z=[poca_z_interp[k]]),
            ],
            layout=dict(
                scene=dict(camera=dict(
                    eye=dict(x=cam_eye[k, 0], y=cam_eye[k, 1], z=cam_eye[k, 2]),
                    center=dict(x=cam_centre[k, 0], y=cam_centre[k, 1], z=cam_centre[k, 2]),
                )),
                scene2=dict(camera=dict(
                    eye=dict(x=cam_eye_2[k, 0], y=cam_eye_2[k, 1], z=cam_eye_2[k, 2]),
                )),
                yaxis=dict(range=[0, 1.1 * np.max(waveform[k])]),
            ),
            traces=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        )
        for k in range(num_records)
    ]

    # -------------------------------------------------------------------
    # Controls
    # -------------------------------------------------------------------

    updatemenus = [{
        "buttons": [
            {
                "args": [None, {"frame": {"duration": 1500, "redraw": False}, "fromcurrent": True, "transition": {"duration": 1}, "mode": "next"}],
                "label": "Play",
                "method": "animate",
            },
            {
                "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                "label": "Pause",
                "method": "animate",
            },
        ],
        "direction": "left",
        "pad": {"r": 10, "t": 85},
        "showactive": False,
        "type": "buttons",
        "x": 0.1, "xanchor": "right",
        "y": 0,   "yanchor": "top",
    }]

    sliders = [{
        "yanchor": "top", "xanchor": "left",
        "currentvalue": {"font": {"size": 16}, "prefix": "Record: ", "xanchor": "right"},
        "transition": {"duration": 0},
        "pad": {"b": 10, "t": 50},
        "len": 0.9, "x": 0.1, "y": 0,
        "steps": [
            {
                "args": [[k], {"frame": {"duration": 1500, "redraw": False}, "fromcurrent": True, "transition": {"duration": 1}, "mode": "next"}],
                "label": k,
                "method": "animate",
            }
            for k in range(record_range[0], record_range[1])
        ],
    }]

    # -------------------------------------------------------------------
    # Finalise figure layout
    # -------------------------------------------------------------------

    satellite_name = "Sentinel-3" if satellite == "S3" else "CryoSat-2" if satellite == "CS2" else "" 
    fig.frames = frames
    fig.update_layout(
        updatemenus=updatemenus,
        sliders=sliders,
        title=f"{satellite_name} SAR Altimetry - Records {record_range[0]}-{record_range[1]}<br>{l2_track_path}",
        template="plotly_dark",
        title_x=0.5,
        title_y=0.965,
        legend_itemclick=False,
        legend_itemdoubleclick=False,
        legend=dict(font=dict(size=10)),
    )

    base_name = output_name if output_name is not None \
        else f"{satellite}_SAR_Altimetry_Records_{record_range[0]}-{record_range[1]}"

    # -------------------------------------------------------------------
    # HTML output
    # -------------------------------------------------------------------

    print("\n")
    if generate_html:
        html_path = f"{base_name}.html"
        with tqdm(total=1, desc="Creating HTML", bar_format="{l_bar}{bar}| [{elapsed}]") as pbar:
            fig.write_html(html_path, auto_play=False)
            pbar.update(1)
        print(f"HTML saved to: {html_path}")

    # -------------------------------------------------------------------
    # Video output
    # -------------------------------------------------------------------

    if generate_video:
        try:
            import imageio
        except ImportError:
            print("Error: imageio is required for video export. Install it with: pip install imageio[ffmpeg]")
            sys.exit(1)

        try:
            import kaleido
            from packaging.version import Version
            if Version(kaleido.__version__) >= Version("0.3"):
                    print(
                        f"Error: Kaleido {kaleido.__version__} requires a system Chrome install, "
                        "which is not suitable for headless VMs. "
                        "Downgrade to the self-contained 0.2.x release: "
                        "pip install kaleido==0.2.1"
                    )
                    sys.exit(1)
        except ImportError:
            print(
                "Error: kaleido is required for video export. "
                "Install the headless-compatible version with: pip install kaleido==0.2.1"
            )
            sys.exit(1)

        video_path = f"{base_name}.mp4"

        with tempfile.TemporaryDirectory(prefix="altimetry_frames_", dir=os.getcwd()) as tmp_dir:

            # Walk through each animation frame, apply its data and layout overrides
            # to the live figure, and export it as a PNG via kaleido.
            print("\n")
            with tqdm(total=num_records, desc="Creating video", unit="frame") as pbar:
                for k in range(num_records):
                    frame = frames[k]

                    for trace_idx, new_trace in zip(frame["traces"], frame["data"]):
                        fig.data[trace_idx].update(new_trace)

                    frame_layout = frame.get("layout", {})
                    if frame_layout:
                        fig.update_layout(frame_layout)

                    frame_path = os.path.join(tmp_dir, f"frame_{k:06d}.png")
                    try:
                        fig.write_image(frame_path, format="png", width=video_resolution[0], height=video_resolution[1])
                    except RuntimeError as e:
                        print(
                            f"Error: Frame rendering failed: {e} "
                            "This usually means kaleido is the wrong version. "
                            "Use the self-contained 0.2.x release on headless VMs: "
                            "pip install kaleido==0.2.1"
                        )
                        sys.exit(1)

                    pbar.update(1)

            frame_files = sorted(
                os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".png")
            )
            with tqdm(total=len(frame_files), desc="Encoding video", unit="frame") as pbar:
                with imageio.get_writer(
                video_path,
                fps=video_fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=1,
                output_params=["-crf", "23", "-preset", "slow"],
                ) as writer:
                    for frame_file in frame_files:
                        writer.append_data(imageio.imread(frame_file))

        print(f"Video saved to: {video_path}")