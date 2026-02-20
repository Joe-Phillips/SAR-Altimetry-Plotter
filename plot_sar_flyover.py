import argparse
from plot import plot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create an interactive plot visualising SAR satellite altimetry alongside DEM data."
    )

    # Positional arguments
    parser.add_argument("satellite",    choices=["S3", "CS2"],  help="Satellite type.")
    parser.add_argument("start",        type=int,               help="First record to plot.")
    parser.add_argument("end",          type=int,               help="Last record to plot (exclusive).")
    parser.add_argument("dem_path",                             help="Path to the DEM GeoTIFF. Resolution should ideally be below the satellite's along-track footprint size.")
    parser.add_argument("l2_path",                              help="Path to the Level-2 NetCDF track data.")

    # Optional arguments
    parser.add_argument("--l1_path",         default=None,       help="Path to the Level-1 NetCDF track data (required for CS2).")
    parser.add_argument("--dem_proj",        default="epsg:3031", help="DEM projection in EPSG format, must be in metres (default: epsg:3031).")
    parser.add_argument("--dem_band",        default=None,        help="Name of the DEM band to use. If omitted, the first band is used.")
    parser.add_argument("--dem_colours",     default=None,        nargs="+", metavar="COLOUR",
                        help="Colours (named or hex) for DEM shading, e.g. --dem_colours blue white red")
    parser.add_argument("--dem_colour_bins", default=None,        nargs="+", type=float, metavar="METRES",
                        help="Bin edges in metres separating dem_colours (one fewer than colours), e.g. --dem_colour_bins 0 500")

    parser.add_argument("--output_name",    default=None,        help="Base name for output files, without extension (default: auto-generated from satellite and record range).")

    # Output format flags
    parser.add_argument("--html",      type=lambda x: x.lower() != "false", default=True,  metavar="BOOL",
                        help="Generate an interactive HTML file (default: true).")
    parser.add_argument("--video",     type=lambda x: x.lower() != "false", default=True,  metavar="BOOL",
                        help="Generate an MP4 video (default: true). Requires kaleido and imageio[ffmpeg].")
    parser.add_argument("--video-fps", type=int, default=10, metavar="FPS",
                        help="Frames (records) per second for the output video (default: 10).")
    parser.add_argument("--video-resolution", type=int, default=[1280, 720], nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="Output video resolution in pixels (default: 1280 720).")

    args = parser.parse_args()

    if not args.html and not args.video:
        parser.error("At least one output format must be enabled (--html and/or --video).")

    plot(
        args.satellite,
        (args.start, args.end),
        args.dem_path,
        args.l2_path,
        args.l1_path,
        args.dem_proj,
        args.dem_band,
        args.dem_colours,
        args.dem_colour_bins,
        generate_html=args.html,
        generate_video=args.video,
        video_fps=args.video_fps,
        video_resolution=tuple(args.video_resolution),
        output_name=args.output_name,
    )