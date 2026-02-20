# SAR Altimetry Flyover Visualiser
An interactive visualisation tool for SAR satellite altimetry alongside Digital Elevation Model (DEM) data, designed to make the altimetry process more intuitive. Animates through records as a flyover with full interactive camera manipulation.

---
Made by **Joe Phillips**  

[![GitHub](https://badgen.net/badge/icon/GitHub/green?icon=github&label)](https://github.com/Joe-Phillips)
[![LinkedIn](https://badgen.net/badge/icon/linkedin/blue?icon=linkedin&label)](https://www.linkedin.com/in/joe-b-phillips/)
&nbsp; ✉️ j.phillips5@lancaster.ac.uk

---
## Overview
The tool takes SAR altimetry track data and a DEM GeoTIFF and produces an animated 3D visualisation that flies along the satellite track record by record. Each frame shows the satellite footprint, range window, leading edge window, and POCA position overlaid on the DEM surface, alongside a live waveform plot and a close-up range window view. Output can be an interactive HTML file, an MP4 video, or both.

Currently, **Sentinel-3** and **CryoSat-2** are supported.

## Structure
```
├── plot_sar_flyover.py    # Entry point - CLI argument parsing
├── plot.py                # Core visualisation logic
├── altimetry.py           # Range correction, leading edge detection, heading
└── dem.py                 # DEM loading, subsetting, and footprint intersection
```

## Usage

Install the required packages:
```bash
pip install -r requirements.txt
```

Then run from the command line:
```bash
python plot-sar-flyover.py <satellite> <start> <end> <dem_path> <l2_path> [options]
```

### Positional Arguments

| Argument | Type | Description |
|---|---|---|
| `satellite` | string | Satellite type: `S3` (Sentinel-3) or `CS2` (CryoSat-2). |
| `start` | int | First record to plot. |
| `end` | int | Last record to plot (exclusive). |
| `dem_path` | string | Path to the DEM GeoTIFF. Resolution should ideally be below the satellite's along-track footprint size. |
| `l2_path` | string | Path to the Level-2 NetCDF track data. |

### Optional Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--l1_path` | string | - | Path to the Level-1 NetCDF track data. Required for CryoSat-2. |
| `--dem_proj` | string | `epsg:3031` | DEM projection in EPSG format. Must be in metres. |
| `--dem_band` | string | - | Name of the DEM band to use. If omitted, the first band is used. |
| `--dem_colours` | string(s) | - | Colours (named or hex) for DEM shading, e.g. `--dem_colours blue white red`. |
| `--dem_colour_bins` | float(s) | - | Bin edges in metres separating `--dem_colours` (one fewer than colours), e.g. `--dem_colour_bins 0 500`. |
| `--output_name` | string | Auto-generated | Base name for output files, without extension. |
| `--html` | bool | `true` | Generate an interactive HTML file. |
| `--video` | bool | `true` | Generate an MP4 video. Requires `kaleido==0.2.1` and `imageio[ffmpeg]`. |
| `--video-fps` | int | `10` | Frames (records) per second for the output video. |
| `--video-resolution` | int int | `1280 720` | Output video resolution in pixels, e.g. `--video-resolution 1920 1080`. |

### Example
```bash
python plot_sar_flyover.py CS2 1000 1250 example_folder/DEM.tif example_folder/L2_track.nc \
    --l1_path example_folder/L1_track.nc \
    --dem_proj epsg:3031 \
    --dem_colours blue white red \
    --dem_colour_bins 0 500 \
    --video-resolution 1920 1080
```

## Preview

![Example](https://github.com/Joe-Phillips/SAR-Altimetry-Plotter/blob/main/s3_example.gif?raw=true)
