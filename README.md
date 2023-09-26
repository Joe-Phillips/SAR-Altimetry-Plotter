# SAR-Altimetry-Plotter

This project works to interactively plot **Satellite Radar SAR Altimetry** data alongside **Digital Elevation Model** (DEM) data in order to help visualise the process more intuitively. The plot animates through records and allows for full view-manipulation within each subplot.

Currently, only **Sentinel-3** and **CryoSat-2** are supported.

Made by Joe Phillips.

[![Repo](https://badgen.net/badge/icon/GitHub/green?icon=github&label)](https://github.com/Joe-Phillips) 
[![Repo](https://badgen.net/badge/icon/linkedin/blue?icon=linkedin&label)](https://www.linkedin.com/in/joe-b-phillips/)
&nbsp;✉️ j.phillips5@lancaster.ac.uk

## 🛠️ Usage Guide

First, make sure you have installed the required packages. This can be done via **pip install -r requirements.txt**.

To generate a plot, simply run **SAR_Altimetry_Plotter.py** from the command line with the following arguments:

- **SAT** (string): *The satellite (S3 or CS2).*
- **RANGE** (tuple): *A tuple containing the range of records to plot (start_record, end_record).*
- **DEM_PATH** (string): *The path to the DEM file. Ideally, the DEM should have resolution below the size of the satellite footprint along-track.*
- **L2_PATH** (string): *The path to the Level-2 track data.*
- --**L1_PATH** (string, optional): *The path to the Level-1 track data. Required for CryoSat-2.*
- --**DEM_PROJ** (string, optional): *The projection of the DEM data in EPSG format (default is EPSG:3031). This should be in meters.*

This produces an interactive HTML file, which can be opened and viewed in a browser.

### Example:

- python SAR_Altimetry_Plotter.py CS2 (1000,1250) example_folder/DEM.tif example_folder/L2_track.nc --L1_PATH example_folder/L1_track.nc --DEM_PROJ EPSG:3031

<br>

## :camera: Images 
**Sentinel-3**
![alt text](https://github.com/Joe-Phillips/SAR-Altimetry-Plotter/blob/main/S3_Example_Figure.png?raw=true)

<br>

**CryoSat-2**
![alt text](https://github.com/Joe-Phillips/SAR-Altimetry-Plotter/blob/main/CS2_Example_Figure.png?raw=true)
