import cdsapi
from pathlib import Path
import xarray as xr

import sys
import os
sys.path.append(os.path.abspath(".."))

######################
## Global variables ##
######################

ALL_YEARS = [   # could change this later
        "2018", 
        "2019",
        "2020"
    ]

ALL_MONTHS = [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ]

MONTH_GROUPS = [ # since we're hitting 'request too big' API errors 
    ["01", "02", "03", "04", "05", "06"],
    ["07", "08", "09", "10", "11", "12"]
]

ALL_DAYS = [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ]

ALL_HOURS = [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ]

ALL_FEATURES = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "surface_pressure"
    ] 

# TODO: how are we gonna add the cyclical variables ?

TEHACHAPI_COORDS = (35.042933, -118.258788)
TEHCHAPI_AREA = [35.05, -118.30, 35.00, -118.20]


######################
## Helper functions ##
######################

DATA_DIR = Path(__file__).resolve().parent

def load_data(                
        dataset:str="reanalysis-era5-single-levels",
        variables=ALL_FEATURES,
        years=ALL_YEARS,
        month_groups=MONTH_GROUPS,
        days=ALL_DAYS,
        hours=ALL_HOURS,
    ):
    """
    Downloads .nc data files into the data/ directory.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_targets = []

    client = cdsapi.Client(verify=False)

    for year in years:
        for i, month_group in enumerate(month_groups, start=1):
            
            # Save inside data/
            target = DATA_DIR / f"era5_tehachapi_{year}_H{i}.nc"
            all_targets.append(target)

            if target.exists():
                print(f"Skipping existing file {target}")
                continue

            request = {
                "product_type": ["reanalysis"],
                "variable": variables,
                "year": year,
                "month": month_group,
                "day": days,
                "time": hours,
                "data_format": "netcdf",
                "download_format": "unarchived",
                "area": TEHCHAPI_AREA
            }

            print(f"Requesting {year}, chunk {i} → {target}")
            client.retrieve(dataset, request, str(target))
    
    return all_targets

def combine_nc_files(targets):
    """
    Open multiple NetCDF files and combine them into one xarray.Dataset
    aligned by coordinates (valid_time, latitude, longitude).
    `targets` is a list of Path objects.
    """
    # Keep only existing files, convert to strings for xarray
    paths = [str(p) for p in targets if p.exists()]

    if not paths:
        raise ValueError("No existing NetCDF files found in targets.")

    ds = xr.open_mfdataset(
        paths,
        combine="by_coords",   # align by valid_time / lat / lon  TODO: are we sure this is it ? maybe by_coords is something built-in to netcfd files, cause we don't have a column name that says by_coords
        engine="netcdf4",      # explicit is nice but optional
    )


    # we don't want ('number', 'expver'). Let's drop em like hot potato
    variables_to_drop = ['number', 'expver']
    ds = ds.drop_vars(variables_to_drop)
    
    return ds


def get_dataframe(
        dataset:str="reanalysis-era5-single-levels",
        variables=ALL_FEATURES,
        years=ALL_YEARS,
        month_groups=MONTH_GROUPS,
        days=ALL_DAYS,
        hours=ALL_HOURS,
        # area=TEHCHAPI_AREA # hardcoded for now
    ):

    # load data
    all_targets = load_data(
        dataset=dataset,
        variables=variables,
        years=years,
        month_groups=month_groups,
        days=days,
        hours=hours,
        # area=area # hardcoded for now
        )

    # combine them
    ds = combine_nc_files(all_targets)

    # make them into a dataframe
    df = ds.to_dataframe()
    df = df.reset_index() # move all index levels (time, lat, lon) to regular columns
    df.rename(columns={'valid_time': 'datetime'}, inplace=True)
    df.drop(columns=['latitude', 'longitude'], inplace=True) # SINCE THEY ARE CONSTANT FOR NOW. maybe in the future we change

    return df

FullDataFrame = get_dataframe()
