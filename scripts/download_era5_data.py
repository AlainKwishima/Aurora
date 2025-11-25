#!/usr/bin/env python3
"""
ERA5 Data Download Script for Rwanda
====================================

Downloads ERA5 reanalysis data for Rwanda region from Copernicus Climate Data Store.

Requirements:
    - CDS API credentials in ~/.cdsapirc
    - Sign up at: https://cds.climate.copernicus.eu/user/register
    - Get API key from: https://cds.climate.copernicus.eu/api-how-to

Usage:
    python download_era5_data.py --years 2020-2024 --output data/rwanda_era5.nc
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
import xarray as xr
import numpy as np

try:
    import cdsapi
except ImportError:
    print("❌ cdsapi not installed. Install with: pip install cdsapi")
    exit(1)


# Rwanda geographic bounds
RWANDA_BOUNDS = {
    'north': -1.047,
    'south': -2.917,
    'west': 28.862,
    'east': 30.899,
}

# Variables to download
ERA5_VARIABLES = [
    '2m_temperature',           # 2t
    'total_precipitation',       # tp
    '10m_u_component_of_wind',  # 10u
    '10m_v_component_of_wind',  # 10v
    'mean_sea_level_pressure',  # msl
]

# Time resolution: 6-hourly
TIMES = ['00:00', '06:00', '12:00', '18:00']


def setup_argparse():
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(
        description='Download ERA5 data for Rwanda',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--years',
        type=str,
        default='2020-2024',
        help='Year range to download (e.g., 2020-2024 or 2023)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/rwanda_era5.nc',
        help='Output NetCDF file path'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate downloaded data'
    )
    
    parser.add_argument(
        '--split',
        action='store_true',
        help='Split into train/val/test sets after download'
    )
    
    return parser.parse_args()


def parse_years(year_str):
    """Parse year string into list of years."""
    if '-' in year_str:
        start, end = map(int, year_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(year_str)]


def download_era5(years, output_path):
    """Download ERA5 data from CDS."""
    print("🌍 ERA5 Data Download for Rwanda")
    print("=" * 60)
    print(f"Years: {min(years)} - {max(years)}")
    print(f"Variables: {', '.join(ERA5_VARIABLES)}")
    print(f"Region: Rwanda ({RWANDA_BOUNDS})")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize CDS API client
    try:
        c = cdsapi.Client()
    except Exception as e:
        print(f"\n❌ Failed to initialize CDS API client: {e}")
        print("\nPlease ensure:")
        print("1. You have an account at: https://cds.climate.copernicus.eu")
        print("2. Your API key is in ~/.cdsapirc")
        print("3. Format of ~/.cdsapirc:")
        print("   url: https://cds.climate.copernicus.eu/api/v2")
        print("   key: YOUR_UID:YOUR_API_KEY")
        return False
    
    # Prepare request
    request = {
        'product_type': 'reanalysis',
        'variable': ERA5_VARIABLES,
        'year': [str(y) for y in years],
        'month': [f'{m:02d}' for m in range(1, 13)],
        'day': [f'{d:02d}' for d in range(1, 32)],
        'time': TIMES,
        'area': [
            RWANDA_BOUNDS['north'],
            RWANDA_BOUNDS['west'],
            RWANDA_BOUNDS['south'],
            RWANDA_BOUNDS['east'],
        ],
        'format': 'netcdf',
    }
    
    print(f"\n📥 Downloading data...")
    print(f"Estimated size: ~{len(years) * 100} MB")
    print(f"Estimated time: ~{len(years) * 5} minutes")
    print("\nThis may take a while. Please be patient...\n")
    
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            request,
            output_path
        )
        print(f"\n✅ Download complete: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False


def validate_data(file_path):
    """Validate downloaded ERA5 data."""
    print("\n🔍 Validating downloaded data...")
    
    try:
        ds = xr.open_dataset(file_path)
        
        print("\n✓ File opened successfully")
        print(f"\nDataset dimensions:")
        for dim, size in ds.dims.items():
            print(f"  {dim}: {size}")
        
        print(f"\nDataset variables:")
        for var in ds.data_vars:
            print(f"  {var}: {ds[var].shape}")
        
        print(f"\nTime range:")
        print(f"  Start: {ds.time.values[0]}")
        print(f"  End: {ds.time.values[-1]}")
        print(f"  Total timesteps: {len(ds.time)}")
        
        print(f"\nSpatial coverage:")
        print(f"  Latitude: [{ds.latitude.min().values:.3f}, {ds.latitude.max().values:.3f}]")
        print(f"  Longitude: [{ds.longitude.min().values:.3f}, {ds.longitude.max().values:.3f}]")
        
        # Check for missing values
        print(f"\nData quality:")
        for var in ds.data_vars:
            missing = ds[var].isnull().sum().values
            total = ds[var].size
            pct = (missing / total) * 100
            print(f"  {var}: {missing}/{total} missing ({pct:.2f}%)")
        
        print("\n✅ Validation complete!")
        ds.close()
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return False


def split_data(file_path):
    """Split data into train/val/test sets."""
    print("\n📊 Splitting data into train/val/test...")
    
    try:
        ds = xr.open_dataset(file_path)
        
        # Sort by time
        ds = ds.sortby('time')
        
        # Split by year
        # Train: 70% (oldest years)
        # Val: 15% (middle years)
        # Test: 15% (most recent years)
        
        years = ds.time.dt.year.values
        unique_years = np.unique(years)
        
        n_years = len(unique_years)
        train_years = unique_years[:int(n_years * 0.7)]
        val_years = unique_years[int(n_years * 0.7):int(n_years * 0.85)]
        test_years = unique_years[int(n_years * 0.85):]
        
        # Create splits
        train_ds = ds.sel(time=ds.time.dt.year.isin(train_years))
        val_ds = ds.sel(time=ds.time.dt.year.isin(val_years))
        test_ds = ds.sel(time=ds.time.dt.year.isin(test_years))
        
        # Save splits
        base_path = Path(file_path)
        train_path = base_path.parent / f"{base_path.stem}_train.nc"
        val_path = base_path.parent / f"{base_path.stem}_val.nc"
        test_path = base_path.parent / f"{base_path.stem}_test.nc"
        
        print(f"\nSaving splits:")
        print(f"  Train: {train_path} ({len(train_ds.time)} timesteps, years {train_years[0]}-{train_years[-1]})")
        train_ds.to_netcdf(train_path)
        
        print(f"  Val: {val_path} ({len(val_ds.time)} timesteps, years {val_years[0]}-{val_years[-1]})")
        val_ds.to_netcdf(val_path)
        
        print(f"  Test: {test_path} ({len(test_ds.time)} timesteps, years {test_years[0]}-{test_years[-1]})")
        test_ds.to_netcdf(test_path)
        
        print("\n✅ Data split complete!")
        
        ds.close()
        return True
        
    except Exception as e:
        print(f"\n❌ Split failed: {e}")
        return False


def main():
    """Main execution function."""
    args = setup_argparse()
    
    # Parse years
    years = parse_years(args.years)
    
    # Download data
    success = download_era5(years, args.output)
    
    if not success:
        print("\n❌ Download failed. Exiting.")
        return 1
    
    # Validate if requested
    if args.validate:
        if not validate_data(args.output):
            print("\n⚠️  Validation had issues, but file was downloaded.")
    
    # Split if requested
    if args.split:
        if not split_data(args.output):
            print("\n⚠️  Split had issues, but file was downloaded.")
    
    print("\n" + "=" * 60)
    print("✅ All operations complete!")
    print("=" * 60)
    print(f"\nData ready at: {args.output}")
    print("\nNext steps:")
    print("1. Update training script to use this data")
    print("2. Train model on GPU (Kaggle/Colab)")
    print("3. Evaluate performance")
    
    return 0


if __name__ == "__main__":
    exit(main())
