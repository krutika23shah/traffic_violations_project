"""Utility functions for data processing."""

import pandas as pd
import numpy as np
import re
from typing import Tuple, Optional


def load_data(filepath: str, sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        sample_size: If provided, load only a sample for testing
    
    Returns:
        DataFrame with loaded data
    """
    try:
        if sample_size:
            df = pd.read_csv(filepath, nrows=sample_size, low_memory=False)
        else:
            df = pd.read_csv(filepath, low_memory=False)
        print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at: {filepath}")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")


def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save DataFrame to CSV."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved {len(df):,} rows to {filepath}")


def get_data_summary(df: pd.DataFrame) -> dict:
    """Generate summary statistics for the dataset."""
    return {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.astype(str).to_dict()
    }


def standardize_boolean(value) -> Optional[bool]:
    """Convert various boolean representations to True/False/None."""
    if pd.isna(value):
        return None
    
    value_str = str(value).strip().upper()
    
    if value_str in ['YES', 'Y', 'TRUE', '1']:
        return True
    elif value_str in ['NO', 'N', 'FALSE', '0', '']:
        return False
    else:
        return None


def extract_district_number(subagency: str) -> Optional[int]:
    """Extract district number from SubAgency field."""
    if pd.isna(subagency):
        return None
    
    match = re.search(r'(\d+)(?:st|nd|rd|th)?\s*District', str(subagency), re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def parse_geolocation(geolocation: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse geolocation string into latitude and longitude."""
    if pd.isna(geolocation):
        return None, None
    
    try:
        # Handle format like "(38.123, -77.456)"
        clean = str(geolocation).strip('()').replace(' ', '')
        parts = clean.split(',')
        if len(parts) == 2:
            lat, lon = float(parts[0]), float(parts[1])
            if lat != 0 and lon != 0:  # Filter out invalid (0,0)
                return lat, lon
    except (ValueError, IndexError):
        pass
    
    return None, None
