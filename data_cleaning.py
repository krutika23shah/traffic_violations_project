"""Data cleaning and preprocessing module."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    VALID_LATITUDE_RANGE, VALID_LONGITUDE_RANGE,
    MIN_VEHICLE_YEAR, MAX_VEHICLE_YEAR,
    BOOLEAN_COLUMNS, COLOR_MAPPING, MAKE_MAPPING
)
from src.utils import standardize_boolean, extract_district_number, parse_geolocation


class TrafficDataCleaner:
    """
    A class to clean and preprocess traffic violation data.
    
    This class provides methods for:
    - Removing duplicates
    - Standardizing date/time formats
    - Cleaning geographic coordinates
    - Normalizing categorical variables
    - Handling missing values
    - Feature engineering
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the cleaner with a DataFrame.
        
        Args:
            df: Raw traffic violations DataFrame
        """
        self.df = df.copy()
        self.cleaning_log = []
        self._log(f"Initialized with {len(self.df):,} rows")
    
    def _log(self, message: str) -> None:
        """Add message to cleaning log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cleaning_log.append(f"[{timestamp}] {message}")
        print(message)
    
    def get_cleaning_report(self) -> str:
        """Return the full cleaning log."""
        return "\n".join(self.cleaning_log)
    
    def remove_duplicates(self) -> 'TrafficDataCleaner':
        """Remove duplicate records based on SeqID."""
        initial_count = len(self.df)
        
        # Check for exact duplicates
        exact_dups = self.df.duplicated().sum()
        self.df = self.df.drop_duplicates()
        
        # Note: Multiple rows with same SeqID but different charges are valid
        # Only remove true duplicates (all columns identical)
        
        removed = initial_count - len(self.df)
        self._log(f"Removed {removed:,} duplicate rows ({exact_dups:,} exact duplicates)")
        
        return self
    
    def clean_datetime(self) -> 'TrafficDataCleaner':
        """Standardize date and time columns."""
        # Clean Date Of Stop
        if 'Date Of Stop' in self.df.columns:
            self.df['Date Of Stop'] = pd.to_datetime(
                self.df['Date Of Stop'], 
                errors='coerce',
                format='mixed'
            )
            invalid_dates = self.df['Date Of Stop'].isna().sum()
            self._log(f"Converted 'Date Of Stop' to datetime ({invalid_dates:,} invalid)")
        
        # Clean Time Of Stop
        if 'Time Of Stop' in self.df.columns:
            # Replace dots with colons for time format
            self.df['Time Of Stop'] = (
                self.df['Time Of Stop']
                .astype(str)
                .str.replace('.', ':', regex=False)
                .str.strip()
            )
            
            # Try to parse as time
            def parse_time(t):
                try:
                    if pd.isna(t) or t == 'nan':
                        return None
                    return pd.to_datetime(t, format='%H:%M:%S').time()
                except:
                    try:
                        return pd.to_datetime(t, format='%H:%M').time()
                    except:
                        return None
            
            self.df['Time Of Stop'] = self.df['Time Of Stop'].apply(parse_time)
            self._log("Standardized 'Time Of Stop' format")
        
        # Create combined datetime column
        if 'Date Of Stop' in self.df.columns and 'Time Of Stop' in self.df.columns:
            def combine_datetime(row):
                if pd.isna(row['Date Of Stop']) or row['Time Of Stop'] is None:
                    return pd.NaT
                try:
                    return datetime.combine(row['Date Of Stop'].date(), row['Time Of Stop'])
                except:
                    return pd.NaT
            
            self.df['DateTime'] = self.df.apply(combine_datetime, axis=1)
            self._log("Created combined 'DateTime' column")
        
        return self
    
    def clean_coordinates(self) -> 'TrafficDataCleaner':
        """Validate and clean geographic coordinates."""
        # Clean Latitude
        if 'Latitude' in self.df.columns:
            self.df['Latitude'] = pd.to_numeric(self.df['Latitude'], errors='coerce')
            
            # Replace invalid values (0 or out of range) with NaN
            invalid_mask = (
                (self.df['Latitude'] == 0) |
                (self.df['Latitude'] < VALID_LATITUDE_RANGE[0]) |
                (self.df['Latitude'] > VALID_LATITUDE_RANGE[1])
            )
            invalid_count = invalid_mask.sum()
            self.df.loc[invalid_mask, 'Latitude'] = np.nan
            self._log(f"Cleaned Latitude: {invalid_count:,} invalid values set to NaN")
        
        # Clean Longitude
        if 'Longitude' in self.df.columns:
            self.df['Longitude'] = pd.to_numeric(self.df['Longitude'], errors='coerce')
            
            invalid_mask = (
                (self.df['Longitude'] == 0) |
                (self.df['Longitude'] < VALID_LONGITUDE_RANGE[0]) |
                (self.df['Longitude'] > VALID_LONGITUDE_RANGE[1])
            )
            invalid_count = invalid_mask.sum()
            self.df.loc[invalid_mask, 'Longitude'] = np.nan
            self._log(f"Cleaned Longitude: {invalid_count:,} invalid values set to NaN")
        
        # Parse Geolocation if it exists and fill missing lat/lon
        if 'Geolocation' in self.df.columns:
            geo_parsed = self.df['Geolocation'].apply(parse_geolocation)
            self.df['Geo_Lat'] = geo_parsed.apply(lambda x: x[0])
            self.df['Geo_Lon'] = geo_parsed.apply(lambda x: x[1])
            
            # Fill missing Latitude/Longitude from Geolocation
            if 'Latitude' in self.df.columns:
                self.df['Latitude'] = self.df['Latitude'].fillna(self.df['Geo_Lat'])
            if 'Longitude' in self.df.columns:
                self.df['Longitude'] = self.df['Longitude'].fillna(self.df['Geo_Lon'])
            
            # Drop temporary columns
            self.df.drop(['Geo_Lat', 'Geo_Lon'], axis=1, inplace=True)
            self._log("Parsed Geolocation and filled missing coordinates")
        
        return self
    
    def clean_boolean_columns(self) -> 'TrafficDataCleaner':
        """Standardize all boolean columns."""
        for col in BOOLEAN_COLUMNS:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(standardize_boolean)
        
        self._log(f"Standardized {len(BOOLEAN_COLUMNS)} boolean columns")
        return self
    
    def clean_vehicle_data(self) -> 'TrafficDataCleaner':
        """Clean vehicle-related columns."""
        # Clean Year
        if 'Year' in self.df.columns:
            self.df['Year'] = pd.to_numeric(self.df['Year'], errors='coerce')
            
            invalid_mask = (
                (self.df['Year'] < MIN_VEHICLE_YEAR) |
                (self.df['Year'] > MAX_VEHICLE_YEAR)
            )
            invalid_count = invalid_mask.sum()
            self.df.loc[invalid_mask, 'Year'] = np.nan
            self._log(f"Cleaned vehicle Year: {invalid_count:,} invalid values")
        
        # Standardize Make
        if 'Make' in self.df.columns:
            self.df['Make'] = (
                self.df['Make']
                .astype(str)
                .str.strip()
                .str.upper()
                .replace(MAKE_MAPPING)
            )
            self._log("Standardized vehicle Make")
        
        # Standardize Color
        if 'Color' in self.df.columns:
            self.df['Color'] = (
                self.df['Color']
                .astype(str)
                .str.strip()
                .str.upper()
                .replace(COLOR_MAPPING)
            )
            self._log("Standardized vehicle Color")
        
        # Clean VehicleType
        if 'VehicleType' in self.df.columns:
            self.df['VehicleType'] = (
                self.df['VehicleType']
                .astype(str)
                .str.strip()
                .str.upper()
            )
            # Extract vehicle category (after the dash)
            self.df['VehicleCategory'] = (
                self.df['VehicleType']
                .str.extract(r'-\s*(.+)', expand=False)
                .str.strip()
            )
            self._log("Cleaned VehicleType and extracted VehicleCategory")
        
        return self
    
    def clean_demographics(self) -> 'TrafficDataCleaner':
        """Clean demographic columns (Race, Gender)."""
        # Clean Gender
        if 'Gender' in self.df.columns:
            self.df['Gender'] = (
                self.df['Gender']
                .astype(str)
                .str.strip()
                .str.upper()
            )
            # Standardize to M/F/Unknown
            gender_map = {'M': 'M', 'F': 'F', 'MALE': 'M', 'FEMALE': 'F'}
            self.df['Gender'] = self.df['Gender'].map(gender_map).fillna('UNKNOWN')
            self._log("Standardized Gender column")
        
        # Clean Race
        if 'Race' in self.df.columns:
            self.df['Race'] = (
                self.df['Race']
                .astype(str)
                .str.strip()
                .str.upper()
            )
            self._log("Standardized Race column")
        
        return self
    
    def clean_location_data(self) -> 'TrafficDataCleaner':
        """Clean location-related columns."""
        # Clean State columns
        for col in ['State', 'Driver State', 'DL State']:
            if col in self.df.columns:
                self.df[col] = (
                    self.df[col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .replace({'NAN': np.nan, '': np.nan})
                )
        
        # Extract district number from SubAgency
        if 'SubAgency' in self.df.columns:
            self.df['District'] = self.df['SubAgency'].apply(extract_district_number)
            self._log("Extracted District number from SubAgency")
        
        # Clean Agency
        if 'Agency' in self.df.columns:
            self.df['Agency'] = (
                self.df['Agency']
                .astype(str)
                .str.strip()
                .str.upper()
            )
        
        self._log("Cleaned location columns")
        return self
    
    def clean_violation_data(self) -> 'TrafficDataCleaner':
        """Clean violation-related columns."""
        # Clean Violation Type
        if 'Violation Type' in self.df.columns:
            self.df['Violation Type'] = (
                self.df['Violation Type']
                .astype(str)
                .str.strip()
                .str.upper()
            )
        
        # Clean Description
        if 'Description' in self.df.columns:
            self.df['Description'] = (
                self.df['Description']
                .astype(str)
                .str.strip()
                .str.upper()
            )
        
        self._log("Cleaned violation columns")
        return self
    
    def engineer_features(self) -> 'TrafficDataCleaner':
        """Create derived features for analysis."""
        # Time-based features
        if 'DateTime' in self.df.columns:
            self.df['Hour'] = self.df['DateTime'].dt.hour
            self.df['DayOfWeek'] = self.df['DateTime'].dt.dayofweek  # 0=Monday
            self.df['DayName'] = self.df['DateTime'].dt.day_name()
            self.df['Month'] = self.df['DateTime'].dt.month
            self.df['MonthName'] = self.df['DateTime'].dt.month_name()
            self.df['Year_Stop'] = self.df['DateTime'].dt.year
            self.df['IsWeekend'] = self.df['DayOfWeek'].isin([5, 6])
            
            # Time of day buckets
            def get_time_bucket(hour):
                if pd.isna(hour):
                    return 'Unknown'
                hour = int(hour)
                if 5 <= hour < 12:
                    return 'Morning'
                elif 12 <= hour < 17:
                    return 'Afternoon'
                elif 17 <= hour < 21:
                    return 'Evening'
                else:
                    return 'Night'
            
            self.df['TimeBucket'] = self.df['Hour'].apply(get_time_bucket)
            self._log("Created time-based features")
        
        # Severity indicator
        severity_cols = ['Accident', 'Personal Injury', 'Property Damage', 'Fatal']
        existing_severity_cols = [col for col in severity_cols if col in self.df.columns]
        
        if existing_severity_cols:
            self.df['HasSeverity'] = self.df[existing_severity_cols].any(axis=1)
            
            def calculate_severity_score(row):
                score = 0
                if row.get('Accident') == True:
                    score += 1
                if row.get('Property Damage') == True:
                    score += 1
                if row.get('Personal Injury') == True:
                    score += 2
                if row.get('Fatal') == True:
                    score += 4
                return score
            
            self.df['SeverityScore'] = self.df.apply(calculate_severity_score, axis=1)
            self._log("Created severity indicators")
        
        # Vehicle age
        if 'Year' in self.df.columns and 'Year_Stop' in self.df.columns:
            self.df['VehicleAge'] = self.df['Year_Stop'] - self.df['Year']
            # Clean invalid ages
            self.df.loc[self.df['VehicleAge'] < 0, 'VehicleAge'] = np.nan
            self.df.loc[self.df['VehicleAge'] > 60, 'VehicleAge'] = np.nan
            self._log("Created VehicleAge feature")
        
        return self
    
    def optimize_dtypes(self) -> 'TrafficDataCleaner':
        """Optimize data types for memory efficiency."""
        initial_memory = self.df.memory_usage(deep=True).sum() / 1024**2
        
        # Convert object columns with few unique values to category
        for col in self.df.select_dtypes(include=['object']).columns:
            num_unique = self.df[col].nunique()
            num_total = len(self.df)
            
            if num_unique / num_total < 0.5:  # Less than 50% unique values
                self.df[col] = self.df[col].astype('category')
        
        # Downcast numeric columns
        for col in self.df.select_dtypes(include=['int64']).columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='integer')
        
        for col in self.df.select_dtypes(include=['float64']).columns:
            self.df[col] = pd.to_numeric(self.df[col], downcast='float')
        
        final_memory = self.df.memory_usage(deep=True).sum() / 1024**2
        reduction = (1 - final_memory / initial_memory) * 100
        
        self._log(f"Optimized dtypes: {initial_memory:.2f}MB → {final_memory:.2f}MB ({reduction:.1f}% reduction)")
        return self
    
    def clean_all(self) -> pd.DataFrame:
        """Run the complete cleaning pipeline."""
        self._log("Starting complete cleaning pipeline...")
        
        (self
         .remove_duplicates()
         .clean_datetime()
         .clean_coordinates()
         .clean_boolean_columns()
         .clean_vehicle_data()
         .clean_demographics()
         .clean_location_data()
         .clean_violation_data()
         .engineer_features()
         .optimize_dtypes())
        
        self._log(f"Cleaning complete. Final dataset: {len(self.df):,} rows")
        return self.df


def main():
    """Main function to run data cleaning."""
    from utils import load_data, save_data
    import sys
    sys.path.append('..')
    from config import RAW_DATA_PATH, CLEANED_DATA_PATH
    
    # Load data
    df = load_data(RAW_DATA_PATH)
    
    # Clean data
    cleaner = TrafficDataCleaner(df)
    cleaned_df = cleaner.clean_all()
    
    # Save cleaned data
    save_data(cleaned_df, CLEANED_DATA_PATH)
    
    # Save cleaning report
    report_path = CLEANED_DATA_PATH.replace('.csv', '_cleaning_report.txt')
    with open(report_path, 'w') as f:
        f.write(cleaner.get_cleaning_report())
    
    print(f"\nCleaning report saved to: {report_path}")


if __name__ == "__main__":
    main()
