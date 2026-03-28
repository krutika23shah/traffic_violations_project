"""Exploratory Data Analysis module."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TrafficViolationsEDA:
    """
    Exploratory Data Analysis for Traffic Violations dataset.
    
    Provides methods to analyze:
    - Violation patterns
    - Temporal trends
    - Geographic distribution
    - Demographic correlations
    - Vehicle characteristics
    - Severity analysis
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize EDA with cleaned DataFrame.
        
        Args:
            df: Cleaned traffic violations DataFrame
        """
        self.df = df
        self.insights = []
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def _add_insight(self, category: str, insight: str) -> None:
        """Add an insight to the collection."""
        self.insights.append({'category': category, 'insight': insight})
    
    def get_summary_statistics(self) -> Dict:
        """Generate overall summary statistics."""
        summary = {
            'total_records': len(self.df),
            'date_range': None,
            'unique_violations': None,
            'unique_locations': None,
        }
        
        if 'Date Of Stop' in self.df.columns:
            valid_dates = self.df['Date Of Stop'].dropna()
            if len(valid_dates) > 0:
                summary['date_range'] = {
                    'start': valid_dates.min().strftime('%Y-%m-%d'),
                    'end': valid_dates.max().strftime('%Y-%m-%d')
                }
        
        if 'Description' in self.df.columns:
            summary['unique_violations'] = self.df['Description'].nunique()
        
        if 'Location' in self.df.columns:
            summary['unique_locations'] = self.df['Location'].nunique()
        
        # Count key metrics
        for col in ['Accident', 'Personal Injury', 'Fatal']:
            if col in self.df.columns:
                summary[f'total_{col.lower().replace(" ", "_")}'] = self.df[col].sum()
        
        return summary
    
    def analyze_top_violations(self, top_n: int = 15) -> Tuple[pd.DataFrame, plt.Figure]:
        """Analyze most common violations."""
        if 'Description' not in self.df.columns:
            return None, None
        
        violation_counts = (
            self.df['Description']
            .value_counts()
            .head(top_n)
            .reset_index()
        )
        violation_counts.columns = ['Violation', 'Count']
        violation_counts['Percentage'] = (
            violation_counts['Count'] / len(self.df) * 100
        ).round(2)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(
            violation_counts['Violation'][::-1],
            violation_counts['Count'][::-1],
            color=sns.color_palette("viridis", top_n)
        )
        
        ax.set_xlabel('Number of Violations', fontsize=12)
        ax.set_title(f'Top {top_n} Most Common Violations', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, count in zip(bars, violation_counts['Count'][::-1]):
            ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                   f'{count:,}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        # Add insight
        top_violation = violation_counts.iloc[0]
        self._add_insight(
            'Violations',
            f"The most common violation is '{top_violation['Violation']}' "
            f"with {top_violation['Count']:,} occurrences ({top_violation['Percentage']}%)"
        )
        
        return violation_counts, fig
    
    def analyze_temporal_patterns(self) -> Dict[str, Tuple[pd.DataFrame, plt.Figure]]:
        """Analyze violations by time dimensions."""
        results = {}
        
        # Hourly distribution
        if 'Hour' in self.df.columns:
            hourly = self.df['Hour'].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(hourly.index, hourly.values, color='steelblue', edgecolor='white')
            ax.set_xlabel('Hour of Day', fontsize=12)
            ax.set_ylabel('Number of Violations', fontsize=12)
            ax.set_title('Violations by Hour of Day', fontsize=14, fontweight='bold')
            ax.set_xticks(range(24))
            plt.tight_layout()
            
            results['hourly'] = (hourly.reset_index(), fig)
            
            peak_hour = hourly.idxmax()
            self._add_insight(
                'Temporal',
                f"Peak violation hour is {peak_hour}:00 with {hourly[peak_hour]:,} violations"
            )
        
        # Day of week distribution
        if 'DayName' in self.df.columns:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily = (
                self.df['DayName']
                .value_counts()
                .reindex(day_order)
            )
            
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#2ecc71' if day in ['Saturday', 'Sunday'] else '#3498db' for day in day_order]
            ax.bar(daily.index, daily.values, color=colors, edgecolor='white')
            ax.set_xlabel('Day of Week', fontsize=12)
            ax.set_ylabel('Number of Violations', fontsize=12)
            ax.set_title('Violations by Day of Week', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            results['daily'] = (daily.reset_index(), fig)
            
            peak_day = daily.idxmax()
            self._add_insight(
                'Temporal',
                f"Most violations occur on {peak_day} with {daily[peak_day]:,} incidents"
            )
        
        # Monthly distribution
        if 'MonthName' in self.df.columns:
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            monthly = (
                self.df['MonthName']
                .value_counts()
                .reindex(month_order)
                .dropna()
            )
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(range(len(monthly)), monthly.values, marker='o', linewidth=2, markersize=8)
            ax.fill_between(range(len(monthly)), monthly.values, alpha=0.3)
            ax.set_xticks(range(len(monthly)))
            ax.set_xticklabels(monthly.index, rotation=45)
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Number of Violations', fontsize=12)
            ax.set_title('Violations by Month', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            results['monthly'] = (monthly.reset_index(), fig)
        
        # Time bucket distribution
        if 'TimeBucket' in self.df.columns:
            bucket_order = ['Morning', 'Afternoon', 'Evening', 'Night']
            buckets = (
                self.df['TimeBucket']
                .value_counts()
                .reindex(bucket_order)
                .dropna()
            )
            
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = ['#f39c12', '#e74c3c', '#9b59b6', '#34495e']
            ax.pie(buckets.values, labels=buckets.index, autopct='%1.1f%%',
                  colors=colors, startangle=90, explode=[0.02]*len(buckets))
            ax.set_title('Violations by Time of Day', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            results['time_bucket'] = (buckets.reset_index(), fig)
        
        return results
    
    def analyze_demographics(self) -> Dict[str, Tuple[pd.DataFrame, plt.Figure]]:
        """Analyze violations by demographic factors."""
        results = {}
        
        # Gender distribution
        if 'Gender' in self.df.columns:
            gender_counts = self.df['Gender'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = {'M': '#3498db', 'F': '#e74c3c', 'UNKNOWN': '#95a5a6'}
            ax.bar(gender_counts.index, gender_counts.values,
                  color=[colors.get(g, '#95a5a6') for g in gender_counts.index])
            ax.set_xlabel('Gender', fontsize=12)
            ax.set_ylabel('Number of Violations', fontsize=12)
            ax.set_title('Violations by Gender', fontsize=14, fontweight='bold')
            
            for i, (gender, count) in enumerate(gender_counts.items()):
                ax.text(i, count + 500, f'{count:,}', ha='center', fontsize=10)
            
            plt.tight_layout()
            results['gender'] = (gender_counts.reset_index(), fig)
            
            if 'M' in gender_counts.index and 'F' in gender_counts.index:
                ratio = gender_counts['M'] / gender_counts['F']
                self._add_insight(
                    'Demographics',
                    f"Male to female violation ratio is {ratio:.2f}:1"
                )
        
        # Race distribution
        if 'Race' in self.df.columns:
            race_counts = self.df['Race'].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(race_counts.index[::-1], race_counts.values[::-1],
                   color=sns.color_palette("coolwarm", len(race_counts)))
            ax.set_xlabel('Number of Violations', fontsize=12)
            ax.set_title('Violations by Race', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            results['race'] = (race_counts.reset_index(), fig)
        
        # Gender vs Violation Type
        if 'Gender' in self.df.columns and 'Violation Type' in self.df.columns:
            gender_violation = pd.crosstab(
                self.df['Gender'],
                self.df['Violation Type'],
                normalize='index'
            ) * 100
            
            fig, ax = plt.subplots(figsize=(10, 6))
            gender_violation.plot(kind='bar', ax=ax, edgecolor='white')
            ax.set_xlabel('Gender', fontsize=12)
            ax.set_ylabel('Percentage', fontsize=12)
            ax.set_title('Violation Type Distribution by Gender', fontsize=14, fontweight='bold')
            ax.legend(title='Violation Type', bbox_to_anchor=(1.02, 1))
            plt.xticks(rotation=0)
            plt.tight_layout()
            
            results['gender_violation_type'] = (gender_violation.reset_index(), fig)
        
        return results
    
    def analyze_vehicles(self) -> Dict[str, Tuple[pd.DataFrame, plt.Figure]]:
        """Analyze vehicle-related patterns."""
        results = {}
        
        # Top vehicle makes
        if 'Make' in self.df.columns:
            make_counts = (
                self.df['Make']
                .replace('NAN', np.nan)
                .dropna()
                .value_counts()
                .head(15)
            )
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(make_counts.index, make_counts.values,
                  color=sns.color_palette("viridis", len(make_counts)))
            ax.set_xlabel('Vehicle Make', fontsize=12)
            ax.set_ylabel('Number of Violations', fontsize=12)
            ax.set_title('Top 15 Vehicle Makes in Violations', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            results['make'] = (make_counts.reset_index(), fig)
            
            top_make = make_counts.index[0]
            self._add_insight(
                'Vehicles',
                f"'{top_make}' is the most common vehicle make with {make_counts.iloc[0]:,} violations"
            )
        
        # Vehicle type distribution
        if 'VehicleCategory' in self.df.columns:
            vtype_counts = (
                self.df['VehicleCategory']
                .dropna()
                .value_counts()
                .head(10)
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(vtype_counts.index[::-1], vtype_counts.values[::-1],
                   color=sns.color_palette("Spectral", len(vtype_counts)))
            ax.set_xlabel('Number of Violations', fontsize=12)
            ax.set_title('Violations by Vehicle Type', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            results['vehicle_type'] = (vtype_counts.reset_index(), fig)
        
        # Vehicle age distribution
        if 'VehicleAge' in self.df.columns:
            vehicle_age = self.df['VehicleAge'].dropna()
            
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.hist(vehicle_age, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
            ax.axvline(vehicle_age.mean(), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {vehicle_age.mean():.1f} years')
            ax.axvline(vehicle_age.median(), color='green', linestyle='--', linewidth=2,
                      label=f'Median: {vehicle_age.median():.1f} years')
            ax.set_xlabel('Vehicle Age (years)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Distribution of Vehicle Age', fontsize=14, fontweight='bold')
            ax.legend()
            plt.tight_layout()
            
            results['vehicle_age'] = (vehicle_age.describe().reset_index(), fig)
            
            self._add_insight(
                'Vehicles',
                f"Average vehicle age at time of violation: {vehicle_age.mean():.1f} years"
            )
        
        # Top colors
        if 'Color' in self.df.columns:
            color_counts = (
                self.df['Color']
                .replace('NAN', np.nan)
                .dropna()
                .value_counts()
                .head(10)
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(color_counts.index, color_counts.values,
                  color=['black', 'white', 'silver', 'gray', 'blue',
                        'red', 'green', 'gold', 'tan', 'maroon'][:len(color_counts)],
                  edgecolor='black')
            ax.set_xlabel('Vehicle Color', fontsize=12)
            ax.set_ylabel('Number of Violations', fontsize=12)
            ax.set_title('Top 10 Vehicle Colors', fontsize=14, fontweight='bold')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            results['color'] = (color_counts.reset_index(), fig)
        
        return results
    
    def analyze_severity(self) -> Dict[str, Tuple[pd.DataFrame, plt.Figure]]:
        """Analyze accident and injury patterns."""
        results = {}
        
        # Overall severity counts
        severity_cols = ['Accident', 'Personal Injury', 'Property Damage', 'Fatal']
        existing_cols = [col for col in severity_cols if col in self.df.columns]
        
        if existing_cols:
            severity_counts = {
                col: self.df[col].sum() for col in existing_cols
            }
            
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6']
            ax.bar(severity_counts.keys(), severity_counts.values(),
                  color=colors[:len(severity_counts)])
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Severity Indicators Overview', fontsize=14, fontweight='bold')
            
            for i, (key, val) in enumerate(severity_counts.items()):
                ax.text(i, val + 100, f'{val:,}', ha='center', fontsize=10)
            
            plt.tight_layout()
            results['severity_overview'] = (pd.DataFrame(severity_counts, index=[0]), fig)
            
            if 'Accident' in severity_counts:
                accident_rate = severity_counts['Accident'] / len(self.df) * 100
                self._add_insight(
                    'Severity',
                    f"Accident rate: {accident_rate:.2f}% of all violations involve accidents"
                )
        
        # Severity by time of day
        if 'TimeBucket' in self.df.columns and 'Accident' in self.df.columns:
            severity_time = (
                self.df.groupby('TimeBucket')['Accident']
                .agg(['sum', 'count'])
            )
            severity_time['rate'] = severity_time['sum'] / severity_time['count'] * 100
            
            bucket_order = ['Morning', 'Afternoon', 'Evening', 'Night']
            severity_time = severity_time.reindex(bucket_order)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(severity_time.index, severity_time['rate'],
                  color=['#f39c12', '#e74c3c', '#9b59b6', '#34495e'])
            ax.set_xlabel('Time of Day', fontsize=12)
            ax.set_ylabel('Accident Rate (%)', fontsize=12)
            ax.set_title('Accident Rate by Time of Day', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            results['severity_by_time'] = (severity_time.reset_index(), fig)
            
            highest_risk = severity_time['rate'].idxmax()
            self._add_insight(
                'Severity',
                f"Highest accident risk during {highest_risk} ({severity_time.loc[highest_risk, 'rate']:.2f}%)"
            )
        
        return results
    
    def analyze_geographic_hotspots(self, top_n: int = 20) -> Tuple[pd.DataFrame, plt.Figure]:
        """Identify high-violation locations."""
        if 'Location' not in self.df.columns:
            return None, None
        
        location_counts = (
            self.df['Location']
            .value_counts()
            .head(top_n)
            .reset_index()
        )
        location_counts.columns = ['Location', 'Count']
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.barh(location_counts['Location'][::-1], location_counts['Count'][::-1],
               color=sns.color_palette("YlOrRd", top_n)[::-1])
        ax.set_xlabel('Number of Violations', fontsize=12)
        ax.set_title(f'Top {top_n} Violation Hotspots', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        top_location = location_counts.iloc[0]
        self._add_insight(
            'Geographic',
            f"Top hotspot: '{top_location['Location']}' with {top_location['Count']:,} violations"
        )
        
        return location_counts, fig
    
    def analyze_violation_outcomes(self) -> Dict[str, Tuple[pd.DataFrame, plt.Figure]]:
        """Analyze violation outcomes (citation, warning, etc.)."""
        results = {}
        
        if 'Violation Type' in self.df.columns:
            outcome_counts = self.df['Violation Type'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = sns.color_palette("Set2", len(outcome_counts))
            wedges, texts, autotexts = ax.pie(
                outcome_counts.values,
                labels=outcome_counts.index,
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                explode=[0.02] * len(outcome_counts)
            )
            ax.set_title('Violation Outcome Distribution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            results['outcomes'] = (outcome_counts.reset_index(), fig)
            
            most_common = outcome_counts.index[0]
            percentage = outcome_counts.iloc[0] / outcome_counts.sum() * 100
            self._add_insight(
                'Outcomes',
                f"Most common outcome: {most_common} ({percentage:.1f}%)"
            )
        
        return results
    
    def generate_correlation_analysis(self) -> Tuple[pd.DataFrame, plt.Figure]:
        """Analyze correlations between numeric/boolean variables."""
        # Select relevant columns
        corr_cols = []
        
        for col in self.df.columns:
            if self.df[col].dtype in ['bool', 'int64', 'float64', 'int32', 'float32']:
                if self.df[col].notna().sum() > 0:
                    corr_cols.append(col)
        
        if len(corr_cols) < 2:
            return None, None
        
        # Convert boolean to int for correlation
        corr_df = self.df[corr_cols].copy()
        for col in corr_df.columns:
            if corr_df[col].dtype == 'bool':
                corr_df[col] = corr_df[col].astype(int)
        
        correlation_matrix = corr_df.corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            ax=ax,
            square=True,
            linewidths=0.5
        )
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return correlation_matrix, fig
    
    def get_all_insights(self) -> pd.DataFrame:
        """Return all collected insights."""
        return pd.DataFrame(self.insights)
    
    def generate_full_report(self, save_path: Optional[str] = None) -> str:
        """Generate a complete EDA report."""
        report = []
        report.append("=" * 80)
        report.append("TRAFFIC VIOLATIONS EXPLORATORY DATA ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        summary = self.get_summary_statistics()
        report.append("## SUMMARY STATISTICS")
        report.append("-" * 40)
        for key, value in summary.items():
            report.append(f"  {key}: {value}")
        report.append("")
        
        # Run all analyses
        self.analyze_top_violations()
        self.analyze_temporal_patterns()
        self.analyze_demographics()
        self.analyze_vehicles()
        self.analyze_severity()
        self.analyze_geographic_hotspots()
        self.analyze_violation_outcomes()
        
        # Insights
        report.append("## KEY INSIGHTS")
        report.append("-" * 40)
        insights_df = self.get_all_insights()
        for _, row in insights_df.iterrows():
            report.append(f"  [{row['category']}] {row['insight']}")
        report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        full_report = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(full_report)
            print(f"Report saved to: {save_path}")
        
        return full_report
