"""
Traffic Violations Insight System
Interactive Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Traffic Violations Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #333;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the cleaned dataset."""
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'cleaned', 'traffic_violations_cleaned.csv'
    )
    
    df = pd.read_csv(data_path, low_memory=False)
    
    # Convert date columns
    if 'Date Of Stop' in df.columns:
        df['Date Of Stop'] = pd.to_datetime(df['Date Of Stop'], errors='coerce')
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    
    return df


def create_metric_card(value, label, delta=None):
    """Create a styled metric card."""
    delta_html = ""
    if delta is not None:
        color = "green" if delta >= 0 else "red"
        delta_html = f'<span style="color: {color}; font-size: 0.9rem;">({delta:+.1f}%)</span>'
    
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value:,}</div>
        <div class="metric-label">{label} {delta_html}</div>
    </div>
    """


def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 Traffic Violations Insight System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("Data file not found. Please run the data cleaning pipeline first.")
        st.info("Run: `python src/data_cleaning.py`")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Date filter
    if 'Date Of Stop' in df.columns:
        date_col = df['Date Of Stop'].dropna()
        if len(date_col) > 0:
            min_date = date_col.min().date()
            max_date = date_col.max().date()
            
            date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                df = df[
                    (df['Date Of Stop'].dt.date >= date_range[0]) &
                    (df['Date Of Stop'].dt.date <= date_range[1])
                ]
    
    # Violation Type filter
    if 'Violation Type' in df.columns:
        violation_types = ['All'] + sorted(df['Violation Type'].dropna().unique().tolist())
        selected_violation_type = st.sidebar.selectbox(
            "Violation Type",
            violation_types
        )
        if selected_violation_type != 'All':
            df = df[df['Violation Type'] == selected_violation_type]
    
    # Gender filter
    if 'Gender' in df.columns:
        genders = ['All'] + sorted(df['Gender'].dropna().unique().tolist())
        selected_gender = st.sidebar.selectbox("Gender", genders)
        if selected_gender != 'All':
            df = df[df['Gender'] == selected_gender]
    
    # Race filter
    if 'Race' in df.columns:
        races = ['All'] + sorted(df['Race'].dropna().unique().tolist())
        selected_race = st.sidebar.selectbox("Race", races)
        if selected_race != 'All':
            df = df[df['Race'] == selected_race]
    
    # Vehicle Make filter
    if 'Make' in df.columns:
        top_makes = df['Make'].value_counts().head(20).index.tolist()
        makes = ['All'] + sorted(top_makes)
        selected_make = st.sidebar.selectbox("Vehicle Make (Top 20)", makes)
        if selected_make != 'All':
            df = df[df['Make'] == selected_make]
    
    # Time of Day filter
    if 'TimeBucket' in df.columns:
        time_buckets = ['All'] + ['Morning', 'Afternoon', 'Evening', 'Night']
        selected_time = st.sidebar.selectbox("Time of Day", time_buckets)
        if selected_time != 'All':
            df = df[df['TimeBucket'] == selected_time]
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Filtered Records:** {len(df):,}")
    
    # Main content
    # Key Metrics Row
    st.markdown('<h2 class="section-header">📊 Key Metrics</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Violations", f"{len(df):,}")
    
    with col2:
        accidents = df['Accident'].sum() if 'Accident' in df.columns else 0
        st.metric("Accidents", f"{int(accidents):,}")
    
    with col3:
        injuries = df['Personal Injury'].sum() if 'Personal Injury' in df.columns else 0
        st.metric("Personal Injuries", f"{int(injuries):,}")
    
    with col4:
        fatalities = df['Fatal'].sum() if 'Fatal' in df.columns else 0
        st.metric("Fatalities", f"{int(fatalities):,}")
    
    with col5:
        unique_locations = df['Location'].nunique() if 'Location' in df.columns else 0
        st.metric("Unique Locations", f"{unique_locations:,}")
    
    st.markdown("---")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Trends", "🗺️ Geographic", "👥 Demographics", "🚙 Vehicles", "⚠️ Severity"
    ])
    
    # Tab 1: Trends
    with tab1:
        st.markdown("### Violation Trends Over Time")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily trend
            if 'Date Of Stop' in df.columns:
                daily_counts = df.groupby(df['Date Of Stop'].dt.date).size().reset_index()
                daily_counts.columns = ['Date', 'Count']
                
                fig = px.line(
                    daily_counts, x='Date', y='Count',
                    title='Daily Violations Trend',
                    labels={'Count': 'Number of Violations'}
                )
                fig.update_traces(line_color='#1f77b4')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Hourly distribution
            if 'Hour' in df.columns:
                hourly = df['Hour'].value_counts().sort_index().reset_index()
                hourly.columns = ['Hour', 'Count']
                
                fig = px.bar(
                    hourly, x='Hour', y='Count',
                    title='Violations by Hour of Day',
                    labels={'Count': 'Number of Violations'}
                )
                fig.update_traces(marker_color='#2ecc71')
                st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Day of week
            if 'DayName' in df.columns:
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily = df['DayName'].value_counts().reindex(day_order).reset_index()
                daily.columns = ['Day', 'Count']
                
                fig = px.bar(
                    daily, x='Day', y='Count',
                    title='Violations by Day of Week',
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Top violations
            if 'Description' in df.columns:
                top_violations = df['Description'].value_counts().head(10).reset_index()
                top_violations.columns = ['Violation', 'Count']
                
                fig = px.bar(
                    top_violations, x='Count', y='Violation',
                    orientation='h',
                    title='Top 10 Violations',
                    color='Count',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Geographic
    with tab2:
        st.markdown("### Geographic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top locations
            if 'Location' in df.columns:
                top_locations = df['Location'].value_counts().head(15).reset_index()
                top_locations.columns = ['Location', 'Count']
                
                fig = px.bar(
                    top_locations, x='Count', y='Location',
                    orientation='h',
                    title='Top 15 Violation Hotspots',
                    color='Count',
                    color_continuous_scale='YlOrRd'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # District distribution
            if 'District' in df.columns:
                district_counts = df['District'].value_counts().reset_index()
                district_counts.columns = ['District', 'Count']
                
                fig = px.pie(
                    district_counts, values='Count', names='District',
                    title='Violations by District',
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Map visualization (if coordinates are valid)
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            valid_coords = df.dropna(subset=['Latitude', 'Longitude'])
            
            if len(valid_coords) > 0:
                st.markdown("### Violation Map")
                
                # Sample for performance
                sample_size = min(10000, len(valid_coords))
                map_data = valid_coords.sample(sample_size) if len(valid_coords) > sample_size else valid_coords
                
                fig = px.scatter_mapbox(
                    map_data,
                    lat='Latitude',
                    lon='Longitude',
                    color='Accident' if 'Accident' in map_data.columns else None,
                    size_max=10,
                    zoom=9,
                    mapbox_style='carto-positron',
                    title=f'Violation Locations (Sample of {sample_size:,})'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Demographics
    with tab3:
        st.markdown("### Demographic Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender distribution
            if 'Gender' in df.columns:
                gender_counts = df['Gender'].value_counts().reset_index()
                gender_counts.columns = ['Gender', 'Count']
                
                fig = px.pie(
                    gender_counts, values='Count', names='Gender',
                    title='Violations by Gender',
                    color_discrete_sequence=['#3498db', '#e74c3c', '#95a5a6']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Race distribution
            if 'Race' in df.columns:
                race_counts = df['Race'].value_counts().head(10).reset_index()
                race_counts.columns = ['Race', 'Count']
                
                fig = px.bar(
                    race_counts, x='Race', y='Count',
                    title='Violations by Race (Top 10)',
                    color='Count',
                    color_continuous_scale='Blues'
                )
                plt.xticks(rotation=45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Gender vs Violation Type
        if 'Gender' in df.columns and 'Violation Type' in df.columns:
            cross_tab = pd.crosstab(df['Gender'], df['Violation Type'])
            
            fig = px.bar(
                cross_tab.reset_index().melt(id_vars='Gender'),
                x='Gender', y='value', color='Violation Type',
                title='Violation Type by Gender',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Vehicles
    with tab4:
        st.markdown("### Vehicle Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top makes
            if 'Make' in df.columns:
                make_counts = df['Make'].value_counts().head(15).reset_index()
                make_counts.columns = ['Make', 'Count']
                
                fig = px.bar(
                    make_counts, x='Count', y='Make',
                    orientation='h',
                    title='Top 15 Vehicle Makes',
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Vehicle colors
            if 'Color' in df.columns:
                color_counts = df['Color'].value_counts().head(10).reset_index()
                color_counts.columns = ['Color', 'Count']
                
                fig = px.pie(
                    color_counts, values='Count', names='Color',
                    title='Top 10 Vehicle Colors',
                    hole=0.3
                )
                st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Vehicle type
            if 'VehicleCategory' in df.columns:
                vtype_counts = df['VehicleCategory'].value_counts().head(10).reset_index()
                vtype_counts.columns = ['Type', 'Count']
                
                fig = px.bar(
                    vtype_counts, x='Type', y='Count',
                    title='Violations by Vehicle Type',
                    color='Count',
                    color_continuous_scale='Oranges'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Vehicle age distribution
            if 'VehicleAge' in df.columns:
                valid_ages = df['VehicleAge'].dropna()
                
                fig = px.histogram(
                    valid_ages, nbins=30,
                    title='Vehicle Age Distribution',
                    labels={'value': 'Vehicle Age (years)', 'count': 'Frequency'}
                )
                fig.add_vline(x=valid_ages.mean(), line_dash="dash", line_color="red",
                             annotation_text=f"Mean: {valid_ages.mean():.1f} years")
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 5: Severity
    with tab5:
        st.markdown("### Severity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Severity overview
            severity_cols = ['Accident', 'Personal Injury', 'Property Damage', 'Fatal']
            severity_data = []
            
            for col in severity_cols:
                if col in df.columns:
                    count = df[col].sum()
                    severity_data.append({'Category': col, 'Count': int(count)})
            
            if severity_data:
                severity_df = pd.DataFrame(severity_data)
                
                fig = px.bar(
                    severity_df, x='Category', y='Count',
                    title='Severity Indicators Overview',
                    color='Count',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Accident rate by time
            if 'TimeBucket' in df.columns and 'Accident' in df.columns:
                time_severity = df.groupby('TimeBucket')['Accident'].agg(['sum', 'count'])
                time_severity['rate'] = time_severity['sum'] / time_severity['count'] * 100
                time_severity = time_severity.reset_index()
                
                bucket_order = ['Morning', 'Afternoon', 'Evening', 'Night']
                time_severity['TimeBucket'] = pd.Categorical(
                    time_severity['TimeBucket'], categories=bucket_order, ordered=True
                )
                time_severity = time_severity.sort_values('TimeBucket')
                
                fig = px.bar(
                    time_severity, x='TimeBucket', y='rate',
                    title='Accident Rate by Time of Day',
                    labels={'rate': 'Accident Rate (%)'},
                    color='rate',
                    color_continuous_scale='YlOrRd'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Severity score distribution
        if 'SeverityScore' in df.columns:
            st.markdown("### Severity Score Distribution")
            
            score_counts = df['SeverityScore'].value_counts().sort_index().reset_index()
            score_counts.columns = ['Score', 'Count']
            
            fig = px.bar(
                score_counts, x='Score', y='Count',
                title='Distribution of Severity Scores',
                color='Score',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Alcohol involvement
        if 'Alcohol' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                alcohol_counts = df['Alcohol'].value_counts()
                
                fig = px.pie(
                    values=alcohol_counts.values,
                    names=['No Alcohol', 'Alcohol Involved'] if False in alcohol_counts.index else alcohol_counts.index,
                    title='Alcohol Involvement',
                    color_discrete_sequence=['#2ecc71', '#e74c3c']
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Traffic Violations Insight System | Built with Streamlit</p>
            <p>Data refreshed on load | Filters apply across all visualizations</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
