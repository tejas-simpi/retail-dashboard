import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import chardet
import os
from datetime import datetime

# Detect file encoding
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        rawdata = f.read(10000)
    return chardet.detect(rawdata)['encoding']

# Data Loading with Error Handling
@st.cache_data
def load_data(uploaded_file):
    try:
        # Save uploaded file to temporary location
        with open("temp_file.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Detect encoding
        encoding = detect_encoding("temp_file.csv")
        st.info(f"Detected file encoding: {encoding}")
        
        # Load CSV with detected encoding
        df = pd.read_csv("temp_file.csv", encoding=encoding)
        
        # Clean up temporary file
        os.remove("temp_file.csv")
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Normalize column names
def normalize_columns(df):
    # Create a mapping of common column name variations
    column_mapping = {
        'sales': 'Sales',
        'profit': 'Profit',
        'order_date': 'Order_Date',
        'order date': 'Order_Date',
        'category': 'Category',
        'region': 'Region',
        'state': 'State',
        'segment': 'Segment',
        'postal_code': 'Postal_Code',
        'postal code': 'Postal_Code',
        'product_id': 'Product_ID'
    }
    
    # Normalize existing columns
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # Rename columns to standard names
    for col in df.columns:
        if col in column_mapping:
            df.rename(columns={col: column_mapping[col]}, inplace=True)
    
    return df

# Data Cleaning Pipeline
def clean_data(df):
    # Normalize column names
    df = normalize_columns(df)
    
    # Handle missing values
    if 'Postal_Code' in df.columns:
        df['Postal_Code'] = df['Postal_Code'].fillna('Unknown').astype(str)
    
    # Convert dates
    if 'Order_Date' in df.columns:
        df['Order_Date'] = pd.to_datetime(df['Order_Date'], errors='coerce')
    
    # Remove rows with missing critical data
    critical_cols = ['Sales', 'Profit', 'Category', 'Region']
    df = df.dropna(subset=[col for col in critical_cols if col in df.columns])
    
    # Remove duplicates
    df = df.drop_duplicates(keep='first')
    
    # Feature engineering
    if 'Order_Date' in df.columns:
        df['Order_Month'] = df['Order_Date'].dt.to_period('M').astype(str)
        df['Order_Year'] = df['Order_Date'].dt.year
    else:
        st.warning("No date column found. Trend analysis will be limited.")
        df['Order_Month'] = 'Unknown'
        df['Order_Year'] = 'Unknown'
    
    # Calculate profit margin safely
    if 'Sales' in df.columns and 'Profit' in df.columns:
        df['Profit_Margin'] = np.where(
            df['Sales'] != 0,
            df['Profit'] / df['Sales'],
            0
        )
        df['Profit_Margin'] = df['Profit_Margin'].clip(lower=-1, upper=1)
    else:
        st.error("Missing Sales or Profit columns. Key metrics unavailable.")
    
    return df

# Visualization Functions
def sales_profit_barchart(df):
    if 'Category' not in df.columns:
        return px.bar(title="Missing Category Column")
    
    cat_df = df.groupby('Category', as_index=False).agg(
        Total_Sales=('Sales', 'sum'),
        Total_Profit=('Profit', 'sum'),
        Avg_Margin=('Profit_Margin', 'mean')
    ).sort_values('Total_Sales', ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cat_df['Category'],
        y=cat_df['Total_Sales'],
        name='Sales',
        marker_color='#1f77b4'
    ))
    fig.add_trace(go.Scatter(
        x=cat_df['Category'],
        y=cat_df['Total_Profit'],
        name='Profit',
        mode='lines+markers',
        marker=dict(color='#ff7f0e', size=10),
        yaxis='y2'
    ))
    fig.update_layout(
        title='Sales & Profit by Category',
        yaxis=dict(title='Sales ($)'),
        yaxis2=dict(title='Profit ($)', overlaying='y', side='right'),
        template='plotly_white'
    )
    return fig

def regional_sales_map(df):
    if 'State' not in df.columns:
        return px.bar(title="Missing State Column")
    
    state_df = df.groupby('State', as_index=False)['Sales'].sum()
    fig = px.choropleth(state_df,
                        locations='State',
                        locationmode='USA-states',
                        color='Sales',
                        scope='usa',
                        color_continuous_scale='Blues',
                        title='Sales by State')
    fig.update_layout(geo=dict(bgcolor='rgba(0,0,0,0)'))
    return fig

def monthly_trend_chart(df):
    if 'Order_Month' not in df.columns or 'Sales' not in df.columns:
        return px.bar(title="Missing Date or Sales Column")
    
    trend_df = df.groupby('Order_Month', as_index=False).agg(
        Sales=('Sales', 'sum'),
        Profit=('Profit', 'sum')
    )
    fig = px.line(trend_df, 
                  x='Order_Month', 
                  y='Sales',
                  markers=True,
                  title='Monthly Sales Trend')
    fig.add_trace(go.Scatter(
        x=trend_df['Order_Month'],
        y=trend_df['Profit'],
        name='Profit',
        line=dict(color='green', dash='dot')
    ))
    return fig

def segment_analysis(df):
    if 'Segment' not in df.columns:
        return px.bar(title="Missing Segment Column")
    
    seg_df = df.groupby('Segment', as_index=False).agg(
        Sales=('Sales', 'sum')
    )
    fig = px.pie(seg_df, 
                 values='Sales', 
                 names='Segment',
                 hole=0.4,
                 title='Sales Distribution by Segment')
    fig.update_traces(textinfo='percent+label')
    return fig

# Main Dashboard
def main():
    st.set_page_config(
        page_title="Retail Analytics Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Dashboard Header
    st.title("📈 Retail Performance Dashboard")
    
    # File Upload Section
    st.sidebar.header("Upload Your Dataset")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file", 
        type=["csv"],
        help="Upload retail data in CSV format"
    )
    
    # Dataset Requirements Section
    with st.sidebar.expander("Dataset Requirements"):
        st.markdown("""
        **Required Columns:**
        - `Sales` (numeric): Transaction amount
        - `Profit` (numeric): Profit from transaction
        - `Order Date` (date): Date of order
        - `Category` (text): Product category
        - `Region` (text): Geographic region
        - `State` (text): For US map visualization
        
        **Optional Columns:**
        - `Segment`: Customer segment
        - `Postal Code`
        - `Product ID`
        
        **Example Structure:**
        | Order Date | Sales | Profit | Category | Region | State |
        |------------|-------|--------|----------|--------|-------|
        | 2023-01-01 | 100.0 | 20.0   | Furniture| West   | CA    |
        """)
    
    # Load sample data if no file uploaded
    if uploaded_file is None:
        st.info("👈 Please upload a CSV file using the sidebar")
        st.markdown("### Don't have a dataset? Try the sample Superstore data:")
        if st.button("Load Sample Dataset"):
            try:
                # Load sample dataset from URL
                sample_url = "https://raw.githubusercontent.com/plotly/datasets/master/superstore.csv"
                df = pd.read_csv(sample_url)
                st.session_state.df = df
                st.success("Sample dataset loaded successfully!")
            except Exception as e:
                st.error(f"Couldn't load sample data: {str(e)}")
        return
    
    # Load data with progress indicator
    with st.spinner('Loading data...'):
        df = load_data(uploaded_file)
        if df is None: 
            st.error("Failed to process file. Please check format and encoding.")
            return
        clean_df = clean_data(df)
        st.session_state.df = clean_df
    
    # Show data preview
    st.subheader("Data Preview")
    st.dataframe(clean_df.head(3))
    
    # KPI Cards
    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics safely
    total_sales = clean_df['Sales'].sum() if 'Sales' in clean_df else 0
    total_profit = clean_df['Profit'].sum() if 'Profit' in clean_df else 0
    profit_margin = (total_profit / total_sales) * 100 if total_sales > 0 else 0
    order_count = clean_df.shape[0]
    
    col1.metric("Total Sales", f"${total_sales:,.0f}")
    col2.metric("Total Profit", f"${total_profit:,.0f}")
    col3.metric("Profit Margin", f"{profit_margin:.1f}%")
    col4.metric("Orders", f"{order_count}")
    
    # Filters
    st.sidebar.header("Data Filters")
    
    # Year filter (if available)
    if 'Order_Year' in clean_df:
        years = sorted(clean_df['Order_Year'].unique())
        selected_year = st.sidebar.multiselect(
            "Select Year:",
            options=years,
            default=years
        )
    else:
        selected_year = None
    
    # Region filter
    if 'Region' in clean_df:
        regions = clean_df['Region'].unique()
        selected_region = st.sidebar.multiselect(
            "Select Region:",
            options=regions,
            default=regions
        )
    else:
        selected_region = None
    
    # Category filter
    if 'Category' in clean_df:
        categories = clean_df['Category'].unique()
        selected_category = st.sidebar.multiselect(
            "Select Category:",
            options=categories,
            default=categories
        )
    else:
        selected_category = None
    
    # Apply filters
    filtered_df = clean_df.copy()
    if selected_year:
        filtered_df = filtered_df[filtered_df['Order_Year'].isin(selected_year)]
    if selected_region:
        filtered_df = filtered_df[filtered_df['Region'].isin(selected_region)]
    if selected_category:
        filtered_df = filtered_df[filtered_df['Category'].isin(selected_category)]
    
    # Visualization Grid
    st.subheader("Performance Analysis")
    col1, col2 = st.columns([6, 4])
    with col1:
        st.plotly_chart(sales_profit_barchart(filtered_df), use_container_width=True)
    with col2:
        st.plotly_chart(segment_analysis(filtered_df), use_container_width=True)
    
    col3, col4 = st.columns([5, 5])
    with col3:
        st.plotly_chart(regional_sales_map(filtered_df), use_container_width=True)
    with col4:
        st.plotly_chart(monthly_trend_chart(filtered_df), use_container_width=True)
    
    # Recommendations Section
    st.subheader("🚀 Strategic Recommendations")
    
    # Calculate insights
    try:
        # Technology margin
        tech_margin = 0
        if 'Category' in filtered_df:
            tech_df = filtered_df[filtered_df['Category'] == 'Technology']
            if not tech_df.empty and 'Profit_Margin' in tech_df:
                tech_margin = tech_df['Profit_Margin'].mean() * 100
        
        # Southern region profit
        south_profit = 0
        if 'Region' in filtered_df:
            south_df = filtered_df[filtered_df['Region'] == 'South']
            if not south_df.empty and 'Profit' in south_df:
                south_profit = south_df['Profit'].sum()
        
        # Home office segment
        home_office_percent = 0
        if 'Segment' in filtered_df:
            home_office_df = filtered_df[filtered_df['Segment'] == 'Home Office']
            if not home_office_df.empty and 'Sales' in home_office_df:
                home_office_sales = home_office_df['Sales'].sum()
                total_sales = filtered_df['Sales'].sum()
                home_office_percent = (home_office_sales / total_sales) * 100 if total_sales > 0 else 0
        
        st.markdown(f"""
        1. **Technology Margin Improvement**:  
        Technology products show average margin of **{tech_margin:.1f}%**.  
        *Recommendation: Bundle with high-margin accessories or negotiate better supplier terms*
        
        2. **Southern Region Focus**:  
        Southern region contributes **${south_profit:,.0f}** in profit.  
        *Recommendation: Run regional promotions and optimize shipping routes*
        
        3. **Home Office Segment Growth**:  
        Home Office segment accounts for **{home_office_percent:.1f}%** of total sales.  
        *Recommendation: Develop targeted B2B marketing campaigns*
        """)
    except Exception as e:
        st.warning(f"Could not generate recommendations: {str(e)}")

if __name__ == "__main__":
    # Install chardet if not available
    try:
        import chardet
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "chardet"])
        import chardet
    
    main()