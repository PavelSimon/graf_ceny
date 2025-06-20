import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

st.set_page_config(page_title='Electricity Price & Consumption Analyzer', page_icon='static/favicon.ico')
st.title('Electricity Price & Consumption Analyzer')

def load_price_data(db_path):
    con = sqlite3.connect(db_path)
    # Read relevant columns
    df = pd.read_sql_query(
        "SELECT deliveryDay, period, deliveryStart, price FROM dam_results",
        con
    )
    # Convert deliveryDay and deliveryStart to datetime
    df['deliveryDay'] = pd.to_datetime(df['deliveryDay'])
    df['deliveryStart'] = pd.to_datetime(df['deliveryStart'])
    # Create a datetime column for merging (use deliveryStart)
    df['datetime'] = df['deliveryStart']
    return df

def get_excel_datetime_column(df):
    # Try to auto-detect datetime column
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        if 'date' in col.lower() or 'time' in col.lower():
            return col
    # If not found, ask user
    return st.selectbox('Select datetime column in Excel', df.columns)

def get_excel_consumption_column(df):
    # Try to auto-detect consumption column
    for col in df.columns:
        if 'consum' in col.lower() or 'kwh' in col.lower():
            return col
    # If not found, ask user
    return st.selectbox('Select consumption column in Excel', df.columns)

# Sidebar navigation
page = st.sidebar.radio('Select page', ['Main Analysis', 'Database Explorer', 'Excel Explorer'])

db_path = 'dam_data.db'
df_price = load_price_data(db_path)

if page == 'Main Analysis':
    # --- Main Analysis Page ---
    # Upload Excel file
    e_file = st.file_uploader('Upload your consumption Excel file', type=['xlsx'])

    if e_file:
        st.success('Excel file uploaded!')
        df_excel = pd.read_excel(e_file)
        # st.write('Excel Data Preview:', df_excel.head())

        # Combine 'Dátum' and 'Čas od' into a single datetime column (dayfirst)
        df_excel['datetime'] = pd.to_datetime(df_excel['Dátum'].astype(str) + ' ' + df_excel['Čas od'].astype(str), errors='coerce', dayfirst=True)
        df_excel = df_excel.sort_values('datetime')
        df_excel = df_excel.dropna(subset=['datetime'])  # Drop rows with missing datetime

        if df_excel.empty:
            st.error('No valid datetime values found after combining Dátum and Čas od. Please check your Excel file.')
            st.stop()

        # Use 'Hodnota profilu' as the consumption column
        cons_col = 'Hodnota profilu'
        dt_col = 'datetime'

        # Date range selection
        min_date = df_excel[dt_col].min().date()
        max_date = df_excel[dt_col].max().date()
        # st.write(f"Excel data covers: {min_date} to {max_date}")

        # Make all datetimes timezone-naive for comparison
        df_excel[dt_col] = df_excel[dt_col].dt.tz_localize(None)
        df_price['datetime'] = df_price['datetime'].dt.tz_localize(None)

        # Two separate date inputs for start and end
        start_date = st.date_input('Start date for graph', min_value=min_date, max_value=max_date, value=min_date)
        end_date = st.date_input('End date for graph', min_value=min_date, max_value=max_date, value=max_date)

        # Filter both dataframes by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df_excel = df_excel[(df_excel[dt_col] >= start_dt) & (df_excel[dt_col] <= end_dt)]
        df_price_filtered = df_price[(df_price['datetime'] >= start_dt) & (df_price['datetime'] <= end_dt)]

        # Merge on datetime (inner join)
        df_merged = pd.merge_asof(
            df_excel.sort_values(dt_col),
            df_price_filtered.sort_values('datetime'),
            left_on=dt_col,
            right_on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta('1H')  # adjust if needed
        )

        # st.write('Merged Data Preview:', df_merged[[dt_col, cons_col, 'price']].head())

        # Plot 1: Electricity price over time
        fig1 = px.line(df_merged, x=dt_col, y='price', title='Electricity Price (€/MWh)')
        st.plotly_chart(fig1)

        # Plot 2: Consumption cost over time (with Hodnota profilu)
        df_merged['price_per_kwh'] = df_merged['price'] / 1000
        df_merged['cost'] = df_merged[cons_col] * df_merged['price_per_kwh']
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_merged[dt_col], y=df_merged['cost'], mode='lines', name='Consumption Cost (€)'))
        fig2.add_trace(go.Scatter(x=df_merged[dt_col], y=df_merged[cons_col], mode='lines', name='Hodnota profilu (kWh)', yaxis='y2'))
        # Calculate global min and max for both series (allow negative)
        min_zero = min(0, df_merged['cost'].min(), df_merged[cons_col].min())
        max_zero = max(df_merged['cost'].max(), df_merged[cons_col].max())
        # Generate tick values for y-axes
        tick_step = (max_zero - min_zero) / 8 if max_zero != min_zero else 1
        tickvals = list(np.arange(min_zero, max_zero + tick_step, tick_step))
        fig2.update_layout(
            title='Consumption Cost (€) and Hodnota profilu (kWh) Over Time',
            yaxis=dict(title='Cost (€)', zeroline=True, zerolinewidth=2, zerolinecolor='gray', range=[min_zero, max_zero], tickvals=tickvals, tickformat='.2f'),
            yaxis2=dict(title='Hodnota profilu (kWh)', overlaying='y', side='right', anchor='x', zeroline=True, zerolinewidth=2, zerolinecolor='gray', range=[min_zero, max_zero], tickvals=tickvals, tickformat='.2f'),
            legend=dict(x=0, y=1)
        )
        st.plotly_chart(fig2)

        # Plot 3: Only cost over time
        fig_cost = px.line(df_merged, x=dt_col, y='cost', title='Consumption Cost (€) Over Time')
        st.plotly_chart(fig_cost)

        st.write("DB date range:", df_price['datetime'].min(), "to", df_price['datetime'].max())
        st.write("Excel date range:", df_excel[dt_col].min(), "to", df_excel[dt_col].max())

elif page == 'Database Explorer':
    # --- Database Explorer Page ---
    st.header('Database Table and Price Graph')
    # Date range for DB
    min_db_date = df_price['datetime'].min().date()
    max_db_date = df_price['datetime'].max().date()
    db_start = st.date_input('DB Start date', min_value=min_db_date, max_value=max_db_date, value=min_db_date, key='db_start')
    db_end = st.date_input('DB End date', min_value=min_db_date, max_value=max_db_date, value=max_db_date, key='db_end')
    db_start_dt = pd.to_datetime(db_start)
    db_end_dt = pd.to_datetime(db_end)
    df_price['datetime'] = df_price['datetime'].dt.tz_localize(None)
    df_db_filtered = df_price[(df_price['datetime'] >= db_start_dt) & (df_price['datetime'] <= db_end_dt)]
    st.write('Database Table:', df_db_filtered)
    fig_db = px.line(df_db_filtered, x='datetime', y='price', title='DB Electricity Price (€/MWh)')
    st.plotly_chart(fig_db)

elif page == 'Excel Explorer':
    st.header('Excel File Explorer')
    e_file = st.file_uploader('Upload your consumption Excel file (for Excel Explorer)', type=['xlsx'], key='excel_explorer_upload')
    if e_file:
        df_excel = pd.read_excel(e_file)
        st.write('Excel Data Preview:', df_excel.head())
        # Combine 'Dátum' and 'Čas od' into a single datetime column (dayfirst)
        df_excel['datetime'] = pd.to_datetime(df_excel['Dátum'].astype(str) + ' ' + df_excel['Čas od'].astype(str), errors='coerce', dayfirst=True)
        df_excel = df_excel.sort_values('datetime')
        df_excel = df_excel.dropna(subset=['datetime'])
        st.write('Cleaned Excel Data:', df_excel[['datetime', 'Hodnota profilu']].head())
        fig_excel = px.line(df_excel, x='datetime', y='Hodnota profilu', title='Consumption (Hodnota profilu) Over Time')
        st.plotly_chart(fig_excel)
    else:
        st.info('Please upload an Excel file to view the graph.')

# Placeholder: Read SQLite DB
# TODO: Read price data from db and merge with Excel

# TODO: Date range selection and plotting 