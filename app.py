import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import os

st.set_page_config(page_title='Electricity Price & Consumption Analyzer', page_icon='static/favicon.ico')
st.title('Electricity Price & Consumption Analyzer')

def load_price_data(db_path):
    con = sqlite3.connect(db_path)
    # Read relevant columns
    df = pd.read_sql_query(
        "SELECT delivery_Day, delivery_Start, period, price FROM dam_results",
        con
    )
    print(df.head())
    # Rename columns to match expected names
    df = df.rename(columns={'delivery_day': 'deliveryday', 'delivery_start': 'deliverystart'})
    # Convert deliveryDay and deliveryStart to datetime
    df['deliveryday'] = pd.to_datetime(df['deliveryday'], yearfirst=True)
    df['deliverystart'] = pd.to_datetime(df['deliverystart'], yearfirst=True)
    # Create a datetime column for merging (use deliveryStart)
    df['datetime'] = df['deliverystart']
    return df

def create_merged_database(df_excel, df_price, merged_db_path):
    """Create a new database with merged Excel and price data"""
    # Combine 'Dátum' and 'Čas od' into a single datetime column (dayfirst)
    df_excel['datetime'] = pd.to_datetime(df_excel['Dátum'].astype(str) + ' ' + df_excel['Čas od'].astype(str), errors='coerce', dayfirst=True)
    # Set Excel datetimes to Europe/Prague (CET/CEST), then convert to UTC
    df_excel['datetime'] = df_excel['datetime'].dt.tz_localize('Europe/Prague', ambiguous='NaT', nonexistent='shift_forward').dt.tz_convert('UTC')
    df_excel = df_excel.sort_values('datetime')
    df_excel = df_excel.dropna(subset=['datetime'])  # Drop rows with missing datetime
    
    if df_excel.empty:
        return None
    
    # Use 'Hodnota profilu' as the consumption column
    cons_col = 'Hodnota profilu'
    dt_col = 'datetime'
    
    # Make all datetimes timezone-naive for comparison (convert UTC to naive)
    df_excel[dt_col] = df_excel[dt_col].dt.tz_convert(None)
    df_price['datetime'] = df_price['datetime'].dt.tz_localize(None)
    
    # Merge on datetime (inner join)
    df_merged = pd.merge_asof(
        df_excel.sort_values(dt_col),
        df_price.sort_values('datetime'),
        left_on=dt_col,
        right_on='datetime',
        direction='nearest',
        tolerance=pd.Timedelta('1h')  # adjust if needed
    )
    
    # Calculate cost
    df_merged['price_per_kwh'] = df_merged['price'] / 1000
    df_merged['cost'] = df_merged[cons_col] * df_merged['price_per_kwh']
    
    # Save to new database
    con = sqlite3.connect(merged_db_path)
    df_merged.to_sql('merged_data', con, if_exists='replace', index=False)
    con.close()
    
    return df_merged

def load_merged_data(merged_db_path):
    """Load merged data from the created database"""
    if not os.path.exists(merged_db_path):
        return None
    
    con = sqlite3.connect(merged_db_path)
    df = pd.read_sql_query("SELECT * FROM merged_data", con)
    con.close()
    
    # Convert datetime column back to datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
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
page = st.sidebar.radio('Select page', ['Main Analysis', 'Database Explorer', 'Excel Explorer', 'Merged Data Explorer'])

db_path = 'okte.db'
merged_db_path = 'merged_data.db'
df_price = load_price_data(db_path)

if page == 'Main Analysis':
    # --- Main Analysis Page ---
    # Upload Excel file
    e_file = st.file_uploader('Upload your consumption Excel file', type=['xlsx'])

    if e_file:
        st.success('Excel file uploaded!')
        df_excel = pd.read_excel(e_file)
        
        # Create merged database
        df_merged = create_merged_database(df_excel, df_price, merged_db_path)
        
        if df_merged is None:
            st.error('No valid datetime values found after combining Dátum and Čas od. Please check your Excel file.')
            st.stop()
        
        st.success('Merged database created successfully!')
        
        # Use the merged data for all operations
        dt_col = 'datetime'
        cons_col = 'Hodnota profilu'
        
        # Date range selection
        min_date = df_merged[dt_col].min().date()
        max_date = df_merged[dt_col].max().date()

        # Two separate date inputs for start and end
        start_date = st.date_input('Start date for graph', min_value=min_date, max_value=max_date, value=min_date)
        
        # Initialize end_date with max_date
        end_date = max_date
        
        # Auto-adjust end date if start date is higher than end date
        if start_date > max_date:
            end_date = start_date + pd.Timedelta(days=1)
            st.warning(f'End date automatically adjusted to {end_date.strftime("%Y-%m-%d")} (start date + 1 day)')
        
        end_date = st.date_input('End date for graph', min_value=min_date, max_value=max_date, value=end_date)

        # Filter merged data by date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df_filtered = df_merged[(df_merged[dt_col] >= start_dt) & (df_merged[dt_col] <= end_dt)]

        # Plot 1: Electricity price over time
        fig1 = px.line(df_filtered, x=dt_col, y='price', title='Electricity Price (€/MWh)')
        st.plotly_chart(fig1)

        # Plot 2: Consumption cost over time (with Hodnota profilu)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_filtered[dt_col], y=df_filtered['cost'], mode='lines', name='Consumption Cost (€)'))
        fig2.add_trace(go.Scatter(x=df_filtered[dt_col], y=df_filtered[cons_col], mode='lines', name='Hodnota profilu (kWh)', yaxis='y2'))
        # Calculate global min and max for both series (allow negative)
        min_zero = min(0, df_filtered['cost'].min(), df_filtered[cons_col].min())
        max_zero = max(df_filtered['cost'].max(), df_filtered[cons_col].max())
        # Generate tick values for y-axes
        if max_zero != min_zero and max_zero > min_zero:
            tick_step = (max_zero - min_zero) / 8
            tickvals = list(np.arange(min_zero, max_zero + tick_step, tick_step))
        else:
            tickvals = [min_zero, max_zero] if min_zero != max_zero else [min_zero]
        fig2.update_layout(
            title='Consumption Cost (€) and Hodnota profilu (kWh) Over Time',
            yaxis=dict(title='Cost (€)', zeroline=True, zerolinewidth=2, zerolinecolor='gray', range=[min_zero, max_zero], tickvals=tickvals, tickformat='.2f'),
            yaxis2=dict(title='Hodnota profilu (kWh)', overlaying='y', side='right', anchor='x', zeroline=True, zerolinewidth=2, zerolinecolor='gray', range=[min_zero, max_zero], tickvals=tickvals, tickformat='.2f'),
            legend=dict(x=0, y=1)
        )
        st.plotly_chart(fig2)

        # Plot 3: Only cost over time
        fig_cost = px.line(df_filtered, x=dt_col, y='cost', title='Consumption Cost (€) Over Time')
        st.plotly_chart(fig_cost)

        # Statistics Table
        st.header('📊 Statistics Summary')
        
        # Calculate statistics for price
        avg_price = df_filtered['price'].mean()
        max_price_idx = df_filtered['price'].idxmax()
        min_price_idx = df_filtered['price'].idxmin()
        max_price = df_filtered.loc[max_price_idx, 'price']
        min_price = df_filtered.loc[min_price_idx, 'price']
        max_price_datetime = df_filtered.loc[max_price_idx, dt_col]
        min_price_datetime = df_filtered.loc[min_price_idx, dt_col]
        
        # Calculate statistics for cost
        avg_cost = df_filtered['cost'].mean()
        max_cost_idx = df_filtered['cost'].idxmax()
        min_cost_idx = df_filtered['cost'].idxmin()
        max_cost = df_filtered.loc[max_cost_idx, 'cost']
        min_cost = df_filtered.loc[min_cost_idx, 'cost']
        max_cost_datetime = df_filtered.loc[max_cost_idx, dt_col]
        min_cost_datetime = df_filtered.loc[min_cost_idx, dt_col]
        
        # Convert average cost to €/MWh (multiply by 1000 since cost is in €/kWh)
        avg_cost_mwh = avg_cost * 1000
        
        # Create the statistics table
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Price", f"{avg_price:.2f} €/MWh")
            st.metric("Average Cost", f"{avg_cost_mwh:.2f} €/MWh")
        
        with col2:
            st.metric("Maximum Price", f"{max_price:.2f} €/MWh")
            st.caption(f"Date: {max_price_datetime.strftime('%Y-%m-%d %H:%M')}")
            st.metric("Maximum Cost", f"{max_cost:.2f} €/MWh")
            st.caption(f"Date: {max_cost_datetime.strftime('%Y-%m-%d %H:%M')}")
        
        with col3:
            st.metric("Minimum Price", f"{min_price:.2f} €/MWh")
            st.caption(f"Date: {min_price_datetime.strftime('%Y-%m-%d %H:%M')}")
            st.metric("Minimum Cost", f"{min_cost:.2f} €/MWh")
            st.caption(f"Date: {min_cost_datetime.strftime('%Y-%m-%d %H:%M')}")
        
        # Alternative table format using st.dataframe
        st.subheader("📋 Detailed Statistics Table")
        
        # Create a DataFrame for the table
        stats_data = {
            'Metric': [
                'Average Price (€/MWh)',
                'Maximum Price (€/MWh)',
                'Minimum Price (€/MWh)',
                'Average Cost (€/MWh)',
                'Maximum Cost (€/MWh)',
                'Minimum Cost (€/MWh)'
            ],
            'Value': [
                f"{avg_price:.2f}",
                f"{max_price:.2f}",
                f"{min_price:.2f}",
                f"{avg_cost_mwh:.2f}",
                f"{max_cost:.2f}",
                f"{min_cost:.2f}"
            ],
            'Date & Time': [
                'N/A',
                max_price_datetime.strftime('%Y-%m-%d %H:%M'),
                min_price_datetime.strftime('%Y-%m-%d %H:%M'),
                'N/A',
                max_cost_datetime.strftime('%Y-%m-%d %H:%M'),
                min_cost_datetime.strftime('%Y-%m-%d %H:%M')
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)

        st.write("DB date range:", df_price['datetime'].min(), "to", df_price['datetime'].max())
        st.write("Merged data date range:", df_merged[dt_col].min(), "to", df_merged[dt_col].max())
        
    else:
        # Check if merged database exists and load it
        df_merged = load_merged_data(merged_db_path)
        
        if df_merged is not None:
            st.info('Using previously uploaded Excel data. Upload a new file to replace it.')
            
            # Use the merged data for all operations
            dt_col = 'datetime'
            cons_col = 'Hodnota profilu'
            
            # Date range selection
            min_date = df_merged[dt_col].min().date()
            max_date = df_merged[dt_col].max().date()

            # Two separate date inputs for start and end
            start_date = st.date_input('Start date for graph', min_value=min_date, max_value=max_date, value=min_date)
            
            # Initialize end_date with max_date
            end_date = max_date
            
            # Auto-adjust end date if start date is higher than end date
            if start_date > max_date:
                end_date = start_date + pd.Timedelta(days=1)
                st.warning(f'End date automatically adjusted to {end_date.strftime("%Y-%m-%d")} (start date + 1 day)')
            
            end_date = st.date_input('End date for graph', min_value=min_date, max_value=max_date, value=end_date)

            # Filter merged data by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df_filtered = df_merged[(df_merged[dt_col] >= start_dt) & (df_merged[dt_col] <= end_dt)]

            # Plot 1: Electricity price over time
            fig1 = px.line(df_filtered, x=dt_col, y='price', title='Electricity Price (€/MWh)')
            st.plotly_chart(fig1)

            # Plot 2: Consumption cost over time (with Hodnota profilu)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_filtered[dt_col], y=df_filtered['cost'], mode='lines', name='Consumption Cost (€)'))
            fig2.add_trace(go.Scatter(x=df_filtered[dt_col], y=df_filtered[cons_col], mode='lines', name='Hodnota profilu (kWh)', yaxis='y2'))
            # Calculate global min and max for both series (allow negative)
            min_zero = min(0, df_filtered['cost'].min(), df_filtered[cons_col].min())
            max_zero = max(df_filtered['cost'].max(), df_filtered[cons_col].max())
            # Generate tick values for y-axes
            if max_zero != min_zero and max_zero > min_zero:
                tick_step = (max_zero - min_zero) / 8
                tickvals = list(np.arange(min_zero, max_zero + tick_step, tick_step))
            else:
                tickvals = [min_zero, max_zero] if min_zero != max_zero else [min_zero]
            fig2.update_layout(
                title='Consumption Cost (€) and Hodnota profilu (kWh) Over Time',
                yaxis=dict(title='Cost (€)', zeroline=True, zerolinewidth=2, zerolinecolor='gray', range=[min_zero, max_zero], tickvals=tickvals, tickformat='.2f'),
                yaxis2=dict(title='Hodnota profilu (kWh)', overlaying='y', side='right', anchor='x', zeroline=True, zerolinewidth=2, zerolinecolor='gray', range=[min_zero, max_zero], tickvals=tickvals, tickformat='.2f'),
                legend=dict(x=0, y=1)
            )
            st.plotly_chart(fig2)

            # Plot 3: Only cost over time
            fig_cost = px.line(df_filtered, x=dt_col, y='cost', title='Consumption Cost (€) Over Time')
            st.plotly_chart(fig_cost)

            # Statistics Table
            st.header('📊 Statistics Summary')
            
            # Calculate statistics for price
            avg_price = df_filtered['price'].mean()
            max_price_idx = df_filtered['price'].idxmax()
            min_price_idx = df_filtered['price'].idxmin()
            max_price = df_filtered.loc[max_price_idx, 'price']
            min_price = df_filtered.loc[min_price_idx, 'price']
            max_price_datetime = df_filtered.loc[max_price_idx, dt_col]
            min_price_datetime = df_filtered.loc[min_price_idx, dt_col]
            
            # Calculate statistics for cost
            avg_cost = df_filtered['cost'].mean()
            max_cost_idx = df_filtered['cost'].idxmax()
            min_cost_idx = df_filtered['cost'].idxmin()
            max_cost = df_filtered.loc[max_cost_idx, 'cost']
            min_cost = df_filtered.loc[min_cost_idx, 'cost']
            max_cost_datetime = df_filtered.loc[max_cost_idx, dt_col]
            min_cost_datetime = df_filtered.loc[min_cost_idx, dt_col]
            
            # Convert average cost to €/MWh (multiply by 1000 since cost is in €/kWh)
            avg_cost_mwh = avg_cost * 1000
            
            # Create the statistics table
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Price", f"{avg_price:.2f} €/MWh")
                st.metric("Average Cost", f"{avg_cost_mwh:.2f} €/MWh")
            
            with col2:
                st.metric("Maximum Price", f"{max_price:.2f} €/MWh")
                st.caption(f"Date: {max_price_datetime.strftime('%Y-%m-%d %H:%M')}")
                st.metric("Maximum Cost", f"{max_cost:.2f} €/MWh")
                st.caption(f"Date: {max_cost_datetime.strftime('%Y-%m-%d %H:%M')}")
            
            with col3:
                st.metric("Minimum Price", f"{min_price:.2f} €/MWh")
                st.caption(f"Date: {min_price_datetime.strftime('%Y-%m-%d %H:%M')}")
                st.metric("Minimum Cost", f"{min_cost:.2f} €/MWh")
                st.caption(f"Date: {min_cost_datetime.strftime('%Y-%m-%d %H:%M')}")
            
            # Alternative table format using st.dataframe
            st.subheader("📋 Detailed Statistics Table")
            
            # Create a DataFrame for the table
            stats_data = {
                'Metric': [
                    'Average Price (€/MWh)',
                    'Maximum Price (€/MWh)',
                    'Minimum Price (€/MWh)',
                    'Average Cost (€/MWh)',
                    'Maximum Cost (€/MWh)',
                    'Minimum Cost (€/MWh)'
                ],
                'Value': [
                    f"{avg_price:.2f}",
                    f"{max_price:.2f}",
                    f"{min_price:.2f}",
                    f"{avg_cost_mwh:.2f}",
                    f"{max_cost:.2f}",
                    f"{min_cost:.2f}"
                ],
                'Date & Time': [
                    'N/A',
                    max_price_datetime.strftime('%Y-%m-%d %H:%M'),
                    min_price_datetime.strftime('%Y-%m-%d %H:%M'),
                    'N/A',
                    max_cost_datetime.strftime('%Y-%m-%d %H:%M'),
                    min_cost_datetime.strftime('%Y-%m-%d %H:%M')
                ]
            }
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df, use_container_width=True)

            st.write("DB date range:", df_price['datetime'].min(), "to", df_price['datetime'].max())
            st.write("Merged data date range:", df_merged[dt_col].min(), "to", df_merged[dt_col].max())
            
        else:
            st.info('Please upload an Excel file to start the analysis.')
            
            # Add a button to clear the merged database if it exists
            if os.path.exists(merged_db_path):
                if st.button('Clear Previous Data'):
                    os.remove(merged_db_path)
                    st.success('Previous data cleared!')
                    st.rerun()

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
    st.write('Database Table:')
    st.dataframe(df_db_filtered, use_container_width=True)
    fig_db = px.line(df_db_filtered, x='datetime', y='price', title='DB Electricity Price (€/MWh)')
    st.plotly_chart(fig_db)

elif page == 'Excel Explorer':
    st.header('Excel File Explorer')
    e_file = st.file_uploader('Upload your consumption Excel file (for Excel Explorer)', type=['xlsx'], key='excel_explorer_upload')
    if e_file:
        df_excel = pd.read_excel(e_file)
        st.write('Excel Data Preview:')
        st.dataframe(df_excel.head(), use_container_width=True)
        # Combine 'Dátum' and 'Čas od' into a single datetime column (dayfirst)
        df_excel['datetime'] = pd.to_datetime(df_excel['Dátum'].astype(str) + ' ' + df_excel['Čas od'].astype(str), errors='coerce', dayfirst=True)
        # Set Excel datetimes to Europe/Prague (CET/CEST), then convert to UTC
        df_excel['datetime'] = df_excel['datetime'].dt.tz_localize('Europe/Prague', ambiguous='NaT', nonexistent='shift_forward').dt.tz_convert('UTC')
        df_excel = df_excel.sort_values('datetime')
        df_excel = df_excel.dropna(subset=['datetime'])
        st.write('Cleaned Excel Data:')
        st.dataframe(df_excel[['datetime', 'Hodnota profilu']].head(), use_container_width=True)
        fig_excel = px.line(df_excel, x='datetime', y='Hodnota profilu', title='Consumption (Hodnota profilu) Over Time')
        st.plotly_chart(fig_excel)
    else:
        st.info('Please upload an Excel file to view the graph.')

elif page == 'Merged Data Explorer':
    st.header('Merged Data Explorer')
    
    if os.path.exists(merged_db_path):
        df_merged = load_merged_data(merged_db_path)
        
        st.write('Merged Data Preview:')
        st.dataframe(df_merged.head(), use_container_width=True)
        
        dt_col = 'datetime'
        
        min_date = df_merged[dt_col].min().date()
        max_date = df_merged[dt_col].max().date()

        start_date = st.date_input('Start date', min_value=min_date, max_value=max_date, value=min_date, key='merged_start')
        end_date = st.date_input('End date', min_value=min_date, max_value=max_date, value=max_date, key='merged_end')

        if start_date > end_date:
            end_date = start_date + pd.Timedelta(days=1)
            st.warning(f'End date automatically adjusted to {end_date.strftime("%Y-%m-%d")} (start date + 1 day)')
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df_filtered = df_merged[(df_merged[dt_col] >= start_dt) & (df_merged[dt_col] <= end_dt)]
        
        st.write('Filtered Merged Data:')
        st.dataframe(df_filtered, use_container_width=True)
        
        fig_merged_cost = px.line(df_filtered, x=dt_col, y='cost', title='Cost from Merged Data (€)')
        st.plotly_chart(fig_merged_cost)
        
    else:
        st.info('No merged data found. Please upload an Excel file on the "Main Analysis" page to create the merged database.')

# Placeholder: Read SQLite DB
# TODO: Read price data from db and merge with Excel

# TODO: Date range selection and plotting 