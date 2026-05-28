import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import requests
from io import StringIO
import logging
import numpy as np

# Set up logging to debug issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Data Loading (Load once to improve performance) ---
df_zillow_data = None
data_loading_error = None
latest_data_date_str = "N/A"

def load_zillow_data():
    global df_zillow_data, data_loading_error, latest_data_date_str
    url = 'https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'
    fallback_url = 'https://www.zillow.com/research/data/'  # Reference for users to find the latest data
    
    # Try primary URL
    try:
        logger.info("Attempting to load data from primary URL")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        df_zillow_data = pd.read_csv(StringIO(response.text), dtype={'RegionName': str})
        logger.info("Successfully loaded data from primary URL")
    except Exception as e:
        logger.warning(f"Primary URL failed: {e}")
        data_loading_error = f"Primary URL failed: {e}. Please check https://www.zillow.com/research/data/ for the latest ZHVI dataset."
        
        # Fallback to sample data
        try:
            logger.info("Using sample data as fallback")
            sample_data = {
                'RegionName': ['90210', '94105', '92101'],
                'State': ['CA', 'CA', 'CA'],
                'City': ['Beverly Hills', 'San Francisco', 'San Diego'],
                '2024-05-31': [5000000, 1200000, 800000],
                '2025-05-31': [5200000, 1250000, 790000]
            }
            df_zillow_data = pd.DataFrame(sample_data)
            logger.info("Successfully loaded sample data")
            data_loading_error = "Using sample data due to failure in loading live data."
        except Exception as e:
            logger.error(f"Fallback also failed: {e}")
            data_loading_error = f"Data loading failed: {e}"
            df_zillow_data = None
            return
    
    # Process date columns
    if df_zillow_data is not None:
        date_columns = [col for col in df_zillow_data.columns if col.count('-') == 2 and col.startswith(('19', '20'))]
        if date_columns:
            latest_data_date_str = pd.to_datetime(date_columns[-1]).strftime('%B %Y')
        else:
            logger.warning("No date columns found in data")
            data_loading_error = "No date columns found in data"
            df_zillow_data = None

load_zillow_data()

# --- Plotting Functions ---

def plot_zip_code_trends_with_insights(state_abbr):
    logger.info(f"Generating ZIP Code Price Trends for state: {state_abbr}")
    if df_zillow_data is None:
        logger.error("Data not loaded for ZIP Code Price Trends")
        return go.Figure().update_layout(
            title_text=data_loading_error or "Zillow data not loaded.",
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Arial", size=14, color="#000000")
        ), "Data not loaded."
    
    state_df = df_zillow_data[df_zillow_data['State'] == state_abbr.upper()].copy()
    if state_df.empty:
        logger.warning(f"No data found for state: {state_abbr}")
        return go.Figure().update_layout(
            title_text=f"No data found for state: {state_abbr.upper()}.",
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Arial", size=14, color="#000000")
        ), f"No data available for {state_abbr.upper()}."
    
    date_cols = [col for col in state_df.columns if col.count('-') == 2 and col.startswith(('19', '20'))]
    if not date_cols:
        logger.warning("No date columns found for state data")
        return go.Figure().update_layout(
            title_text="No date data available.",
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Arial", size=14, color="#000000")
        ), "No date data available."
    
    x_dates = pd.to_datetime(date_cols)
    
    # Calculate percentage change over the last year (approx. 12 months)
    one_year_ago_idx = max(len(date_cols) - 12, 0)
    recent_col = date_cols[-1]
    one_year_ago_col = date_cols[one_year_ago_idx]
    
    # Initialize figure
    fig = go.Figure()
    
    # Track overall trend for summary
    percentage_changes = []
    
    for _, row in state_df.iterrows():
        zip_code = row['RegionName']
        city = row['City']
        prices = row[date_cols].values
        valid_indices = [i for i, price in enumerate(prices) if pd.notna(price)]
        
        if valid_indices:
            recent_price = prices[-1] if pd.notna(prices[-1]) else None
            one_year_ago_price = prices[one_year_ago_idx] if pd.notna(prices[one_year_ago_idx]) else None
            if recent_price and one_year_ago_price and one_year_ago_price != 0:
                pct_change = ((recent_price - one_year_ago_price) / one_year_ago_price) * 100
                percentage_changes.append(pct_change)
            else:
                pct_change = None
            
            hover_text = [f"ZIP: {zip_code}<br>City: {city}<br>Date: {x_dates[i].strftime('%Y-%m')}<br>Price: ${prices[i]:,.0f}" for i in valid_indices]
            if pct_change is not None:
                trend = "up" if pct_change > 0 else "down"
                hover_text[-1] += f"<br>1-Year Change: {pct_change:+.1f}% ({trend})"
            
            fig.add_trace(go.Scatter(
                x=x_dates[valid_indices],
                y=prices[valid_indices],
                mode='lines',
                name=str(zip_code),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>"
            ))
    
    fig.update_layout(
        title_text=f'Housing Price Trends by ZIP Code in {state_abbr.upper()} ({latest_data_date_str})',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        legend_title="ZIP Code",
        hovermode="closest",  # Changed from "x unified" to show only the hovered line's info
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Arial", size=14, color="#000000"),
        xaxis=dict(gridcolor="#D3D3D3"),
        yaxis=dict(gridcolor="#D3D3D3")
    )
    
    if len(state_df['RegionName'].unique()) > 30:
        fig.update_layout(showlegend=False, title_text=fig.layout.title.text + " (Legend hidden due to >30 ZIPs)")
    
    summary = f"Prices in {state_abbr.upper()} are generally trending {'upward' if np.mean(percentage_changes) > 0 else 'downward'} (average 1-year change: {np.mean(percentage_changes):+.1f}%)." if percentage_changes else f"Insufficient data to determine price trends in {state_abbr.upper()}."
    
    logger.info(f"ZIP Code Price Trends plot generated for {state_abbr}")
    return fig, summary

def get_zip_code_table(state_abbr):
    logger.info(f"Generating ZIP Code Table for state: {state_abbr}")
    if df_zillow_data is None:
        logger.error("Data not loaded for ZIP Code Table")
        return pd.DataFrame({"Error": [data_loading_error or "Zillow data not loaded."]})
    
    state_df = df_zillow_data[df_zillow_data['State'] == state_abbr.upper()].copy()
    if state_df.empty:
        logger.warning(f"No data for state: {state_abbr}")
        return pd.DataFrame({"Error": [f"No data available for {state_abbr.upper()}."]})
    
    # Identify date columns and latest column
    date_cols = [col for col in state_df.columns if col.count('-') == 2 and col.startswith(('19', '20'))]
    if not date_cols:
        logger.warning("No date columns found for state data")
        return pd.DataFrame({"Error": ["No date data available."]})
    latest_col = date_cols[-1]

    # Build numeric price for sorting, and formatted string for display
    state_df['Final ZHVI'] = pd.to_numeric(state_df[latest_col], errors='coerce')
    state_df['Final ZHVI (USD)'] = state_df['Final ZHVI'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
    state_df['ZIP Code - City'] = state_df['RegionName'].astype(str) + " - " + state_df['City'].astype(str)

    # Sort by numeric price (desc), put NaNs last, return display columns
    table_df = (state_df
                .sort_values(by='Final ZHVI', ascending=False, na_position='last')
                [['ZIP Code - City', 'Final ZHVI (USD)']]
                .reset_index(drop=True))
    logger.info(f"ZIP Code Table generated for {state_abbr}")
    return table_df

# --- Main Function for Gradio ---
def generate_visualizations(state_input_text):
    if not state_input_text or not isinstance(state_input_text, str) or len(state_input_text.strip()) != 2:
        empty_fig = go.Figure().update_layout(
            title_text="Please enter a valid two-letter state abbreviation (e.g., CA, NY, TX).",
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Arial", size=14, color="#000000")
        )
        return empty_fig, "Invalid input. Please enter a two-letter state abbreviation.", pd.DataFrame({"Error": ["Invalid input."]})
    
    state_abbr = state_input_text.strip().upper()
    if df_zillow_data is None or state_abbr not in df_zillow_data['State'].unique():
        empty_fig = go.Figure().update_layout(
            title_text=f"Invalid state abbreviation or no data: {state_abbr}.",
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Arial", size=14, color="#000000")
        )
        return empty_fig, f"Invalid state abbreviation or no data for {state_abbr}.", pd.DataFrame({"Error": [f"No data for {state_abbr}."]})
    
    plot, summary = plot_zip_code_trends_with_insights(state_abbr)
    table = get_zip_code_table(state_abbr)
    return plot, summary, table

# --- Gradio Interface ---
custom_theme = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="red",
    neutral_hue="gray",
    text_size="lg",
    font=["Times New Roman", "serif"]
)

with gr.Blocks(theme=custom_theme) as iface:
    gr.Markdown(
        f"""
        # 🇺🇸 USA States Real Estate Market Trends
        Track housing price trends and current values by ZIP code to make informed decisions as a property owner or buyer.  
        **Data up to {latest_data_date_str}**. Enter a two-letter state abbreviation below (e.g., CA, NY, TX).
        ### [Contact a real estate broker](https://micheled.com)
        """
    )
    with gr.Row():
        state_input = gr.Textbox(
            label="State Abbreviation",
            placeholder="Enter two-letter state code (e.g., CA)",
            max_lines=1,
            value="CA",
            elem_classes="input-box"
        )
        submit_btn = gr.Button("Analyze Trends", variant="primary")
    
    plot = gr.Plot(label="ZIP Code Price Trends")
    summary = gr.Textbox(label="Market Insights for Property Owners and Buyers", lines=2, interactive=False)
    table = gr.Dataframe(label="Current ZHVI by ZIP Code", headers=["ZIP Code - City", "Final ZHVI (USD)"], wrap=True)
    
    submit_btn.click(
        fn=generate_visualizations,
        inputs=state_input,
        outputs=[plot, summary, table]
    )

if __name__ == "__main__":
    iface.launch(share=False, debug=True)
