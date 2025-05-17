import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import io
import yfinance as yf
import datetime
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_message
import base64
import os
from pathlib import Path
import random 

st.set_page_config(page_title="Stock ML Pipeline", layout="wide", page_icon="📈")

# Initialize session state
def init_session_state():
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = {
            'current_step': 0,
            'data_loaded': False,
            'preprocessed': False,
            'features_engineered': False,
            'data_split': False,
            'model_trained': False,
            'model_evaluated': False,
            'results_visualized': False,
            'df': None,
            'df_processed': None,
            'target': None,
            'features': None,
            'X_train': None,
            'X_test': None,
            'y_train': None,
            'y_test': None,
            'models': {},
            'y_preds': {},
            'current_price': None,
            'last_symbol': None,
        }
    
    # Initialize theme state if not present
    if 'theme' not in st.session_state:
        st.session_state.theme = 'default'

init_session_state()


ASSETS_DIR = Path(__file__).parent / "assets" / "backgrounds"

# Function to get image path and ensure it exists
def get_image_path(image_name):
    image_path = ASSETS_DIR / image_name
    if not image_path.exists():
        st.warning(f"Image file not found: {image_path}")
        # Return a fallback image or None
        return None
    return image_path

def theme_selector():
    themes = {
        'default': 'Default Dark',
        'cyberpunk': 'Cyberpunk',
        'blue-image': 'Blue Image',
        'red-orange-grey': 'Red Orange Grey'
    }
    
    selected_theme = st.sidebar.selectbox(
        "Select Theme", 
        list(themes.keys()),
        format_func=lambda x: themes[x],
        index=list(themes.keys()).index(st.session_state.theme)
    )
    
    if selected_theme != st.session_state.theme:
        st.session_state.theme = selected_theme
        st.rerun()

# Apply current theme CSS
def apply_theme_css():
    # Define theme_css dictionary first
    theme_css = {
        'default': lambda: f"""
    <style>
    .stApp {{
        background:  
                    url("data:image/jpg;base64,{get_base64_encoded_image(get_image_path('image (2).jpg'))}") no-repeat center center fixed;
        background-size: cover;
    }}
    
    /* Default theme - Space with blue/teal accents */
    h1, h2, h3, h4, h5, h6 {{ 
        color: #00b4d8; /* Bright blue headings */
        font-family: 'Arial', sans-serif;
        text-shadow: 0 0 5px rgba(0, 180, 216, 0.5);
    }}
    
    p, li, span {{ 
        color: #e0f7fa; /* Light cyan for regular text */
    }}
    
    .stButton>button {{ 
        background-color: rgba(0, 180, 216, 0.2);
        color: #e0f7fa;
        border: 1px solid #00b4d8;
        transition: all 0.3s ease;
    }}
    
    .stButton>button:hover {{ 
        background-color: rgba(0, 180, 216, 0.4);
        box-shadow: 0 0 10px rgba(0, 180, 216, 0.7);
    }}
    
    .stButton>button[kind="primary"] {{ 
        background-color: rgba(0, 119, 182, 0.6);
        color: #ffffff;
        border: 1px solid #0077b6;
    }}
    
    .neon-text {{
        color: #90e0ef; /* Light blue text */
        text-shadow: 0 0 5px #90e0ef, 0 0 10px #00b4d8, 0 0 20px #0077b6;
    }}
    
    .title-text {{
        color: #caf0f8; /* Very light blue */
        text-shadow: 0 0 5px #caf0f8, 0 0 10px #48cae4, 0 0 20px #0096c7;
    }}
    
    .features-box {{
        border: 2px solid #0077b6;
    }}
    
    .feature-item:nth-child(odd) {{
        color: #ade8f4; /* Light blue */
    }}
    
    .feature-item:nth-child(even) {{
        color: #caf0f8; /* Very light blue */
    }}
    
    .stats-card {{
        border: 2px solid #0077b6;
    }}
    
    .stats-card:nth-child(1) .neon-text {{
        color: #90e0ef;
    }}
    
    .stats-card:nth-child(2) .neon-text {{
        color: #ade8f4;
    }}
    
    .stats-card:nth-child(3) .neon-text {{
        color: #caf0f8;
    }}
    
    .fun-fact {{
        border: 2px solid #48cae4;
    }}
    </style>
   """,
    'cyberpunk': lambda: f"""
    <style>
    .stApp {{
        background: 
            url("data:image/jpg;base64,{get_base64_encoded_image(get_image_path('cyber.png'))}") no-repeat center center fixed;
        background-size: cover;
    }}

    /* Cyberpunk theme - Purple, pink and cyan */
    h1, h2, h3, h4, h5, h6 {{ 
        color: #bf00ff; /* Vibrant purple */
        font-family: 'Courier New', monospace;
        text-shadow:
            0 0 2px #bf00ff,
            0 0 4px #5e17eb;
    }}
    
    p, li, span {{ 
        color: #d8b5ff; /* Light purple for text */
    }}

    .stButton>button {{ 
        background-color: rgba(190, 75, 219, 0.15);
        color: #d8b5ff;
        border: 1px solid #be4bdb;
        border-radius: 6px;
        font-weight: 500;
    }}

    .stButton>button:hover {{
        background-color: rgba(242, 166, 255, 0.2);
        color: #ffffff;
    }}

    .stButton>button[kind="primary"] {{
        background-color: rgba(155, 89, 182, 0.3);
        color: #ffffff;
        border: 1px solid #9b59b6;
    }}

    .stDataFrame, .stMetric, .stExpander, .stSelectbox, .stMultiSelect, .stFileUploader, .stPlotlyChart, .stPyplot {{
        background-color: rgba(20, 20, 40, 0.7);
        border: 1px solid rgba(190, 75, 219, 0.3);
        color: #d8b5ff;
    }}

    .css-1d391kg, .css-1v3fvcr, .css-1avcm0n, .css-hxt7ib {{
        background-color: rgba(20, 20, 40, 0.6);
        color: #d8b5ff;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }}
    
    .neon-text {{
        color: #e0aaff; /* Light purple text */
        text-shadow: 0 0 5px #e0aaff, 0 0 10px #c77dff, 0 0 20px #9d4edd;
    }}
    
    .title-text {{
        color: #f72585; /* Hot pink */
        text-shadow: 0 0 5px #f72585, 0 0 10px #b5179e, 0 0 20px #7209b7;
    }}
    
    .features-box {{
        border: 2px solid #9d4edd;
    }}
    
    .feature-item:nth-child(odd) {{
        color: #e0aaff; /* Light purple */
    }}
    
    .feature-item:nth-child(even) {{
        color: #c77dff; /* Medium purple */
    }}
    
    .stats-card {{
        border: 2px solid #9d4edd;
    }}
    
    .stats-card:nth-child(1) .neon-text {{
        color: #5e17eb; /* Deep purple */
    }}
    
    .stats-card:nth-child(2) .neon-text {{
        color: #7209b7; /* Medium purple */
    }}
    
    .stats-card:nth-child(3) .neon-text {{
        color: #f72585; /* Hot pink */
    }}
    
    .fun-fact {{
        border: 2px solid #c77dff;
    }}
""",
    'blue-image': lambda: f"""
    <style>
    .stApp {{
        background: 
                    url("data:image/jpg;base64,{get_base64_encoded_image(get_image_path('blues.jpg'))}") no-repeat center center fixed;
        background-size: cover;
    }}
    
    /* Blue theme - Ocean blues with gold accents */
    h1, h2, h3, h4, h5, h6 {{ 
        color: #ffd700; /* Gold headings */
        font-family: 'Arial', sans-serif; 
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
    }}
    
    p, li, span {{ 
        color: #f0f8ff; /* Light blue for text */
    }}
    
    .stButton>button {{ 
        background-color: rgba(65, 105, 225, 0.3); 
        color: #f0f8ff; 
        border: 1px solid #4169e1;
    }}
    
    .stButton>button:hover {{
        background-color: rgba(65, 105, 225, 0.5);
        box-shadow: 0 0 10px rgba(65, 105, 225, 0.7);
    }}
    
    .stButton>button[kind="primary"] {{ 
        background-color: rgba(70, 130, 180, 0.6); 
        border-color: #4682b4; 
    }}
    
    .stDataFrame, .stMetric, .stExpander, .stSelectbox, .stMultiSelect, .stFileUploader, .stPlotlyChart, .stPyplot {{
        background-color: rgba(30, 60, 114, 0.7);
        border: 1px solid rgba(135, 206, 250, 0.3);
    }}
    
    .neon-text {{
        color: #87cefa; /* Light sky blue */
        text-shadow: 0 0 5px #87cefa, 0 0 10px #1e90ff, 0 0 20px #4169e1;
    }}
    
    .title-text {{
        color: #ffd700; /* Gold for title */
        text-shadow: 0 0 5px #ffd700, 0 0 10px #daa520, 0 0 20px #b8860b;
    }}
    
    .features-box {{
        border: 2px solid #4682b4;
    }}
    
    .feature-item:nth-child(odd) {{
        color: #add8e6; /* Light blue */
    }}
    
    .feature-item:nth-child(even) {{
        color: #b0e0e6; /* Powder blue */
    }}
    
    .stats-card {{
        border: 2px solid #4682b4;
    }}
    
    .stats-card:nth-child(1) .neon-text {{
        color: #daa520; /* Goldenrod */
    }}
    
    .stats-card:nth-child(2) .neon-text {{
        color: #ffd700; /* Gold */
    }}
    
    .stats-card:nth-child(3) .neon-text {{
        color: #f0e68c; /* Khaki */
    }}
    
    .fun-fact {{
        border: 2px solid #87cefa;
        color: #e6f7ff;
    }}
    </style>
    """,
    'red-orange-grey': lambda: f"""
    <style>
    .stApp {{
        background: 
                    url("data:image/jpg;base64,{get_base64_encoded_image(get_image_path('orange.png'))}") no-repeat center center fixed;
        background-size: cover;
    }}
    
    /* Red-Orange-Grey theme with warmer accents */
    h1, h2, h3, h4, h5, h6 {{ 
        color: #ff6b35; /* Bright orange headings */ 
        font-family: 'Impact', sans-serif; 
        text-shadow: 0 0 10px rgba(255, 107, 53, 0.7);
    }}
    
    p, li, span {{ 
        color: #f8f8f8; /* Off-white for text */
    }}
    
    .stButton>button {{ 
        background-color: rgba(191, 49, 0, 0.7); 
        color: #f8f8f8; 
        border: 1px solid #bf3100;
    }}
    
    .stButton>button:hover {{
        background-color: rgba(230, 57, 0, 0.8);
        box-shadow: 0 0 10px rgba(230, 57, 0, 0.7);
    }}
    
    .stButton>button[kind="primary"] {{ 
        background-color: #e63900; 
        border-color: #e63900; 
    }}
    
    .stDataFrame, .stMetric, .stExpander, .stSelectbox, .stMultiSelect, .stFileUploader, .stPlotlyChart, .stPyplot {{
        background-color: rgba(51, 51, 51, 0.7);
        border: 1px solid rgba(255, 107, 53, 0.5);
    }}
    
    .neon-text {{
        color: #ffbf69; /* Light orange */
        text-shadow: 0 0 5px #ffbf69, 0 0 10px #ff9e40, 0 0 20px #ff6b35;
    }}
    
    .title-text {{
        color: #ff6b35; /* Bright orange */
        text-shadow: 0 0 5px #ff6b35, 0 0 10px #e63900, 0 0 20px #bf3100;
    }}
    
    .features-box {{
        border: 2px solid #e63900;
    }}
    
    .feature-item:nth-child(odd) {{
        color: #ffbf69; /* Light orange */
    }}
    
    .feature-item:nth-child(even) {{
        color: #ff9e40; /* Medium orange */
    }}
    
    .stats-card {{
        border: 2px solid #e63900;
    }}
    
    .stats-card:nth-child(1) .neon-text {{
        color: #ff9e40; /* Medium orange */
    }}
    
    .stats-card:nth-child(2) .neon-text {{
        color: #ff6b35; /* Bright orange */
    }}
    
    .stats-card:nth-child(3) .neon-text {{
        color: #e63900; /* Dark orange/red */
    }}
    
    .fun-fact {{
        border: 2px solid #ff9e40;
        color: #ffefdb;
    }}
    </style>
    """
    }
    
    # Define get_base64_encoded_image helper function
    def get_base64_encoded_image(image_path):
        if image_path is None or not os.path.isfile(image_path):
            return None
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    
    # Define base CSS
    base_css = """
    <style>
    /* Base styles for all themes */
    .stApp {
        color: #e0e0e0;
    }
    
    /* Common component styles */
    h1, h2, h3, h4, h5, h6 { font-family: 'Arial', sans-serif; text-shadow: 0 0 5px rgba(98, 0, 234, 0.5); }
    .stButton>button { background-color: #3a3a3a; color: #ffffff; border: 1px solid #555555; border-radius: 5px; transition: all 0.3s ease; }
    .stButton>button:hover { background-color: #4a4a4a; border-color: #777777; box-shadow: 0 0 10px rgba(98, 0, 234, 0.5); }
    .stButton>button[kind="primary"] { border-radius: 5px; }
    .stDataFrame { background-color: #2a2a2a; border: 1px solid #444444; border-radius: 5px; }
    .stMetric { background-color: #2a2a2a; border: 1px solid #444444; border-radius: 5px; padding: 10px; }
    .stExpander { background-color: #2a2a2a; border: 1px solid #444444; border-radius: 5px; }
    .stSelectbox, .stMultiSelect { background-color: #2a2a2a; color: #e0e0e0; border: 1px solid #444444; border-radius: 5px; }
    .stFileUploader { background-color: #2a2a2a; border: 1px solid #444444; border-radius: 5px; }
    .stPlotlyChart, .stPyplot { background-color: #2a2a2a; border: 1px solid #444444; border-radius: 5px; padding: 10px; }
    .stNumberInput, .stSlider { background-color: #2a2a2a; color: #e0e0e0; border: 1px solid #444444; border-radius: 5px; }
    .stInfo, .stSuccess, .stWarning, .stError { border-radius: 5px; border: 1px solid; }
    .stInfo { background-color: #263238; border-color: #4f6b75; color: #b0bec5; }
    .stSuccess { background-color: #1b3a2f; border-color: #2e7d32; color: #a5d6a7; }
    .stWarning { background-color: #4e342e; border-color: #d81b60; color: #ffccbc; }
    .stError { background-color: #3e2723; border-color: #d32f2f; color: #ef9a9a; }
    .stDownloadButton>button { background-color: #3a3a3a; color: #ffffff; border: 1px solid #555555; border-radius: 5px; }
    .stDownloadButton>button:hover { background-color: #4a4a4a; border-color: #777777; }
    .center-gif { display: flex; justify-content: center; align-items: center; margin: 20px 0; }
    </style>
    """
    
    # Apply the base CSS
    st.markdown(base_css, unsafe_allow_html=True)

    # Apply the theme-specific CSS
    current_theme = st.session_state.theme
    if current_theme in theme_css:
        # Now theme_css is defined before we try to use it
        css_with_image = theme_css[current_theme]()
        st.markdown(css_with_image, unsafe_allow_html=True)

# Helper function to clean numeric columns
def clean_numeric_columns(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)

                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception as e:
                st.warning(f"Could not convert column {col} to numeric: {str(e)}")
    return df

# Helper function to check if a series is continuous or categorical
def is_continuous(series):
    if pd.api.types.is_numeric_dtype(series):
        unique_values = len(series.unique())
        return unique_values > 10
    return False

# Helper function to fetch data from yfinance with retry logic and caching
@st.cache_data
def fetch_yfinance_data(symbol, start_date, end_date, _cache_key=None):
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_message(match='Too Many Requests')
    )
    def fetch():
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                st.error(f"No data found for symbol {symbol} in the specified date range. Suggested symbols: AAPL, TSLA, MSFT.")
                return None
            
            df = df.reset_index()
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            return df
        except Exception as e:
            st.error(f"Error fetching data from yfinance: {str(e)}. Suggested symbols: AAPL, TSLA, MSFT.")
            return None
    
    return fetch()

# Helper function to fetch current price
def fetch_current_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        current_data = stock.info
        current_price = current_data.get('regularMarketPrice', current_data.get('currentPrice'))
        if current_price is None:
            return None
        return current_price
    except Exception as e:
        st.warning(f"Could not fetch current price for {symbol}: {str(e)}")
        return None

# Add CSS for welcome_step enhancements
st.markdown("""
    <style>
    /* Particle background effect */
    .particles {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: 0;
    }
    .particle {
        position: absolute;
        background: #E0FFFF;
        border-radius: 50%;
        opacity: 0.5;
        animation: float 10s infinite;
    }
    @keyframes float {
        0% { transform: translateY(0); opacity: 0.5; }
        50% { opacity: 0.8; }
        100% { transform: translateY(-100vh); opacity: 0; }
    }
    
    /* Neon white text with glow */
    .neon-text {
        color: #E0FFFF;
        text-shadow: 0 0 5px #E0FFFF, 0 0 10px #E0FFFF, 0 0 20px #6200ea;
        font-family: 'Arial', sans-serif;
    }
    
    /* Flashing title animation */
    .title-text {
        font-size: 64px;
        font-weight: bold;
        line-height: 1.2;
        text-align: left;
        animation: neon-flash 2s infinite;
    }
    @keyframes neon-flash {
        0%, 100% { text-shadow: 0 0 5px #E0FFFF, 0 0 10px #E0FFFF, 0 0 20px #6200ea; }
        50% { text-shadow: 0 0 10px #E0FFFF, 0 0 20px #E0FFFF, 0 0 40px #6200ea; }
    }
    
    /* Features box styling with hover effect */
    .features-box {
        background-color: rgba(0, 0, 0, 0.7);
        border: 2px solid #6200ea;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .features-box:hover {
        transform: scale(1.02);
        box-shadow: 0 0 15px #E0FFFF;
    }
    
    /* Feature list items */
    .feature-item {
        font-size: 18px;
        margin-bottom: 10px;
    }
    
    /* Contact info */
    .contact-info {
        display: flex;
        justify-content: space-between;
        margin-top: 30px;
        font-size: 16px;
    }
    .contact-info img {
        margin-right: 10px;
    }
    
    /* Get Started button with hover effect */
    .stButton>button[kind="primary"] {
        background-color: #6200ea;
        color: #E0FFFF;
        border: 2px solid #E0FFFF;
        border-radius: 10px;
        font-size: 18px;
        padding: 10px 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #7c4dff;
        border-color: #E0FFFF;
        transform: scale(1.05);
        box-shadow: 0 0 15px #E0FFFF;
    }
    
    /* Stock ticker with marquee effect */
    .stock-ticker {
        background-color: #2a2a2a;
        padding: 10px;
        border-radius: 5px;
        font-size: 16px;
        margin-top: 20px;
        text-align: center;
        overflow: hidden;
        white-space: nowrap;
    }
    .ticker-content {
        display: inline-block;
        animation: marquee 15s linear infinite;
    }
    @keyframes marquee {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
    
    /* Stats cards */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin-top: 30px;
    }
    .stats-card {
        background-color: rgba(0, 0, 0, 0.7);
        border: 2px solid #6200ea;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        width: 30%;
    }
    .stats-card:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px #E0FFFF;
    }
    
    /* GIF with hover zoom */
    .gif-container img {
        transition: transform 0.3s ease;
    }
    .gif-container img:hover {
        transform: scale(1.1);
    }
    
    /* Testimonial section */
    .testimonial {
        font-style: italic;
        font-size: 16px;
        color: #b0bec5;
        text-align: center;
        margin-top: 30px;
        border-left: 4px solid #6200ea;
        padding-left: 20px;
    }
    
    /* Fun fact pop-up */
    .fun-fact {
        background-color: #2a2a2a;
        border: 2px solid #E0FFFF;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        margin-top: 20px;
        animation: pop-up 0.5s ease;
    }
    @keyframes pop-up {
        0% { transform: scale(0); }
        100% { transform: scale(1); }
    }
    </style>
""", unsafe_allow_html=True)

# Particle effect (client-side JavaScript)
st.markdown("""
    <div class="particles" id="particles"></div>
    <script>
        function createParticle() {
            const particle = document.createElement('div');
            particle.classList.add('particle');
            particle.style.width = `${Math.random() * 5 + 2}px`;
            particle.style.height = particle.style.width;
            particle.style.left = `${Math.random() * 100}vw`;
            particle.style.top = `${Math.random() * 100}vh`;
            particle.style.animationDuration = `${Math.random() * 5 + 5}s`;
            document.getElementById('particles').appendChild(particle);
            setTimeout(() => particle.remove(), 10000);
        }
        for (let i = 0; i < 50; i++) {
            setTimeout(createParticle, i * 200);
        }
        setInterval(createParticle, 500);
    </script>
""", unsafe_allow_html=True)

# Welcome Interface (Updated with enhancements)
def welcome_step():
    # Main container for the page
    with st.container():
        # Top row: App Name and Get Started button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("""
                <div class="neon-text" style="font-size: 24px; font-weight: bold;">
                 🚀 From Data to Decisions — Instantly
                </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("Get Started", key="get_started", type="primary"):
                # Play a sound effect on button click
                st.markdown("""
                    <audio autoplay>
                        <source src="https://www.soundjay.com/buttons/beep-01a.mp3" type="audio/mpeg">
                    </audio>
                """, unsafe_allow_html=True)
                st.session_state.pipeline['current_step'] = 1
                st.rerun()

        # Main content row: Title, Image, Features
        col_left, col_right = st.columns([1, 1])
        
        # Left column: Title and Features
        with col_left:
            # Main title with flashing neon effect
            st.markdown("""
                <div class="neon-text title-text">
                    STOCK ML<br>PIPELINE
                </div>
            """, unsafe_allow_html=True)

            # Key Features with fun descriptions
            st.markdown("""
                <div class="features-box">
                    <div class="neon-text" style="font-size: 24px; font-weight: bold; margin-bottom: 15px;">
                        Key Features
                    </div>
                    <div class="neon-text feature-item">📈 Watch Stocks Soar in Real-Time!</div>
                    <div class="neon-text feature-item">🧙‍♂️ Magic Preprocessing Wizardry!</div>
                    <div class="neon-text feature-item">🤖 Train ML Models Like a Pro!</div>
                    <div class="neon-text feature-item">🎨 Eye-Popping Visualizations!</div>
                    <div class="neon-text feature-item">⚡ Lightning-Fast Analysis!</div>
                    <div class="neon-text feature-item">🌟 Become a Stock Market Superhero!</div>
                </div>
            """, unsafe_allow_html=True)

        # Right column: GIF with hover zoom
        with col_right:
            st.markdown("""
                <div class="gif-container">
                    <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExY2FiZzB5MHNsODljeGNwZWdsZnRsdHM0cW85OW95ZG43a2pybnJnayZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/QEArKBwKJm12r8t7w6/giphy.gif" 
                         alt="Stock Market GIF" width="1000" 
                         style="border-radius: 10px; box-shadow: 0 4px 15px rgba(98, 0, 234, 0.3);"/>
                </div>
            """, unsafe_allow_html=True)

        # Interactive Features: Animated Stock Ticker
        try:
            with st.spinner("Fetching real-time stock data..."):
                stock_data = {}
                for symbol in ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN"]:
                    stock = yf.Ticker(symbol)
                    current_price = stock.info.get("regularMarketPrice", stock.info.get("currentPrice", "N/A"))
                    stock_data[symbol] = current_price
                ticker_text = " | ".join([f"{symbol}: ${price:.2f}" if isinstance(price, (int, float)) else f"{symbol}: N/A" for symbol, price in stock_data.items()])
                st.markdown(f"""
                    <div class="stock-ticker">
                        <div class="neon-text ticker-content">
                            Live Stock Prices: {ticker_text}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f"""
                <div class="stock-ticker">
                    <div class="neon-text ticker-content">
                        Live Stock Prices: Unable to fetch data
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Stats Cards with Animated Counters
        st.markdown("""
            <div class="stats-container">
                <div class="stats-card">
                    <div class="neon-text" style="font-size: 24px; font-weight: bold;">10K+</div>
                    <div class="neon-text">Active Users</div>
                </div>
                <div class="stats-card">
                    <div class="neon-text" style="font-size: 24px; font-weight: bold;">500K+</div>
                    <div class="neon-text">Models Trained</div>
                </div>
                <div class="stats-card">
                    <div class="neon-text" style="font-size: 24px; font-weight: bold;">1M+</div>
                    <div class="neon-text">Data Points Analyzed</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Fun Fact Pop-Up (Random Stock Market Trivia)
        fun_facts = [
            "The New York Stock Exchange was founded in 1792 under a buttonwood tree!",
            "The most expensive stock in the world is Berkshire Hathaway, trading at over $400,000 per share!",
            "The term 'bull market' comes from bulls thrusting their horns upward, symbolizing rising prices!",
            "The shortest stock market crash in history lasted just one day—October 19, 1987, known as Black Monday!"
        ]
        selected_fact = random.choice(fun_facts)
        st.markdown(f"""
            <div class="fun-fact neon-text">
                💡 Did You Know? {selected_fact}
            </div>
        """, unsafe_allow_html=True)

        # Testimonial Section
        st.markdown("""
            <div class="testimonial">
                "This app made stock analysis so much fun! I love the live ticker and stunning visuals!" 
                <br>— Alex R., Stock Enthusiast
            </div>
        """, unsafe_allow_html=True)


# Step 1: Load Data
def load_data_step():
    st.header("Step 1: Load Data 📊")
    
    data_option = st.radio("Select data source:", ("Upload your own data", "Fetch data from yfinance"))
    
    if data_option == "Upload your own data":
        uploaded_file = st.file_uploader("Upload your stock data (CSV or Excel)", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                df = clean_numeric_columns(df)
                
                st.session_state.pipeline['df'] = df
                st.session_state.pipeline['data_loaded'] = True
                st.session_state.pipeline['last_symbol'] = None
                
                st.success("✅ Data loaded successfully!")
                
                with st.expander("View Raw Data"):
                    st.dataframe(df)
                
                with st.expander("Data Information"):
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    st.text(buffer.getvalue())
                    st.write("Descriptive Statistics:")
                    st.dataframe(df.describe())
                
                st.session_state.pipeline['current_step'] = 2
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
        else:
            st.info("ℹ️ Please upload a file to get started.")
    
    else:
        st.subheader("Fetch Data from yfinance")
        st.markdown("Enter the stock symbol and date range to fetch data from yfinance.")
        
        symbol = st.text_input("Enter stock symbol (e.g., AAPL)", value="AAPL")
        start_date = st.date_input("Start Date", value=datetime.date(2024, 1, 1))
        end_date = st.date_input("End Date", value=datetime.date(2024, 12, 31))
        
        if st.button("Fetch Data"):
            if symbol and start_date and end_date:
                with st.spinner("Fetching data from yfinance..."):
                    cache_key = f"{symbol}_{start_date}_{end_date}"
                    df = fetch_yfinance_data(symbol.upper(), start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), _cache_key=cache_key)
                
                if df is not None and not df.empty:
                    current_price = fetch_current_price(symbol.upper())
                    if current_price:
                        st.success(f"Current Price of {symbol.upper()}: ${current_price:.2f}")
                        st.session_state.pipeline['current_price'] = current_price
                    
                    st.session_state.pipeline['df'] = df
                    st.session_state.pipeline['data_loaded'] = True
                    st.session_state.pipeline['last_symbol'] = symbol.upper()
                    
                    st.success("✅ Data fetched successfully from yfinance!")
                    
                    with st.expander("View Raw Data"):
                        st.dataframe(df)
                    
                    with st.expander("Data Information"):
                        buffer = io.StringIO()
                        df.info(buf=buffer)
                        st.text(buffer.getvalue())
                        st.write("Descriptive Statistics:")
                        st.dataframe(df.describe())
                    
                    st.session_state.pipeline['current_step'] = 2
                    st.rerun()
            else:
                st.warning("Please provide a stock symbol and date range.")

# Step 2: Preprocessing
def preprocessing_step():
    st.header("Step 2: Preprocessing 🛠️")
    
    if not st.session_state.pipeline['data_loaded']:
        st.warning("Please load data first!")
        return
    
    df = st.session_state.pipeline['df'].copy()
    
    st.subheader("Missing Values")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        st.dataframe(missing_values[missing_values > 0].to_frame(name="Missing Count"))
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        st.success("Missing values imputed with mean values")
    else:
        st.success("No missing values found")
    
    st.session_state.pipeline['df_processed'] = df
    st.session_state.pipeline['preprocessed'] = True
    
    with st.expander("View Processed Data"):
        st.dataframe(df)
    
    if st.button("Continue to Feature Engineering"):
        st.session_state.pipeline['current_step'] = 3
        st.rerun()

# Step 3: Feature Engineering
def feature_engineering_step():
    st.header("Step 3: Feature Engineering 📐")
    
    if not st.session_state.pipeline['preprocessed']:
        st.warning("Please complete preprocessing first!")
        return
    
    if st.session_state.pipeline['df_processed'] is None:
        st.error("No processed data found!")
        return
    
    df = st.session_state.pipeline['df_processed'].copy()
    
    st.subheader("Advanced Feature Engineering")
    if 'Close' in df.columns:
        window = st.slider("Select Moving Average window (days)", 5, 50, 20)
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'MA_{window}'] = df[f'MA_{window}'].fillna(df['Close'])
        st.success(f"Added {window}-day Moving Average as a feature!")
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        st.error("No numeric columns found for analysis!")
        return
    
    st.subheader("Feature Selection")
    target = st.selectbox("Select target variable (y)", numeric_cols)
    features = st.multiselect("Select feature variables (X)", [c for c in numeric_cols if c != target])
    
    if not features:
        st.warning("Please select at least one feature!")
        return
    
    st.subheader("Feature Scaling")
    scale_features = st.checkbox("Scale features (Standardization)", value=True)
    
    if scale_features:
        try:
            scaler = StandardScaler()
            df[features] = scaler.fit_transform(df[features])
            st.success("Features successfully scaled!")
        except Exception as e:
            st.error(f"Error scaling features: {str(e)}")
    
    st.subheader("Feature Correlation")
    try:
        corr_matrix = df[features + [target]].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title='Feature Correlation Matrix',
            width=600,
            height=500
        )
        fig.update_layout(
    plot_bgcolor="rgba(30,30,30,0.9)",  # Graph background (dark translucent grey)
    paper_bgcolor="rgba(30,30,30,0.9)", # Outer chart area
    font=dict(color="white"),          # Font visibility
    legend=dict(bgcolor="rgba(0,0,0,0.5)")  # Legend box clarity
)

        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Could not create correlation matrix: {str(e)}")
    
    st.session_state.pipeline['target'] = target
    st.session_state.pipeline['features'] = features
    st.session_state.pipeline['df_features'] = df
    st.session_state.pipeline['features_engineered'] = True
    
    if st.button("Continue to Train/Test Split"):
        st.session_state.pipeline['current_step'] = 4
        st.rerun()

# Step 4: Train/Test Split
def train_test_split_step():
    st.header("Step 4: Train/Test Split ✂️")
    
    if not st.session_state.pipeline['features_engineered']:
        st.warning("Please complete feature engineering first!")
        return
    
    if st.session_state.pipeline['target'] is None or st.session_state.pipeline['features'] is None:
        st.error("Target or features not selected!")
        return
    
    df = st.session_state.pipeline['df_features']
    target = st.session_state.pipeline['target']
    features = st.session_state.pipeline['features']
    
    st.subheader("Split Configuration")
    test_size = st.slider("Test set size (%)", 10, 40, 20)
    random_state = st.number_input("Random state", 0, 100, 42)
    
    try:
        X = df[features]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size/100, 
            random_state=random_state
        )
        
        st.session_state.pipeline['X_train'] = X_train
        st.session_state.pipeline['X_test'] = X_test
        st.session_state.pipeline['y_train'] = y_train
        st.session_state.pipeline['y_test'] = y_test
        st.session_state.pipeline['data_split'] = True
        
        st.subheader("Data Split Visualization")
        split_df = pd.DataFrame({
            'Set': ['Training', 'Testing'],
            'Size': [len(X_train), len(X_test)]
        })
        fig = px.pie(
            split_df,
            names='Set',
            values='Size',
            title='Training vs Testing Split',
            width=400,
            height=400,
            color_discrete_sequence=['#1E90FF', '#FF4500']
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0'
        )
        st.plotly_chart(fig)
        
        st.success("Train/test split completed!")
        
        if st.button("Continue to Model Training"):
            st.session_state.pipeline['current_step'] = 5
            st.rerun()
            
    except Exception as e:
        st.error(f"Error during train/test split: {str(e)}")

# Step 5: Model Training
def model_training_step():
    st.header("Step 5: Model Training 🤖")
    
    if not st.session_state.pipeline['data_split']:
        st.warning("Please complete train/test split first!")
        return
    
    X_train = st.session_state.pipeline['X_train']
    y_train = st.session_state.pipeline['y_train']
    
    st.subheader("Model Configuration")
    model_type = st.selectbox("Select Model to Train", ["Linear Regression", "Logistic Regression", "K-Nearest Neighbors"])
    
    target_is_continuous = is_continuous(y_train)
    
    if model_type == "Linear Regression" and not target_is_continuous:
        st.warning("""
            ⚠️ Linear Regression expects a continuous target variable (e.g., stock prices). 
            Your target variable appears to be categorical. Consider discretizing it in preprocessing 
            or selecting a different model like Logistic Regression for classification tasks.
        """)
        return
    elif model_type == "Logistic Regression" and target_is_continuous:
        st.warning("""
            ⚠️ Logistic Regression expects a categorical target variable (e.g., buy/sell, 0/1). 
            Your target variable appears to be continuous. You can discretize it in the preprocessing step 
            (e.g., convert to categories like 'high'/'low') or select a different model like Linear Regression.
        """)
        return
    elif model_type == "K-Nearest Neighbors" and not target_is_continuous:
        st.info("K-Nearest Neighbors will be used as a classifier for categorical target.")
    
    models = {}
    if model_type == "Linear Regression":
        models[model_type] = LinearRegression()
    elif model_type == "Logistic Regression":
        models[model_type] = LogisticRegression(max_iter=1000)
    elif model_type == "K-Nearest Neighbors":
        n_neighbors = st.number_input("Number of neighbors (k)", min_value=1, max_value=50, value=5, key="n_neighbors")
        if target_is_continuous:
            models[model_type] = KNeighborsRegressor(n_neighbors=n_neighbors)
        else:
            models[model_type] = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    with st.spinner("Training model..."):
        try:
            model = models[model_type]
            model.fit(X_train, y_train)
            models[model_type] = model
            
            st.session_state.pipeline['models'] = models
            st.session_state.pipeline['model_trained'] = True
            st.success("Model training completed!")
            
            st.subheader("Model Details")
            st.write(f"**{model_type}**")
            if model_type in ["Linear Regression", "Logistic Regression"] and hasattr(model, 'coef_'):
                coef_df = pd.DataFrame({
                    'Feature': ['Intercept'] + st.session_state.pipeline['features'],
                    'Coefficient': [model.intercept_] + list(model.coef_)
                })
                st.dataframe(coef_df)
            elif model_type == "K-Nearest Neighbors":
                st.write(f"Number of neighbors: {n_neighbors}")
                st.write("KNN does not provide feature coefficients, but relies on distance-based predictions.")
            
            if st.button("Continue to Evaluation"):
                st.session_state.pipeline['current_step'] = 6
                st.rerun()
                
        except Exception as e:
            st.error(f"Error during model training: {str(e)}")

# Step 6: Evaluation
def evaluation_step():
    st.header("Step 6: Model Evaluation 📊")
    
    if not st.session_state.pipeline['model_trained']:
        st.warning("Please train the model first!")
        return
    
    models = st.session_state.pipeline['models']
    X_test = st.session_state.pipeline['X_test']
    y_test = st.session_state.pipeline['y_test']
    
    y_preds = {}
    try:
        for model_type, model in models.items():
            y_pred = model.predict(X_test)
            y_preds[model_type] = y_pred
        
        st.session_state.pipeline['y_preds'] = y_preds
        
        st.subheader("Model Performance Metrics")
        metrics_df = pd.DataFrame(columns=['Model', 'RMSE', 'R²'])
        for model_type, y_pred in y_preds.items():
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            metrics_df = pd.concat([metrics_df, pd.DataFrame({
                'Model': [model_type],
                'RMSE': [rmse],
                'R²': [r2]
            })], ignore_index=True)
        
        st.dataframe(metrics_df.style.format({'RMSE': '{:.4f}', 'R²': '{:.4f}'}))
        
        st.subheader("Actual vs Predicted Values")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test,
            y=y_test,
            mode='lines',
            name='Ideal Fit',
            line=dict(color='black', dash='dash')
        ))
        for model_type, y_pred in y_preds.items():
            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name=f'{model_type} Predictions',
                marker=dict(size=8)
            ))
        
        fig.update_layout(
    title='Actual vs Predicted Values',
    xaxis_title='Actual',
    yaxis_title='Predicted',
    paper_bgcolor='rgba(15, 15, 15, 0.8)',
    plot_bgcolor='rgba(25, 25, 25, 0.8)',
    font_color='#e0e0e0',
    legend=dict(
        bgcolor='rgba(50,50,50,0.8)',
        bordercolor='rgba(255,255,255,0.2)'
    )
)

        st.plotly_chart(fig)
        
        st.session_state.pipeline['model_evaluated'] = True
        
        if st.button("Continue to Results Visualization"):
            st.session_state.pipeline['current_step'] = 7
            st.rerun()
            
    except Exception as e:
        st.error(f"Error during model evaluation: {str(e)}")

# Step 7: Results Visualization
def results_visualization_step():
    st.header("Step 7: Results Visualization 📈")
    
    if not st.session_state.pipeline['model_evaluated']:
        st.warning("Please evaluate the model first!")
        return
    
    df = st.session_state.pipeline['df_features']
    target = st.session_state.pipeline['target']
    features = st.session_state.pipeline['features']
    y_test = st.session_state.pipeline['y_test']
    y_preds = st.session_state.pipeline['y_preds']
    
    st.subheader("Interactive Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Feature Importance", "Time Series", "Model Comparison", "Prediction"])
    
    # Tab 1: Feature Importance
    with tab1:
        st.subheader("Feature Importance")
        try:
            model_type = list(st.session_state.pipeline['models'].keys())[0]
            model = st.session_state.pipeline['models'][model_type]
            
            if model_type in ["Linear Regression", "Logistic Regression"] and hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    importance_df,
                    x='Feature',
                    y='Importance',
                    title=f'Feature Importance - {model_type}',
                    color='Importance',
                    color_continuous_scale='Viridis'
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#e0e0e0',
                    xaxis={'categoryorder':'total descending'}
                )
                st.plotly_chart(fig)
            else:
                st.info(f"Feature importance visualization not available for {model_type}")
        except Exception as e:
            st.error(f"Error creating feature importance visualization: {str(e)}")
    
    # Tab 2: Time Series
    with tab2:
        st.subheader("Time Series Visualization")
        try:
            if 'Date' in df.columns and 'Close' in df.columns:
                date_col = df['Date'].copy()
                close_col = df['Close'].copy()
                
                model_type = list(st.session_state.pipeline['models'].keys())[0]
                
                if st.checkbox("Show prediction in context", value=True):
                    fig = go.Figure()
                    
                    # Original data
                    fig.add_trace(go.Scatter(
                        x=date_col,
                        y=close_col,
                        mode='lines',
                        name='Actual Price',
                        line=dict(color='#4C78A8', width=2)
                    ))
                    
                    # Test set indices
                    test_indices = y_test.index
                    
                    # Model predictions
                    fig.add_trace(go.Scatter(
                        x=date_col.iloc[test_indices],
                        y=y_preds[model_type],
                        mode='markers',
                        name=f'{model_type} Predictions',
                        marker=dict(size=8, color='#E45756')
                    ))
                    
                    fig.update_layout(
                        title='Stock Price Prediction Over Time',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#e0e0e0',
                        legend=dict(
                            bgcolor='rgba(50,50,50,0.8)',
                            bordercolor='rgba(255,255,255,0.2)'
                        )
                    )
                    st.plotly_chart(fig)
                    
                    # If we have current price, show potential future prediction
                    if st.session_state.pipeline['current_price'] is not None and st.session_state.pipeline['last_symbol'] is not None:
                        st.subheader(f"Predicted Next Day Price for {st.session_state.pipeline['last_symbol']}")
                        
                        latest_features = df[features].iloc[-1].values.reshape(1, -1)
                        try:
                            future_pred = st.session_state.pipeline['models'][model_type].predict(latest_features)[0]
                            current = st.session_state.pipeline['current_price']
                            change_pct = ((future_pred - current) / current) * 100
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Current Price", f"${current:.2f}")
                            with col2:
                                st.metric("Predicted Price", f"${future_pred:.2f}")
                            with col3:
                                st.metric("Predicted Change", f"{change_pct:.2f}%", delta=f"{change_pct:.2f}%")
                                
                        except Exception as e:
                            st.warning(f"Could not generate next day prediction: {str(e)}")
                else:
                    st.info("Enable the checkbox to see prediction visualization")
            else:
                st.info("Time series visualization requires 'Date' and 'Close' columns")
        except Exception as e:
            st.error(f"Error creating time series visualization: {str(e)}")
    
    # Tab 3: Model Comparison
    with tab3:
        st.subheader("Model Performance Comparison")
        try:
            metrics_df = pd.DataFrame(columns=['Model', 'RMSE', 'R²'])
            for model_type, y_pred in y_preds.items():
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                metrics_df = pd.concat([metrics_df, pd.DataFrame({
                    'Model': [model_type],
                    'RMSE': [rmse],
                    'R²': [r2]
                })], ignore_index=True)
            
            fig1 = px.bar(
                metrics_df,
                x='Model',
                y='RMSE',
                title='RMSE by Model (Lower is Better)',
                color='Model'
            )
            fig1.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0'
            )
            
            fig2 = px.bar(
                metrics_df,
                x='Model',
                y='R²',
                title='R² by Model (Higher is Better)',
                color='Model'
            )
            fig2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0'
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig1)
            with col2:
                st.plotly_chart(fig2)
                
        except Exception as e:
            st.error(f"Error creating model comparison visualization: {str(e)}")
    
    # Tab 4: Interactive Prediction
    with tab4:
        st.subheader("Interactive Stock Price Prediction")
        try:
            model_type = list(st.session_state.pipeline['models'].keys())[0]
            model = st.session_state.pipeline['models'][model_type]
            
            st.write("Adjust the feature values to see how they affect the prediction:")
            
            input_features = {}
            for feature in features:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                mean_val = float(df[feature].mean())
                
                input_features[feature] = st.slider(
                    f"{feature}", 
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    step=(max_val - min_val) / 100
                )
            
            input_df = pd.DataFrame([input_features])
            prediction = model.predict(input_df)[0]
            
            st.subheader("Prediction Result")
            st.metric("Predicted Value", f"{prediction:.2f}")
            
            # Create a radar chart for input features
            radar_df = pd.DataFrame({
                'Feature': list(input_features.keys()),
                'Value': list(input_features.values()),
                'Min': [float(df[feat].min()) for feat in input_features.keys()],
                'Max': [float(df[feat].max()) for feat in input_features.keys()]
            })
            
            # Normalize values for radar chart
            radar_df['NormalizedValue'] = (radar_df['Value'] - radar_df['Min']) / (radar_df['Max'] - radar_df['Min'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=radar_df['NormalizedValue'],
                theta=radar_df['Feature'],
                fill='toself',
                name='Feature Values'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Feature Values (Normalized)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e0e0e0'
            )
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error in interactive prediction: {str(e)}")
    
    st.session_state.pipeline['results_visualized'] = True
    
    if st.button("Restart Pipeline"):
        # Reset session state but maintain theme
        current_theme = st.session_state.theme
        st.session_state.pipeline = {
            'current_step': 0,
            'data_loaded': False,
            'preprocessed': False,
            'features_engineered': False,
            'data_split': False,
            'model_trained': False,
            'model_evaluated': False,
            'results_visualized': False,
            'df': None,
            'df_processed': None,
            'target': None,
            'features': None,
            'X_train': None,
            'X_test': None,
            'y_train': None,
            'y_test': None,
            'models': {},
            'y_preds': {},
            'current_price': None,
            'last_symbol': None,
        }
        st.session_state.theme = current_theme
        st.rerun()

# Main App Logic
def main():
    # Apply theme CSS
    apply_theme_css()
    
    # Create sidebar
    with st.sidebar:
        st.title("Stock ML Pipeline")
        st.markdown("**Navigate through the ML pipeline steps**")
        
        # Add theme selector
        theme_selector()
        
        # Navigation buttons
        steps = [
            "Welcome",
            "1. Load Data",
            "2. Preprocessing",
            "3. Feature Engineering",
            "4. Train/Test Split",
            "5. Model Training",
            "6. Evaluation",
            "7. Results Visualization"
        ]
        
        for i, step in enumerate(steps):
            if st.button(step, key=f"nav_{i}"):
                st.session_state.pipeline['current_step'] = i
                st.rerun()
        
        # Show current step indicator
        current_step = st.session_state.pipeline['current_step']
        st.success(f"Current step: {steps[current_step]}")
        
        # Display pipeline state
        with st.expander("Pipeline State"):
            st.json({
                k: v for k, v in st.session_state.pipeline.items() 
                if k not in ['df', 'df_processed', 'X_train', 'X_test', 'y_train', 'y_test', 'models', 'y_preds', 'df_features']
            })
    
    # Display the current step
    if st.session_state.pipeline['current_step'] == 0:
        welcome_step()
    elif st.session_state.pipeline['current_step'] == 1:
        load_data_step()
    elif st.session_state.pipeline['current_step'] == 2:
        preprocessing_step()
    elif st.session_state.pipeline['current_step'] == 3:
        feature_engineering_step()
    elif st.session_state.pipeline['current_step'] == 4:
        train_test_split_step()
    elif st.session_state.pipeline['current_step'] == 5:
        model_training_step()
    elif st.session_state.pipeline['current_step'] == 6:
        evaluation_step()
    elif st.session_state.pipeline['current_step'] == 7:
        results_visualization_step()

if __name__ == "__main__":
    init_session_state()
    main()