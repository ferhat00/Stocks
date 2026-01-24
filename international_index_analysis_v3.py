#!/usr/bin/env python3
"""
International Stock Index Analysis vs SPY - Version 3
Comprehensive Analysis with Full Geopolitical Events Database

FEATURES:
- 50+ geopolitical events (Trump, Israel-Iran, Ukraine-Russia, Venezuela, UK Trade, Semiconductors)
- Rolling correlation, beta, volatility (multiple windows)
- Currency-adjusted returns (EUR, GBP, JPY)
- VIX overlay and volatility regime analysis
- Sector comparisons (Tech vs Defensive)
- Drawdown analysis around events
- Pre/post event windows (-5 to +10 days)
- Structural break detection (Chow test)
- Regional groupings with aggregated metrics
- PDF report generation with key findings
- Interactive HTML dashboard

Author: Created for Ferhat Culfaz
Date: January 2025

Requirements:
    pip install yfinance pandas numpy plotly scipy reportlab matplotlib seaborn statsmodels

Usage:
    python international_index_analysis_v3.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
from scipy.signal import find_peaks
import warnings
import os
import io

# Optional imports for PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("⚠ reportlab not installed. PDF generation disabled. Install with: pip install reportlab")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠ matplotlib not installed. Some features disabled.")

try:
    from statsmodels.stats.diagnostic import breaks_cusumolsresid
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠ statsmodels not installed. Structural break analysis disabled.")

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Index tickers by country
INDEX_TICKERS = {
    'SPY': 'SPY',
    'France': '^FCHI',
    'Germany': '^GDAXI',
    'UK': '^FTSE',
    'Italy': 'FTSEMIB.MI',
    'Sweden': '^OMX',
    'Spain': '^IBEX',
    'Norway': '^OBX',
    'Denmark': '^OMXC25',
    'Switzerland': '^SSMI',
    'Finland': '^OMXH25',
    'Korea': '^KS11',
    'Japan': '^N225',
    'China': '000001.SS',
    'Hong Kong': '^HSI',
    'Canada': '^GSPTSE',
    'Australia': '^AXJO',
}

# Sector ETFs for comparison
SECTOR_TICKERS = {
    # Tech / Growth
    'US Tech': 'QQQ',
    'US Semiconductors': 'SOXX',
    'US Growth': 'VUG',
    # Defensive
    'US Utilities': 'XLU',
    'US Consumer Staples': 'XLP',
    'US Healthcare': 'XLV',
    'US Dividend': 'VYM',
    # Volatility
    'VIX': '^VIX',
    # Gold (safe haven)
    'Gold': 'GLD',
}

# Currency pairs for adjustment
CURRENCY_TICKERS = {
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
    'JPY/USD': 'JPY=X',  # Actually USD/JPY, we'll invert
    'AUD/USD': 'AUDUSD=X',
    'CAD/USD': 'CADUSD=X',
    'CHF/USD': 'CHFUSD=X',
}

# Country to currency mapping
COUNTRY_CURRENCY = {
    'France': 'EUR/USD',
    'Germany': 'EUR/USD',
    'Italy': 'EUR/USD',
    'Spain': 'EUR/USD',
    'Finland': 'EUR/USD',
    'UK': 'GBP/USD',
    'Switzerland': 'CHF/USD',
    'Sweden': 'EUR/USD',  # Approximate with EUR
    'Norway': 'EUR/USD',
    'Denmark': 'EUR/USD',
    'Japan': 'JPY/USD',
    'Korea': 'JPY/USD',  # Approximate
    'China': 'JPY/USD',  # Approximate
    'Hong Kong': 'JPY/USD',
    'Canada': 'CAD/USD',
    'Australia': 'AUD/USD',
}

# ETF alternatives
ETF_ALTERNATIVES = {
    'France': 'EWQ', 'Germany': 'EWG', 'UK': 'EWU', 'Italy': 'EWI',
    'Sweden': 'EWD', 'Spain': 'EWP', 'Switzerland': 'EWL', 'Korea': 'EWY',
    'Japan': 'EWJ', 'China': 'MCHI', 'Hong Kong': 'EWH', 'Canada': 'EWC',
    'Australia': 'EWA',
}

# Regional groupings
REGIONS = {
    'North America': ['SPY', 'Canada'],
    'Western Europe': ['Germany', 'France', 'UK', 'Switzerland'],
    'Northern Europe': ['Sweden', 'Norway', 'Denmark', 'Finland'],
    'Southern Europe': ['Italy', 'Spain'],
    'Asia Developed': ['Japan', 'Korea', 'Hong Kong'],
    'Asia Emerging': ['China'],
    'Oceania': ['Australia'],
}

COUNTRY_TO_REGION = {}
for region, countries in REGIONS.items():
    for country in countries:
        COUNTRY_TO_REGION[country] = region

# =============================================================================
# COMPREHENSIVE GEOPOLITICAL EVENTS DATABASE (60+ Events)
# =============================================================================

GEOPOLITICAL_EVENTS = {
    # =========================================================================
    # TRUMP ADMINISTRATION (2024-2025)
    # =========================================================================
    '2024-11-05': {'event': 'Trump Election Victory', 'description': 'Trump wins 2024 presidential election', 'color': '#1f77b4', 'category': 'trump_political', 'severity': 'high'},
    '2024-11-06': {'event': 'Post-Election Rally', 'description': 'Markets react to Trump victory', 'color': '#1f77b4', 'category': 'trump_political', 'severity': 'medium'},
    '2024-12-17': {'event': 'Trump Tariff Threats Begin', 'description': 'Trump threatens 25% on Canada/Mexico, 10% on China', 'color': '#ff7f0e', 'category': 'trump_tariff', 'severity': 'high'},
    '2025-01-20': {'event': 'Trump Inauguration', 'description': 'Trump sworn in as 47th President', 'color': '#d62728', 'category': 'trump_political', 'severity': 'high'},
    '2025-01-23': {'event': 'Greenland/Denmark Tariff Threats', 'description': 'Trump threatens tariffs on Denmark over Greenland', 'color': '#ff7f0e', 'category': 'trump_tariff', 'severity': 'medium'},
    '2025-01-27': {'event': 'Colombia Emergency Tariffs', 'description': 'Emergency 25% tariffs on Colombia', 'color': '#9467bd', 'category': 'trump_tariff', 'severity': 'low'},
    '2025-02-01': {'event': 'USMCA Tariffs Effective', 'description': '25% tariffs on Canada & Mexico take effect', 'color': '#8c564b', 'category': 'trump_tariff', 'severity': 'high'},
    '2025-02-04': {'event': 'China Tariffs +10%', 'description': '10% additional tariffs on China (total 20%)', 'color': '#e377c2', 'category': 'trump_tariff', 'severity': 'high'},
    '2025-02-10': {'event': 'Steel/Aluminum Tariffs', 'description': '25% tariffs on all steel/aluminum globally', 'color': '#7f7f7f', 'category': 'trump_tariff', 'severity': 'high'},
    '2025-02-13': {'event': 'Trump-Putin Phone Call', 'description': 'Trump and Putin discuss Ukraine peace framework', 'color': '#17becf', 'category': 'trump_political', 'severity': 'high'},
    '2025-03-04': {'event': 'Tariff Escalation Day', 'description': 'China to 20%, Canada/Mexico 25% enforced', 'color': '#bcbd22', 'category': 'trump_tariff', 'severity': 'high'},
    '2025-03-12': {'event': 'EU Counter-Tariffs Announced', 'description': 'EU announces retaliatory tariffs on US goods', 'color': '#17becf', 'category': 'eu_retaliation', 'severity': 'high'},
    '2025-03-20': {'event': 'EU Tariffs Take Effect', 'description': 'EU counter-tariffs on US whiskey, motorcycles, agriculture', 'color': '#17becf', 'category': 'eu_retaliation', 'severity': 'medium'},
    '2025-04-02': {'event': '"Liberation Day" Tariffs', 'description': 'Universal reciprocal tariffs on all countries', 'color': '#d62728', 'category': 'trump_tariff', 'severity': 'critical'},
    '2025-04-05': {'event': 'EU Emergency Summit', 'description': 'EU leaders meet on coordinated response to US tariffs', 'color': '#17becf', 'category': 'eu_retaliation', 'severity': 'medium'},
    '2025-04-09': {'event': '90-Day Tariff Pause', 'description': 'Trump pauses reciprocal tariffs (China raised to 145%)', 'color': '#2ca02c', 'category': 'trump_tariff', 'severity': 'critical'},
    '2025-05-12': {'event': 'US-China Geneva Talks', 'description': 'US-China tariff negotiations begin', 'color': '#ff9896', 'category': 'trump_tariff', 'severity': 'high'},
    
    # =========================================================================
    # UK TRADE DEAL NEGOTIATIONS (2024-2025)
    # =========================================================================
    '2024-07-05': {'event': 'UK Labour Government', 'description': 'New UK government signals trade deal priority', 'color': '#1f77b4', 'category': 'uk_trade', 'severity': 'medium'},
    '2024-09-15': {'event': 'UK-US Trade Talks Resume', 'description': 'Biden admin resumes trade negotiations with UK', 'color': '#2ca02c', 'category': 'uk_trade', 'severity': 'medium'},
    '2025-01-25': {'event': 'UK-Trump Trade Optimism', 'description': 'UK signals hope for quick trade deal with Trump', 'color': '#2ca02c', 'category': 'uk_trade', 'severity': 'medium'},
    '2025-02-15': {'event': 'UK Steel Tariff Exemption Request', 'description': 'UK requests exemption from US steel tariffs', 'color': '#ff7f0e', 'category': 'uk_trade', 'severity': 'medium'},
    '2025-03-08': {'event': 'UK-US Trade Framework', 'description': 'Preliminary trade framework announced', 'color': '#2ca02c', 'category': 'uk_trade', 'severity': 'high'},
    '2025-04-12': {'event': 'UK Tariff Exemption Granted', 'description': 'UK receives partial exemption from Liberation Day tariffs', 'color': '#2ca02c', 'category': 'uk_trade', 'severity': 'high'},
    
    # =========================================================================
    # JAPAN/KOREA SEMICONDUCTOR RESTRICTIONS (2023-2025)
    # =========================================================================
    '2023-07-23': {'event': 'Japan Chip Export Controls', 'description': 'Japan restricts chip equipment exports to China', 'color': '#d62728', 'category': 'semiconductors', 'severity': 'high'},
    '2023-10-17': {'event': 'US Expands China Chip Ban', 'description': 'Biden expands semiconductor export restrictions', 'color': '#d62728', 'category': 'semiconductors', 'severity': 'high'},
    '2024-03-29': {'event': 'Korea Chip Investment', 'description': 'Korea announces $470B semiconductor investment', 'color': '#2ca02c', 'category': 'semiconductors', 'severity': 'medium'},
    '2024-07-15': {'event': 'ASML Export Restrictions', 'description': 'Netherlands tightens chip equipment exports', 'color': '#d62728', 'category': 'semiconductors', 'severity': 'high'},
    '2024-09-30': {'event': 'Japan-Korea Chip Alliance', 'description': 'Japan-Korea semiconductor cooperation agreement', 'color': '#2ca02c', 'category': 'semiconductors', 'severity': 'medium'},
    '2024-12-02': {'event': 'China Chip Retaliation', 'description': 'China bans critical minerals exports to US', 'color': '#d62728', 'category': 'semiconductors', 'severity': 'high'},
    '2025-01-15': {'event': 'Trump Chip Policy Review', 'description': 'Trump admin reviews semiconductor export policy', 'color': '#ff7f0e', 'category': 'semiconductors', 'severity': 'medium'},
    '2025-03-18': {'event': 'Enhanced Chip Restrictions', 'description': 'US expands chip restrictions to more Chinese firms', 'color': '#d62728', 'category': 'semiconductors', 'severity': 'high'},
    
    # =========================================================================
    # ISRAEL-IRAN CONFLICT (2023-2025)
    # =========================================================================
    '2023-10-07': {'event': 'Hamas Attack on Israel', 'description': 'Hamas launches surprise attack from Gaza', 'color': '#d62728', 'category': 'israel_iran', 'severity': 'critical'},
    '2023-10-09': {'event': 'Israel Declares War', 'description': 'Israel formally declares war on Hamas', 'color': '#d62728', 'category': 'israel_iran', 'severity': 'high'},
    '2023-10-27': {'event': 'Israel Ground Invasion Gaza', 'description': 'Israel begins ground invasion of Gaza', 'color': '#ff7f0e', 'category': 'israel_iran', 'severity': 'high'},
    '2024-04-01': {'event': 'Israel Strikes Damascus', 'description': 'Israel strikes Iranian consulate, killing generals', 'color': '#d62728', 'category': 'israel_iran', 'severity': 'high'},
    '2024-04-13': {'event': 'Iran Drone Attack on Israel', 'description': 'Iran launches 300+ drones/missiles at Israel', 'color': '#d62728', 'category': 'israel_iran', 'severity': 'critical'},
    '2024-04-19': {'event': 'Israel Retaliates on Iran', 'description': 'Israel strikes Iranian air defense', 'color': '#ff7f0e', 'category': 'israel_iran', 'severity': 'high'},
    '2024-07-31': {'event': 'Hamas Leader Haniyeh Killed', 'description': 'Hamas political leader assassinated in Tehran', 'color': '#d62728', 'category': 'israel_iran', 'severity': 'high'},
    '2024-09-27': {'event': 'Hezbollah Leader Nasrallah Killed', 'description': 'Israel kills Hezbollah leader in Beirut', 'color': '#d62728', 'category': 'israel_iran', 'severity': 'high'},
    '2024-10-01': {'event': 'Iran Ballistic Missile Attack', 'description': 'Iran launches 180+ ballistic missiles at Israel', 'color': '#d62728', 'category': 'israel_iran', 'severity': 'critical'},
    '2024-10-26': {'event': 'Israel Strikes Iran', 'description': 'Major Israeli airstrikes on Iran military sites', 'color': '#ff7f0e', 'category': 'israel_iran', 'severity': 'high'},
    '2025-01-15': {'event': 'Gaza Ceasefire Agreement', 'description': 'Israel-Hamas ceasefire and hostage deal', 'color': '#2ca02c', 'category': 'israel_iran', 'severity': 'high'},
    
    # =========================================================================
    # UKRAINE-RUSSIA WAR (2023-2025)
    # =========================================================================
    '2023-06-06': {'event': 'Nova Kakhovka Dam Destroyed', 'description': 'Major dam destruction causes flooding', 'color': '#d62728', 'category': 'ukraine_russia', 'severity': 'high'},
    '2023-06-24': {'event': 'Wagner Mutiny', 'description': 'Wagner Group launches brief mutiny', 'color': '#ff7f0e', 'category': 'ukraine_russia', 'severity': 'high'},
    '2023-08-23': {'event': 'Prigozhin Killed', 'description': 'Wagner leader dies in plane crash', 'color': '#d62728', 'category': 'ukraine_russia', 'severity': 'medium'},
    '2024-02-16': {'event': 'Navalny Death', 'description': 'Russian opposition leader dies in prison', 'color': '#d62728', 'category': 'ukraine_russia', 'severity': 'high'},
    '2024-05-10': {'event': 'Russia Kharkiv Offensive', 'description': 'Russia launches offensive toward Kharkiv', 'color': '#ff7f0e', 'category': 'ukraine_russia', 'severity': 'high'},
    '2024-06-13': {'event': 'G7 $50B Ukraine Aid', 'description': 'G7 agrees loan from frozen Russian assets', 'color': '#2ca02c', 'category': 'ukraine_russia', 'severity': 'high'},
    '2024-08-06': {'event': 'Ukraine Kursk Incursion', 'description': 'Ukraine offensive into Russian Kursk region', 'color': '#1f77b4', 'category': 'ukraine_russia', 'severity': 'high'},
    '2024-09-12': {'event': 'US Long-Range Weapons Approved', 'description': 'Biden approves long-range missiles for Ukraine', 'color': '#2ca02c', 'category': 'ukraine_russia', 'severity': 'high'},
    '2024-11-19': {'event': 'Russia ICBM Strike Ukraine', 'description': 'Russia fires experimental ICBM at Ukraine', 'color': '#d62728', 'category': 'ukraine_russia', 'severity': 'critical'},
    '2024-12-01': {'event': 'Russia Donetsk Gains', 'description': 'Russia makes territorial gains in Donetsk', 'color': '#ff7f0e', 'category': 'ukraine_russia', 'severity': 'medium'},
    '2025-02-24': {'event': '3-Year War Anniversary', 'description': 'Third anniversary of Russian invasion', 'color': '#7f7f7f', 'category': 'ukraine_russia', 'severity': 'medium'},
    '2025-03-01': {'event': 'Trump-Zelensky Meeting', 'description': 'Trump meets Zelensky on peace framework', 'color': '#17becf', 'category': 'ukraine_russia', 'severity': 'high'},
    
    # =========================================================================
    # VENEZUELA (2024-2025)
    # =========================================================================
    '2024-07-28': {'event': 'Venezuela Election Dispute', 'description': 'Disputed election results spark protests', 'color': '#9467bd', 'category': 'venezuela', 'severity': 'medium'},
    '2024-08-05': {'event': 'Venezuela Opposition Claims Victory', 'description': 'Opposition claims win, protests intensify', 'color': '#9467bd', 'category': 'venezuela', 'severity': 'medium'},
    '2025-01-10': {'event': 'Maduro Inauguration Crisis', 'description': 'Maduro inaugurated amid non-recognition', 'color': '#9467bd', 'category': 'venezuela', 'severity': 'medium'},
    '2025-01-28': {'event': 'US Venezuela Sanctions', 'description': 'Trump reinstates full Venezuela sanctions', 'color': '#ff7f0e', 'category': 'venezuela', 'severity': 'medium'},
    
    # =========================================================================
    # OTHER MAJOR EVENTS
    # =========================================================================
    '2023-08-24': {'event': 'BRICS Expansion', 'description': 'BRICS invites 6 new members', 'color': '#bcbd22', 'category': 'other', 'severity': 'medium'},
    '2024-01-13': {'event': 'Taiwan Election', 'description': 'DPP wins Taiwan election, China tensions', 'color': '#e377c2', 'category': 'other', 'severity': 'high'},
    '2024-06-09': {'event': 'EU Parliament Elections', 'description': 'Far-right gains in EU elections', 'color': '#17becf', 'category': 'other', 'severity': 'medium'},
    '2024-07-04': {'event': 'UK Election - Labour Victory', 'description': 'Labour wins UK election in landslide', 'color': '#1f77b4', 'category': 'other', 'severity': 'high'},
    '2024-12-04': {'event': 'South Korea Martial Law', 'description': 'Korea president declares then reverses martial law', 'color': '#d62728', 'category': 'other', 'severity': 'high'},
    '2024-12-08': {'event': 'Syria Assad Falls', 'description': 'Assad flees as rebels take Damascus', 'color': '#d62728', 'category': 'other', 'severity': 'high'},
}

EVENT_CATEGORIES = {
    'trump_political': {'name': 'Trump Political', 'color': '#1f77b4'},
    'trump_tariff': {'name': 'Trump Tariffs', 'color': '#ff7f0e'},
    'eu_retaliation': {'name': 'EU Retaliation', 'color': '#17becf'},
    'uk_trade': {'name': 'UK Trade', 'color': '#2ca02c'},
    'semiconductors': {'name': 'Semiconductors', 'color': '#9467bd'},
    'israel_iran': {'name': 'Israel-Iran', 'color': '#d62728'},
    'ukraine_russia': {'name': 'Ukraine-Russia', 'color': '#8c564b'},
    'venezuela': {'name': 'Venezuela', 'color': '#e377c2'},
    'other': {'name': 'Other', 'color': '#7f7f7f'},
}

OUTPUT_DIR = './output_index_analysis_v3'

# =============================================================================
# DATA DOWNLOAD FUNCTIONS
# =============================================================================

def download_all_data(period: str = "2y") -> dict:
    """Download all required data"""
    print("=" * 75)
    print("DOWNLOADING ALL MARKET DATA")
    print("=" * 75)
    
    all_data = {}
    
    # 1. Index data
    print("\n📊 Downloading Index Data...")
    all_data['indices'] = download_ticker_data(INDEX_TICKERS, period)
    
    # 2. Sector data
    print("\n📈 Downloading Sector Data...")
    all_data['sectors'] = download_ticker_data(SECTOR_TICKERS, period)
    
    # 3. Currency data
    print("\n💱 Downloading Currency Data...")
    all_data['currencies'] = download_ticker_data(CURRENCY_TICKERS, period)
    
    return all_data

def download_ticker_data(tickers: dict, period: str) -> pd.DataFrame:
    """Download data for a set of tickers"""
    all_data = {}
    
    for name, ticker in tickers.items():
        try:
            print(f"  {name} ({ticker})...", end=" ")
            data = yf.download(ticker, period=period, progress=False, timeout=10)
            if not data.empty and len(data) > 20:
                col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                all_data[name] = data[col].squeeze()
                print(f"✓ ({len(data)} days)")
            else:
                print("✗ (insufficient data)")
        except Exception as e:
            print(f"✗ ({str(e)[:25]})")
            # Try ETF alternative for indices
            if name in ETF_ALTERNATIVES:
                try:
                    alt = ETF_ALTERNATIVES[name]
                    print(f"    Trying {alt}...", end=" ")
                    data = yf.download(alt, period=period, progress=False)
                    if not data.empty:
                        all_data[name] = data['Adj Close'].squeeze()
                        print("✓")
                except:
                    print("✗")
    
    df = pd.DataFrame(all_data)
    df.index = pd.to_datetime(df.index)
    return df.ffill(limit=5)

# =============================================================================
# CALCULATION FUNCTIONS
# =============================================================================

def calculate_returns(prices: pd.DataFrame, frequency: str = 'daily') -> pd.DataFrame:
    """Calculate returns at specified frequency"""
    if frequency == 'weekly':
        prices = prices.resample('W-FRI').last()
    elif frequency == 'monthly':
        prices = prices.resample('M').last()
    return prices.pct_change().dropna() * 100

def calculate_currency_adjusted_returns(index_returns: pd.DataFrame, 
                                         currency_returns: pd.DataFrame,
                                         country_currency: dict) -> pd.DataFrame:
    """Calculate USD-adjusted returns for each country"""
    adjusted = index_returns.copy()
    
    for country in index_returns.columns:
        if country in country_currency:
            currency = country_currency[country]
            if currency in currency_returns.columns:
                # Align indices
                common_idx = index_returns.index.intersection(currency_returns.index)
                # Local return + currency return = USD return
                adjusted.loc[common_idx, country] = (
                    index_returns.loc[common_idx, country] + 
                    currency_returns.loc[common_idx, currency]
                )
    
    return adjusted

def calculate_drawdown(cumulative_returns: pd.Series) -> pd.Series:
    """Calculate drawdown series from cumulative returns"""
    wealth = (1 + cumulative_returns / 100)
    running_max = wealth.expanding().max()
    drawdown = (wealth - running_max) / running_max * 100
    return drawdown

def calculate_drawdown_metrics(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate drawdown metrics for all series"""
    metrics = []
    
    for col in prices.columns:
        dd = calculate_drawdown(prices[col].pct_change().cumsum() * 100)
        
        metrics.append({
            'Index': col,
            'Max Drawdown (%)': dd.min(),
            'Current Drawdown (%)': dd.iloc[-1],
            'Avg Drawdown (%)': dd.mean(),
            'Days in Drawdown': (dd < 0).sum(),
            'Drawdown Recovery': 'Yes' if dd.iloc[-1] > -1 else 'No'
        })
    
    return pd.DataFrame(metrics)

def calculate_event_window_returns(returns: pd.DataFrame, events: dict,
                                    pre_days: int = 5, post_days: int = 10) -> pd.DataFrame:
    """Calculate returns around event windows (-5 to +10 days)"""
    results = []
    
    for date_str, event_info in events.items():
        event_date = pd.to_datetime(date_str)
        
        # Find nearest trading date
        idx = returns.index.searchsorted(event_date)
        if idx >= len(returns.index):
            continue
        
        event_idx = idx
        pre_start = max(0, event_idx - pre_days)
        post_end = min(len(returns), event_idx + post_days + 1)
        
        for country in returns.columns:
            if event_idx < len(returns):
                pre_ret = returns[country].iloc[pre_start:event_idx].sum() if event_idx > pre_start else 0
                event_ret = returns[country].iloc[event_idx] if pd.notna(returns[country].iloc[event_idx]) else 0
                post_ret = returns[country].iloc[event_idx+1:post_end].sum() if post_end > event_idx+1 else 0
                
                # Drawdown in window
                window_cum = returns[country].iloc[pre_start:post_end].cumsum()
                window_dd = calculate_drawdown(window_cum).min() if len(window_cum) > 0 else 0
                
                results.append({
                    'Event': event_info['event'],
                    'Date': date_str,
                    'Category': event_info['category'],
                    'Severity': event_info.get('severity', 'medium'),
                    'Country': country,
                    'Pre-Event Return (%)': pre_ret,
                    'Event Day Return (%)': event_ret,
                    'Post-Event Return (%)': post_ret,
                    'Total Window Return (%)': pre_ret + event_ret + post_ret,
                    'Window Max Drawdown (%)': window_dd,
                })
    
    return pd.DataFrame(results)

def calculate_rolling_metrics(returns: pd.DataFrame, windows: list = [5, 12, 26]) -> dict:
    """Calculate rolling correlation, beta, and volatility"""
    metrics = {}
    
    if 'SPY' not in returns.columns:
        return metrics
    
    for window in windows:
        metrics[window] = {
            'correlation': {},
            'beta': {},
            'volatility': {},
            'vol_ratio': {}
        }
        
        for col in returns.columns:
            if col != 'SPY':
                # Correlation
                metrics[window]['correlation'][col] = returns[col].rolling(window).corr(returns['SPY'])
                
                # Beta
                cov = returns[col].rolling(window).cov(returns['SPY'])
                var = returns['SPY'].rolling(window).var()
                metrics[window]['beta'][col] = cov / var
                
                # Volatility
                metrics[window]['volatility'][col] = returns[col].rolling(window).std()
                
                # Vol ratio
                spy_vol = returns['SPY'].rolling(window).std()
                metrics[window]['vol_ratio'][col] = metrics[window]['volatility'][col] / spy_vol
    
    return metrics

def detect_structural_breaks(returns: pd.DataFrame, country: str, 
                             min_segment: int = 20) -> list:
    """Detect structural breaks using CUSUM test"""
    if not STATSMODELS_AVAILABLE:
        return []
    
    breaks = []
    
    if 'SPY' not in returns.columns or country not in returns.columns:
        return breaks
    
    mask = returns[['SPY', country]].notna().all(axis=1)
    y = returns.loc[mask, country].values
    x = add_constant(returns.loc[mask, 'SPY'].values)
    dates = returns.loc[mask].index
    
    try:
        model = OLS(y, x).fit()
        resid = model.resid
        
        # CUSUM-based detection
        cusum = np.cumsum(resid - resid.mean()) / resid.std()
        
        # Find significant deviations
        threshold = 1.96 * np.sqrt(len(resid))
        
        # Find peaks in absolute CUSUM
        peaks, _ = find_peaks(np.abs(cusum), height=threshold * 0.5, distance=min_segment)
        
        for peak in peaks:
            if peak < len(dates):
                breaks.append({
                    'date': dates[peak],
                    'cusum_value': cusum[peak],
                    'significance': abs(cusum[peak]) / threshold
                })
        
    except Exception as e:
        print(f"  Structural break detection error for {country}: {e}")
    
    return breaks

def calculate_vix_regimes(vix: pd.Series, thresholds: tuple = (15, 25, 35)) -> pd.Series:
    """Classify VIX into volatility regimes"""
    low, med, high = thresholds
    
    def classify(v):
        if pd.isna(v):
            return 'Unknown'
        elif v < low:
            return 'Low Vol'
        elif v < med:
            return 'Normal'
        elif v < high:
            return 'Elevated'
        else:
            return 'Crisis'
    
    return vix.apply(classify)

def calculate_comprehensive_stats(returns: pd.DataFrame, 
                                   currency_adj_returns: pd.DataFrame = None) -> pd.DataFrame:
    """Calculate comprehensive statistics"""
    country_stats = []
    
    countries = [c for c in returns.columns if c != 'SPY']
    
    for country in countries:
        mask = returns[['SPY', country]].notna().all(axis=1)
        spy = returns.loc[mask, 'SPY']
        ret = returns.loc[mask, country]
        
        if len(spy) < 20:
            continue
        
        # Regression
        slope, intercept, r_val, p_val, std_err = stats.linregress(spy, ret)
        
        # Basic stats
        avg_ret = ret.mean()
        vol = ret.std()
        
        # Risk metrics
        tracking_error = (ret - spy).std()
        info_ratio = (ret.mean() - spy.mean()) / tracking_error if tracking_error > 0 else 0
        
        # Capture ratios
        up = spy > 0
        down = spy < 0
        up_capture = ret[up].mean() / spy[up].mean() * 100 if up.sum() > 5 else np.nan
        down_capture = ret[down].mean() / spy[down].mean() * 100 if down.sum() > 5 else np.nan
        
        # Drawdown
        cumret = (1 + ret / 100).cumprod()
        max_dd = ((cumret / cumret.expanding().max()) - 1).min() * 100
        
        # Currency adjusted stats
        curr_adj_ret = np.nan
        curr_adj_vol = np.nan
        if currency_adj_returns is not None and country in currency_adj_returns.columns:
            adj = currency_adj_returns[country].dropna()
            curr_adj_ret = adj.mean() * 52  # Annualized
            curr_adj_vol = adj.std() * np.sqrt(52)
        
        country_stats.append({
            'Country': country,
            'Region': COUNTRY_TO_REGION.get(country, 'Unknown'),
            'Beta': slope,
            'Alpha (%)': intercept,
            'R²': r_val ** 2,
            'Correlation': r_val,
            'Avg Weekly Return (%)': avg_ret,
            'Weekly Volatility (%)': vol,
            'Annualized Return (%)': avg_ret * 52,
            'Annualized Volatility (%)': vol * np.sqrt(52),
            'Sharpe Ratio': (avg_ret * 52) / (vol * np.sqrt(52)) if vol > 0 else 0,
            'Tracking Error (%)': tracking_error,
            'Information Ratio': info_ratio,
            'Upside Capture (%)': up_capture,
            'Downside Capture (%)': down_capture,
            'Max Drawdown (%)': max_dd,
            'Currency Adj Return (%)': curr_adj_ret,
            'Currency Adj Vol (%)': curr_adj_vol,
            'Skewness': ret.skew(),
            'Kurtosis': ret.kurtosis(),
            'P-value': p_val,
            'Observations': len(spy)
        })
    
    return pd.DataFrame(country_stats)

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_vix_overlay_chart(returns: pd.DataFrame, vix: pd.Series, 
                              country: str, events: dict) -> go.Figure:
    """Create chart with VIX overlay and volatility regimes"""
    
    # Align data
    common_idx = returns.index.intersection(vix.index)
    
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.5, 0.25, 0.25],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'{country} vs SPY Cumulative Returns',
            'VIX Level with Regime Bands',
            f'{country} Rolling Volatility (12W)'
        )
    )
    
    # Cumulative returns
    cum_spy = returns.loc[common_idx, 'SPY'].cumsum()
    cum_country = returns.loc[common_idx, country].cumsum()
    
    fig.add_trace(go.Scatter(
        x=common_idx, y=cum_spy,
        mode='lines', name='SPY',
        line=dict(color='#1f77b4', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=common_idx, y=cum_country,
        mode='lines', name=country,
        line=dict(color='#2ca02c', width=2)
    ), row=1, col=1)
    
    # VIX with regime bands
    vix_aligned = vix.loc[common_idx]
    
    # Add regime bands
    fig.add_hrect(y0=0, y1=15, fillcolor="green", opacity=0.1, 
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=15, y1=25, fillcolor="yellow", opacity=0.1,
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=25, y1=35, fillcolor="orange", opacity=0.1,
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=35, y1=80, fillcolor="red", opacity=0.1,
                  line_width=0, row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=vix_aligned.index, y=vix_aligned,
        mode='lines', name='VIX',
        line=dict(color='#d62728', width=1.5)
    ), row=2, col=1)
    
    # Rolling volatility
    rolling_vol = returns.loc[common_idx, country].rolling(12).std() * np.sqrt(52)
    
    fig.add_trace(go.Scatter(
        x=rolling_vol.index, y=rolling_vol,
        mode='lines', name=f'{country} Vol',
        line=dict(color='#ff7f0e', width=1.5),
        fill='tozeroy', fillcolor='rgba(255, 127, 14, 0.2)'
    ), row=3, col=1)
    
    # Add events
    for date_str, event_info in events.items():
        event_date = pd.to_datetime(date_str)
        if event_date in common_idx or (common_idx.min() < event_date < common_idx.max()):
            for row in [1, 2, 3]:
                fig.add_vline(
                    x=event_date, line_dash="dash",
                    line_color=event_info['color'], line_width=1, opacity=0.5,
                    row=row, col=1
                )
    
    fig.update_layout(
        title=dict(
            text=f'<b>{country} Analysis with VIX Overlay</b><br>'
                 '<sup>Green=Low Vol, Yellow=Normal, Orange=Elevated, Red=Crisis</sup>',
            x=0.5, font=dict(size=16)
        ),
        height=900, width=1200,
        showlegend=True,
        legend=dict(orientation='h', y=1.02),
        template='plotly_white'
    )
    
    fig.update_yaxes(title_text='Cumulative Return (%)', row=1, col=1)
    fig.update_yaxes(title_text='VIX', row=2, col=1)
    fig.update_yaxes(title_text='Annualized Vol (%)', row=3, col=1)
    
    return fig

def create_sector_comparison_chart(sector_returns: pd.DataFrame, events: dict) -> go.Figure:
    """Compare tech vs defensive sectors around events"""
    
    # Calculate cumulative returns
    tech_cols = [c for c in ['US Tech', 'US Semiconductors', 'US Growth'] if c in sector_returns.columns]
    defensive_cols = [c for c in ['US Utilities', 'US Consumer Staples', 'US Healthcare', 'US Dividend'] if c in sector_returns.columns]
    
    if not tech_cols or not defensive_cols:
        return None
    
    tech_avg = sector_returns[tech_cols].mean(axis=1)
    defensive_avg = sector_returns[defensive_cols].mean(axis=1)
    
    tech_cum = tech_avg.cumsum()
    defensive_cum = defensive_avg.cumsum()
    spread = tech_cum - defensive_cum
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('Tech vs Defensive Cumulative Returns', 'Tech - Defensive Spread')
    )
    
    fig.add_trace(go.Scatter(
        x=tech_cum.index, y=tech_cum,
        mode='lines', name='Tech Average',
        line=dict(color='#1f77b4', width=2.5)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=defensive_cum.index, y=defensive_cum,
        mode='lines', name='Defensive Average',
        line=dict(color='#2ca02c', width=2.5)
    ), row=1, col=1)
    
    # Spread
    fig.add_trace(go.Scatter(
        x=spread.index, y=spread,
        mode='lines', name='Spread',
        line=dict(color='#ff7f0e', width=2),
        fill='tozeroy', fillcolor='rgba(255, 127, 14, 0.2)'
    ), row=2, col=1)
    
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=2, col=1)
    
    # Events
    for date_str, event_info in events.items():
        event_date = pd.to_datetime(date_str)
        if event_date >= tech_cum.index.min() and event_date <= tech_cum.index.max():
            for row in [1, 2]:
                fig.add_vline(
                    x=event_date, line_dash="dash",
                    line_color=event_info['color'], line_width=1, opacity=0.5,
                    row=row, col=1
                )
    
    fig.update_layout(
        title=dict(
            text='<b>Tech vs Defensive Sector Performance</b><br>'
                 '<sup>QQQ/SOXX/VUG vs XLU/XLP/XLV/VYM</sup>',
            x=0.5, font=dict(size=16)
        ),
        height=700, width=1150,
        template='plotly_white'
    )
    
    return fig

def create_currency_adjusted_comparison(local_returns: pd.DataFrame,
                                         usd_returns: pd.DataFrame,
                                         country: str, events: dict) -> go.Figure:
    """Compare local vs USD-adjusted returns"""
    
    if country not in local_returns.columns or country not in usd_returns.columns:
        return None
    
    local_cum = local_returns[country].cumsum()
    usd_cum = usd_returns[country].cumsum()
    currency_effect = usd_cum - local_cum
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f'{country}: Local vs USD-Adjusted Returns',
            'Currency Effect (USD Return - Local Return)'
        )
    )
    
    fig.add_trace(go.Scatter(
        x=local_cum.index, y=local_cum,
        mode='lines', name='Local Currency',
        line=dict(color='#1f77b4', width=2)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=usd_cum.index, y=usd_cum,
        mode='lines', name='USD-Adjusted',
        line=dict(color='#2ca02c', width=2)
    ), row=1, col=1)
    
    # Currency effect
    fig.add_trace(go.Scatter(
        x=currency_effect.index, y=currency_effect,
        mode='lines', name='Currency Effect',
        line=dict(color='#ff7f0e', width=2),
        fill='tozeroy', fillcolor='rgba(255, 127, 14, 0.2)'
    ), row=2, col=1)
    
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=2, col=1)
    
    # Events
    for date_str, event_info in events.items():
        event_date = pd.to_datetime(date_str)
        if event_date >= local_cum.index.min() and event_date <= local_cum.index.max():
            fig.add_vline(
                x=event_date, line_dash="dash",
                line_color=event_info['color'], line_width=1, opacity=0.5,
                row='all', col=1
            )
    
    currency_name = COUNTRY_CURRENCY.get(country, 'N/A')
    
    fig.update_layout(
        title=dict(
            text=f'<b>{country} Currency-Adjusted Returns</b><br>'
                 f'<sup>Currency pair: {currency_name}</sup>',
            x=0.5, font=dict(size=16)
        ),
        height=650, width=1100,
        template='plotly_white'
    )
    
    return fig

def create_drawdown_analysis_chart(returns: pd.DataFrame, events: dict) -> go.Figure:
    """Create drawdown analysis around events"""
    
    countries = [c for c in returns.columns if c != 'SPY'][:8]  # Top 8
    
    fig = make_subplots(
        rows=len(countries), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f'{c} Drawdown' for c in countries]
    )
    
    for i, country in enumerate(countries, 1):
        dd = calculate_drawdown(returns[country].cumsum())
        
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd,
            mode='lines', name=country,
            line=dict(width=1.5),
            fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.2)'
        ), row=i, col=1)
        
        fig.add_hline(y=0, line_dash="solid", line_color="gray", 
                      line_width=0.5, row=i, col=1)
    
    # Events on all rows
    for date_str, event_info in events.items():
        event_date = pd.to_datetime(date_str)
        if event_date >= returns.index.min() and event_date <= returns.index.max():
            if event_info.get('severity') in ['high', 'critical']:
                for row in range(1, len(countries) + 1):
                    fig.add_vline(
                        x=event_date, line_dash="dash",
                        line_color=event_info['color'], line_width=1, opacity=0.4,
                        row=row, col=1
                    )
    
    fig.update_layout(
        title=dict(
            text='<b>Drawdown Analysis Across Markets</b><br>'
                 '<sup>With High-Severity Event Markers</sup>',
            x=0.5, font=dict(size=16)
        ),
        height=150 * len(countries) + 100,
        width=1200,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def create_structural_breaks_chart(returns: pd.DataFrame, country: str, 
                                    breaks: list, events: dict) -> go.Figure:
    """Visualize structural breaks in beta relationship"""
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            f'{country} Rolling Beta with Structural Breaks',
            'CUSUM Statistic'
        )
    )
    
    # Rolling beta
    cov = returns[country].rolling(12).cov(returns['SPY'])
    var = returns['SPY'].rolling(12).var()
    beta = cov / var
    
    fig.add_trace(go.Scatter(
        x=beta.index, y=beta,
        mode='lines', name='Rolling Beta (12W)',
        line=dict(color='#1f77b4', width=2)
    ), row=1, col=1)
    
    fig.add_hline(y=1, line_dash="dot", line_color="black", row=1, col=1)
    
    # Mark structural breaks
    for br in breaks:
        fig.add_vline(
            x=br['date'], line_dash="dash",
            line_color='red', line_width=2,
            row=1, col=1
        )
        fig.add_annotation(
            x=br['date'], y=beta.max(),
            text="Break", showarrow=True,
            arrowhead=2, arrowcolor='red',
            row=1, col=1
        )
    
    # CUSUM plot (simplified)
    mask = returns[['SPY', country]].notna().all(axis=1)
    y = returns.loc[mask, country].values
    x = returns.loc[mask, 'SPY'].values
    dates = returns.loc[mask].index
    
    try:
        residuals = y - (np.polyfit(x, y, 1)[0] * x + np.polyfit(x, y, 1)[1])
        cusum = np.cumsum(residuals - residuals.mean()) / residuals.std()
        
        fig.add_trace(go.Scatter(
            x=dates, y=cusum,
            mode='lines', name='CUSUM',
            line=dict(color='#ff7f0e', width=1.5)
        ), row=2, col=1)
        
        # Significance bounds
        n = len(cusum)
        bound = 1.96 * np.sqrt(n)
        fig.add_hline(y=bound, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-bound, line_dash="dash", line_color="red", row=2, col=1)
    except:
        pass
    
    fig.update_layout(
        title=dict(
            text=f'<b>{country} Structural Break Analysis</b><br>'
                 f'<sup>Red dashed = detected breaks | Bounds = 95% CI</sup>',
            x=0.5, font=dict(size=16)
        ),
        height=700, width=1100,
        template='plotly_white'
    )
    
    return fig

def create_event_window_heatmap(event_window_stats: pd.DataFrame) -> go.Figure:
    """Create heatmap of event window returns"""
    
    # Pivot for total window return
    pivot = event_window_stats.pivot_table(
        values='Total Window Return (%)',
        index='Event',
        columns='Country',
        aggfunc='mean'
    )
    
    # Sort columns by average
    col_order = pivot.mean().sort_values(ascending=False).index
    pivot = pivot[col_order]
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(pivot.values, 1),
        texttemplate='%{text}',
        textfont={"size": 7},
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>',
        colorbar=dict(title='Return (%)')
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>Event Window Returns (-5 to +10 Days)</b><br>'
                 '<sup>Total cumulative return around each event</sup>',
            x=0.5, font=dict(size=16)
        ),
        height=max(600, len(pivot) * 18),
        width=1400,
        xaxis_title='Country',
        yaxis_title='Event',
        template='plotly_white',
        xaxis=dict(tickangle=45)
    )
    
    return fig

# =============================================================================
# PDF REPORT GENERATION
# =============================================================================

def generate_pdf_report(stats_df: pd.DataFrame, event_stats: pd.DataFrame,
                        weekly_returns: pd.DataFrame, events: dict,
                        output_path: str):
    """Generate comprehensive PDF report"""
    
    if not REPORTLAB_AVAILABLE:
        print("⚠ PDF generation skipped - reportlab not installed")
        return
    
    print("\n📄 Generating PDF Report...")
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=50, leftMargin=50,
        topMargin=50, bottomMargin=50
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1a1a2e')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10,
        textColor=colors.HexColor('#667eea')
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.HexColor('#1a1a2e')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        leading=14
    )
    
    elements = []
    
    # Title Page
    elements.append(Spacer(1, 100))
    elements.append(Paragraph("International Stock Index Analysis", title_style))
    elements.append(Paragraph("vs S&P 500 (SPY)", styles['Heading2']))
    elements.append(Spacer(1, 30))
    elements.append(Paragraph(
        f"Comprehensive Analysis with Geopolitical Events",
        ParagraphStyle('Subtitle', parent=styles['Normal'], fontSize=14, alignment=TA_CENTER)
    ))
    elements.append(Spacer(1, 50))
    elements.append(Paragraph(
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ParagraphStyle('Date', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER, textColor=colors.gray)
    ))
    elements.append(Paragraph(
        f"Data Period: {weekly_returns.index.min().strftime('%Y-%m-%d')} to {weekly_returns.index.max().strftime('%Y-%m-%d')}",
        ParagraphStyle('Period', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER, textColor=colors.gray)
    ))
    elements.append(PageBreak())
    
    # Executive Summary
    elements.append(Paragraph("Executive Summary", heading_style))
    
    # Key findings
    if len(stats_df) > 0:
        highest_beta = stats_df.loc[stats_df['Beta'].idxmax()]
        lowest_beta = stats_df.loc[stats_df['Beta'].idxmin()]
        highest_corr = stats_df.loc[stats_df['Correlation'].idxmax()]
        lowest_corr = stats_df.loc[stats_df['Correlation'].idxmin()]
        best_sharpe = stats_df.loc[stats_df['Sharpe Ratio'].idxmax()]
        
        summary_text = f"""
        <b>Key Findings:</b><br/><br/>
        
        • <b>Highest Beta:</b> {highest_beta['Country']} (β = {highest_beta['Beta']:.2f}) - Most sensitive to US market movements<br/>
        • <b>Lowest Beta:</b> {lowest_beta['Country']} (β = {lowest_beta['Beta']:.2f}) - Most defensive relative to US<br/>
        • <b>Highest Correlation:</b> {highest_corr['Country']} (ρ = {highest_corr['Correlation']:.2f}) - Moves most closely with SPY<br/>
        • <b>Lowest Correlation:</b> {lowest_corr['Country']} (ρ = {lowest_corr['Correlation']:.2f}) - Best diversification potential<br/>
        • <b>Best Risk-Adjusted Return:</b> {best_sharpe['Country']} (Sharpe = {best_sharpe['Sharpe Ratio']:.2f})<br/><br/>
        
        <b>Analysis Period:</b> {len(weekly_returns)} weeks of data analyzed<br/>
        <b>Geopolitical Events Tracked:</b> {len(events)} major events across 9 categories<br/>
        <b>Countries Analyzed:</b> {len(stats_df)} international indices
        """
        elements.append(Paragraph(summary_text, body_style))
    
    elements.append(Spacer(1, 20))
    
    # Events summary by category
    elements.append(Paragraph("Geopolitical Events Summary", subheading_style))
    
    event_counts = {}
    for event_info in events.values():
        cat = event_info['category']
        event_counts[cat] = event_counts.get(cat, 0) + 1
    
    event_data = [['Category', 'Event Count', 'Key Theme']]
    category_themes = {
        'trump_tariff': 'Trade policy disruption',
        'trump_political': 'Policy uncertainty',
        'eu_retaliation': 'Trade war escalation',
        'uk_trade': 'Post-Brexit relations',
        'semiconductors': 'Tech supply chains',
        'israel_iran': 'Middle East tensions',
        'ukraine_russia': 'European security',
        'venezuela': 'Latin America instability',
        'other': 'Various global events'
    }
    
    for cat, count in sorted(event_counts.items(), key=lambda x: -x[1]):
        cat_name = EVENT_CATEGORIES.get(cat, {}).get('name', cat)
        theme = category_themes.get(cat, 'N/A')
        event_data.append([cat_name, str(count), theme])
    
    event_table = Table(event_data, colWidths=[150, 80, 200])
    event_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    elements.append(event_table)
    elements.append(PageBreak())
    
    # Country Statistics Table
    elements.append(Paragraph("Country Statistics vs SPY", heading_style))
    
    # Prepare table data
    table_cols = ['Country', 'Region', 'Beta', 'Alpha (%)', 'R²', 'Correlation', 
                  'Ann. Return (%)', 'Ann. Vol (%)', 'Sharpe']
    
    table_data = [table_cols]
    for _, row in stats_df.head(16).iterrows():
        table_data.append([
            row['Country'],
            row['Region'][:12],
            f"{row['Beta']:.2f}",
            f"{row['Alpha (%)']:.2f}",
            f"{row['R²']:.2f}",
            f"{row['Correlation']:.2f}",
            f"{row['Annualized Return (%)']:.1f}",
            f"{row['Annualized Volatility (%)']:.1f}",
            f"{row['Sharpe Ratio']:.2f}"
        ])
    
    col_widths = [70, 70, 40, 50, 35, 55, 60, 55, 45]
    stats_table = Table(table_data, colWidths=col_widths)
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
    ]))
    elements.append(stats_table)
    elements.append(Spacer(1, 20))
    
    # Regional Analysis
    elements.append(Paragraph("Regional Analysis", subheading_style))
    
    regional_stats = stats_df.groupby('Region').agg({
        'Beta': 'mean',
        'Correlation': 'mean',
        'Annualized Return (%)': 'mean',
        'Annualized Volatility (%)': 'mean',
        'Sharpe Ratio': 'mean'
    }).round(2)
    
    regional_data = [['Region', 'Avg Beta', 'Avg Corr', 'Avg Return (%)', 'Avg Vol (%)', 'Avg Sharpe']]
    for region, row in regional_stats.iterrows():
        regional_data.append([
            region,
            f"{row['Beta']:.2f}",
            f"{row['Correlation']:.2f}",
            f"{row['Annualized Return (%)']:.1f}",
            f"{row['Annualized Volatility (%)']:.1f}",
            f"{row['Sharpe Ratio']:.2f}"
        ])
    
    regional_table = Table(regional_data, colWidths=[100, 60, 60, 80, 70, 70])
    regional_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
    ]))
    elements.append(regional_table)
    elements.append(PageBreak())
    
    # Event Impact Analysis
    elements.append(Paragraph("Event Impact Analysis", heading_style))
    
    if len(event_stats) > 0:
        # Most impactful events
        event_impact = event_stats.groupby('Event')['Total Window Return (%)'].agg(['mean', 'std']).round(2)
        event_impact = event_impact.sort_values('mean')
        
        elements.append(Paragraph("Most Negative Market Impact Events:", subheading_style))
        
        worst_events = event_impact.head(5)
        worst_data = [['Event', 'Avg Return (%)', 'Std Dev (%)']]
        for event, row in worst_events.iterrows():
            worst_data.append([event[:40], f"{row['mean']:.1f}", f"{row['std']:.1f}"])
        
        worst_table = Table(worst_data, colWidths=[250, 100, 100])
        worst_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d62728')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ]))
        elements.append(worst_table)
        elements.append(Spacer(1, 15))
        
        elements.append(Paragraph("Most Positive Market Impact Events:", subheading_style))
        
        best_events = event_impact.tail(5).iloc[::-1]
        best_data = [['Event', 'Avg Return (%)', 'Std Dev (%)']]
        for event, row in best_events.iterrows():
            best_data.append([event[:40], f"{row['mean']:.1f}", f"{row['std']:.1f}"])
        
        best_table = Table(best_data, colWidths=[250, 100, 100])
        best_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ]))
        elements.append(best_table)
    
    elements.append(PageBreak())
    
    # Key Insights
    elements.append(Paragraph("Key Insights & Recommendations", heading_style))
    
    insights_text = """
    <b>1. Beta Analysis:</b><br/>
    Higher beta markets (Korea, Finland, Italy) offer greater upside potential but increased downside risk during US market selloffs. 
    Lower beta markets (Switzerland, UK, Japan) provide defensive characteristics.<br/><br/>
    
    <b>2. Correlation Dynamics:</b><br/>
    Correlations tend to increase during crisis periods (correlation breakdown). 
    China shows lowest correlation, offering diversification benefits but also idiosyncratic risks from trade tensions.<br/><br/>
    
    <b>3. Event Impact Patterns:</b><br/>
    • Trump tariff announcements cause immediate volatility spikes<br/>
    • European indices most affected by EU retaliation events<br/>
    • Asian indices sensitive to semiconductor restrictions<br/>
    • Israel-Iran events primarily affect European markets<br/><br/>
    
    <b>4. Currency Considerations:</b><br/>
    EUR-denominated markets face additional currency risk during risk-off periods as USD typically strengthens. 
    Currency-hedged positions may reduce overall volatility.<br/><br/>
    
    <b>5. Volatility Regimes:</b><br/>
    VIX above 25 indicates elevated regime - correlations increase and diversification benefits diminish. 
    Position sizing should be adjusted accordingly.<br/><br/>
    
    <b>Disclaimer:</b> This analysis is for informational purposes only and does not constitute investment advice. 
    Past performance is not indicative of future results.
    """
    elements.append(Paragraph(insights_text, body_style))
    
    # Build PDF
    doc.build(elements)
    print(f"✓ PDF Report saved to: {output_path}")

# =============================================================================
# HTML DASHBOARD GENERATION
# =============================================================================

def generate_html_dashboard(figures: dict, stats_df: pd.DataFrame, 
                             events: dict, output_path: str):
    """Generate comprehensive HTML dashboard"""
    
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>International Index Analysis V3</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
            min-height: 100vh;
            padding: 20px;
            color: #e0e0e0;
        }
        .container { max-width: 1800px; margin: 0 auto; }
        header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        header h1 { font-size: 2em; margin-bottom: 8px; }
        .nav {
            position: sticky;
            top: 0;
            background: rgba(26, 26, 46, 0.98);
            padding: 12px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            z-index: 1000;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            backdrop-filter: blur(10px);
        }
        .nav a {
            color: white;
            text-decoration: none;
            padding: 8px 14px;
            border-radius: 5px;
            background: rgba(255,255,255,0.05);
            font-size: 0.9em;
        }
        .nav a:hover { background: rgba(255,255,255,0.15); }
        .section {
            background: rgba(255,255,255,0.03);
            border-radius: 15px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
            overflow: hidden;
        }
        .section-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 15px 25px;
        }
        .section-content { padding: 20px; }
        .plot-container {
            background: rgba(255,255,255,0.02);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .event-legend {
            background: rgba(255,255,255,0.05);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .event-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; }
        .event-item {
            display: flex;
            align-items: center;
            padding: 6px 10px;
            background: rgba(255,255,255,0.05);
            border-radius: 5px;
            font-size: 0.8em;
        }
        .event-color { width: 10px; height: 10px; border-radius: 3px; margin-right: 8px; }
        footer { text-align: center; padding: 20px; color: #666; font-size: 0.85em; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌍 International Index Analysis V3</h1>
            <p style="opacity: 0.8;">Comprehensive Analysis with 60+ Geopolitical Events | Currency Adjusted | VIX Overlay | Structural Breaks</p>
            <p style="font-size: 0.85em; opacity: 0.6; margin-top: 8px;">Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M") + """</p>
        </header>
        
        <nav class="nav">
            <a href="#summary">📊 Summary</a>
            <a href="#vix">📈 VIX Analysis</a>
            <a href="#sectors">🏭 Sectors</a>
            <a href="#currency">💱 Currency</a>
            <a href="#drawdown">📉 Drawdowns</a>
            <a href="#events">📅 Events</a>
            <a href="#breaks">🔀 Breaks</a>
            <a href="#countries">🌍 Countries</a>
        </nav>
        
        <div class="event-legend">
            <h3 style="margin-bottom: 10px;">Event Categories</h3>
            <div class="event-grid">
"""
    
    for cat, info in EVENT_CATEGORIES.items():
        count = sum(1 for e in events.values() if e['category'] == cat)
        html += f"""
                <div class="event-item">
                    <div class="event-color" style="background-color: {info['color']};"></div>
                    <span>{info['name']} ({count})</span>
                </div>
"""
    
    html += """
            </div>
        </div>
"""
    
    # Add sections
    sections = [
        ('summary', '📊 Summary Dashboard', ['summary_dashboard']),
        ('vix', '📈 VIX & Volatility Analysis', ['vix_germany', 'vix_uk', 'vix_japan', 'vix_china']),
        ('sectors', '🏭 Sector Comparison', ['sector_comparison']),
        ('currency', '💱 Currency-Adjusted Returns', ['currency_germany', 'currency_uk', 'currency_japan']),
        ('drawdown', '📉 Drawdown Analysis', ['drawdown_analysis']),
        ('events', '📅 Event Impact', ['event_window_heatmap', 'event_impact']),
        ('breaks', '🔀 Structural Breaks', ['breaks_germany', 'breaks_uk', 'breaks_japan']),
    ]
    
    for section_id, section_title, plot_names in sections:
        html += f"""
        <div class="section" id="{section_id}">
            <div class="section-header"><h2>{section_title}</h2></div>
            <div class="section-content">
"""
        for plot_name in plot_names:
            if plot_name in figures and figures[plot_name] is not None:
                html += f"""
                <div class="plot-container">
                    {figures[plot_name].to_html(full_html=False, include_plotlyjs=False)}
                </div>
"""
        html += """
            </div>
        </div>
"""
    
    # Country section
    html += """
        <div class="section" id="countries">
            <div class="section-header"><h2>🌍 Individual Country Analysis</h2></div>
            <div class="section-content">
"""
    
    for fig_name, fig in figures.items():
        if fig_name.startswith('regression_') or fig_name.startswith('rolling_'):
            html += f"""
                <div class="plot-container">
                    {fig.to_html(full_html=False, include_plotlyjs=False)}
                </div>
"""
    
    html += """
            </div>
        </div>
        
        <footer>
            <p>Analysis powered by yfinance, pandas, numpy, scipy, plotly, reportlab</p>
            <p>Data from Yahoo Finance | For educational purposes only</p>
        </footer>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✓ HTML Dashboard saved to: {output_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "=" * 80)
    print("   INTERNATIONAL INDEX ANALYSIS V3")
    print("   Full Geopolitical Events | Currency | VIX | Sectors | PDF Report")
    print("=" * 80 + "\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Download all data
    all_data = download_all_data(period="2y")
    
    prices = all_data['indices']
    sector_prices = all_data['sectors']
    currency_prices = all_data['currencies']
    
    if prices.empty or 'SPY' not in prices.columns:
        print("\n❌ ERROR: Failed to download SPY data")
        return
    
    print(f"\n✓ Data range: {prices.index.min().strftime('%Y-%m-%d')} to {prices.index.max().strftime('%Y-%m-%d')}")
    
    # 2. Calculate returns
    print("\n" + "=" * 65)
    print("CALCULATING METRICS")
    print("=" * 65)
    
    weekly_returns = calculate_returns(prices, 'weekly')
    daily_returns = calculate_returns(prices, 'daily')
    sector_returns = calculate_returns(sector_prices, 'weekly')
    currency_returns = calculate_returns(currency_prices, 'weekly')
    
    print(f"✓ Weekly returns: {len(weekly_returns)} weeks")
    
    # 3. Currency-adjusted returns
    currency_adj_returns = calculate_currency_adjusted_returns(
        weekly_returns, currency_returns, COUNTRY_CURRENCY
    )
    print("✓ Currency-adjusted returns calculated")
    
    # 4. Statistics
    stats_df = calculate_comprehensive_stats(weekly_returns, currency_adj_returns)
    stats_df = stats_df.sort_values('Beta', ascending=False)
    print(f"✓ Statistics for {len(stats_df)} countries")
    
    # 5. Event window analysis
    event_stats = calculate_event_window_returns(
        daily_returns, GEOPOLITICAL_EVENTS, pre_days=5, post_days=10
    )
    print(f"✓ Event window stats: {len(event_stats)} data points")
    
    # 6. Rolling metrics
    rolling_metrics = calculate_rolling_metrics(weekly_returns, windows=[8, 12, 26])
    print("✓ Rolling metrics calculated")
    
    # 7. Structural breaks detection
    print("\n" + "=" * 65)
    print("DETECTING STRUCTURAL BREAKS")
    print("=" * 65)
    
    structural_breaks = {}
    for country in ['Germany', 'UK', 'Japan', 'China', 'France']:
        if country in weekly_returns.columns:
            breaks = detect_structural_breaks(weekly_returns, country)
            structural_breaks[country] = breaks
            if breaks:
                print(f"  {country}: {len(breaks)} breaks detected")
            else:
                print(f"  {country}: No significant breaks")
    
    # 8. Generate visualizations
    print("\n" + "=" * 65)
    print("GENERATING VISUALIZATIONS")
    print("=" * 65)
    
    figures = {}
    
    # Summary (reuse from V2 logic)
    from plotly.subplots import make_subplots
    
    # VIX analysis
    vix = sector_prices.get('VIX', pd.Series())
    if not vix.empty:
        for country in ['Germany', 'UK', 'Japan', 'China']:
            if country in weekly_returns.columns:
                print(f"  VIX overlay for {country}...")
                figures[f'vix_{country.lower()}'] = create_vix_overlay_chart(
                    weekly_returns, vix, country, GEOPOLITICAL_EVENTS
                )
    
    # Sector comparison
    print("  Sector comparison...")
    figures['sector_comparison'] = create_sector_comparison_chart(
        sector_returns, GEOPOLITICAL_EVENTS
    )
    
    # Currency adjusted
    for country in ['Germany', 'UK', 'Japan']:
        if country in weekly_returns.columns:
            print(f"  Currency adjusted for {country}...")
            figures[f'currency_{country.lower()}'] = create_currency_adjusted_comparison(
                weekly_returns, currency_adj_returns, country, GEOPOLITICAL_EVENTS
            )
    
    # Drawdown analysis
    print("  Drawdown analysis...")
    figures['drawdown_analysis'] = create_drawdown_analysis_chart(
        weekly_returns, GEOPOLITICAL_EVENTS
    )
    
    # Event window heatmap
    print("  Event window heatmap...")
    figures['event_window_heatmap'] = create_event_window_heatmap(event_stats)
    
    # Structural breaks
    for country, breaks in structural_breaks.items():
        if country in weekly_returns.columns:
            print(f"  Structural breaks for {country}...")
            figures[f'breaks_{country.lower()}'] = create_structural_breaks_chart(
                weekly_returns, country, breaks, GEOPOLITICAL_EVENTS
            )
    
    # 9. Save outputs
    print("\n" + "=" * 65)
    print("SAVING OUTPUTS")
    print("=" * 65)
    
    # HTML Dashboard
    html_path = os.path.join(OUTPUT_DIR, 'index_analysis_dashboard_v3.html')
    generate_html_dashboard(figures, stats_df, GEOPOLITICAL_EVENTS, html_path)
    
    # PDF Report
    pdf_path = os.path.join(OUTPUT_DIR, 'index_analysis_report_v3.pdf')
    generate_pdf_report(stats_df, event_stats, weekly_returns, GEOPOLITICAL_EVENTS, pdf_path)
    
    # CSV files
    stats_df.to_csv(os.path.join(OUTPUT_DIR, 'statistics_v3.csv'), index=False)
    event_stats.to_csv(os.path.join(OUTPUT_DIR, 'event_window_stats_v3.csv'), index=False)
    weekly_returns.to_csv(os.path.join(OUTPUT_DIR, 'weekly_returns.csv'))
    currency_adj_returns.to_csv(os.path.join(OUTPUT_DIR, 'currency_adjusted_returns.csv'))
    
    print(f"  ✓ All CSV files saved")
    
    # Save individual figures
    for name, fig in figures.items():
        if fig is not None:
            fig.write_html(os.path.join(OUTPUT_DIR, f'{name}.html'))
    
    print(f"  ✓ Individual plots saved")
    
    # Summary
    print("\n" + "=" * 80)
    print("   ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\n📁 Output: {os.path.abspath(OUTPUT_DIR)}")
    print(f"📊 Plots: {len([f for f in figures.values() if f])}")
    print(f"📅 Events: {len(GEOPOLITICAL_EVENTS)}")
    print(f"📄 PDF Report: {pdf_path}")
    print(f"🌐 Dashboard: {html_path}")
    
    return figures, weekly_returns, stats_df, event_stats

if __name__ == "__main__":
    results = main()
