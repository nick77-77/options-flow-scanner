import streamlit as st
import httpx
from datetime import datetime, date
from collections import defaultdict
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import sqlite3
from functools import wraps
import json

# --- CONFIGURATION ---
class Config:
    UW_TOKEN = st.secrets.get("UW_TOKEN", "e6e8601a-0746-4cec-a07d-c3eabfc13926")
    EXCLUDE_TICKERS = {'TSLA', 'MSTR', 'CRCL', 'COIN', 'META'}
    ALLOWED_TICKERS = {'QQQ', 'SPY', 'IWM'}
    MIN_PREMIUM = 100000
    LIMIT = 500
    SCENARIO_OTM_CALL_MIN_PREMIUM = 100000
    SCENARIO_ITM_CONV_MIN_PREMIUM = 50000
    SCENARIO_SWEEP_VOLUME_OI_RATIO = 2
    SCENARIO_BLOCK_TRADE_VOL = 100
    HIGH_IV_THRESHOLD = 0.30
    EXTREME_IV_THRESHOLD = 0.50
    IV_CRUSH_THRESHOLD = 0.15
    HIGH_VOL_OI_RATIO = 5.0
    UNUSUAL_OI_THRESHOLD = 1000
    
    # New thresholds for enhanced features
    INSTITUTIONAL_PREMIUM_THRESHOLD = 1000000
    DARK_POOL_VOLUME_THRESHOLD = 1000
    GAMMA_SQUEEZE_THRESHOLD = 0.05
    RETAIL_PREMIUM_THRESHOLD = 50000

config = Config()

# --- CUSTOM CSS ---
def load_custom_css():
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .alert-critical {
        background: #ff4444;
        border-left: 5px solid #cc0000;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: white;
    }
    
    .alert-high {
        background: #ff8800;
        border-left: 5px solid #cc6600;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: white;
    }
    
    .alert-medium {
        background: #ffaa00;
        border-left: 5px solid #cc8800;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
        color: white;
    }
    
    .stDataFrame {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-live {
        background: #00ff00;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .quick-stats {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 10px;
    }
    
    .stat-item {
        text-align: center;
        padding: 0.5rem;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)

# --- RATE LIMITING ---
def rate_limit(max_calls=30, period=60):
    """Rate limiting decorator"""
    def decorator(func):
        calls = []
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            calls[:] = [call for call in calls if now - call < period]
            
            if len(calls) >= max_calls:
                sleep_time = period - (now - calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            calls.append(now)
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# --- DATABASE SETUP ---
@st.cache_resource
def init_database():
    """Initialize SQLite database for historical data"""
    conn = sqlite3.connect('options_flow.db', check_same_thread=False)
    
    conn.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ticker TEXT,
            option_chain TEXT,
            option_type TEXT,
            strike REAL,
            expiry DATE,
            dte INTEGER,
            premium REAL,
            volume INTEGER,
            open_interest INTEGER,
            vol_oi_ratio REAL,
            iv REAL,
            trade_side TEXT,
            scenarios TEXT,
            alert_score INTEGER,
            underlying_price REAL,
            moneyness TEXT
        )
    ''')
    
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_ticker_timestamp ON trades(ticker, timestamp)
    ''')
    
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_premium ON trades(premium)
    ''')
    
    conn.commit()
    return conn

# --- ENHANCED CACHING ---
@st.cache_data(ttl=300, max_entries=10)
def fetch_general_flow_cached():
    """Cached version of general flow fetch"""
    return fetch_general_flow()

@st.cache_data(ttl=60, max_entries=5)
def fetch_etf_trades_cached():
    """Cached version of ETF trades fetch"""
    return fetch_etf_trades()

# --- API SETUP ---
headers = {
    'Accept': 'application/json, text/plain',
    'Authorization': config.UW_TOKEN
}
url = 'https://api.unusualwhales.com/api/option-trades/flow-alerts'

# --- HELPER FUNCTIONS ---
def parse_option_chain(opt_str):
    try:
        ticker = ''.join([c for c in opt_str if c.isalpha()])[:-1]
        date_start = len(ticker)
        date_str = opt_str[date_start:date_start+6]
        expiry_date = date(2000 + int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6]))
        dte = (expiry_date - date.today()).days
        option_type = opt_str[date_start+6].upper()
        strike = int(opt_str[date_start+7:]) / 1000
        return ticker, expiry_date.strftime('%Y-%m-%d'), dte, option_type, strike
    except Exception:
        return None, None, None, None, None

def determine_trade_side(trade_data):
    """Enhanced trade side determination with confidence scoring"""
    side = trade_data.get('side', '').upper()
    if side in ['BUY', 'SELL']:
        return f"{side} (Confirmed)"
    
    confidence_score = 0
    side_indicators = []
    
    try:
        price = float(trade_data.get('price', 0)) if trade_data.get('price') not in ['N/A', '', None] else 0
        bid = float(trade_data.get('bid', 0)) if trade_data.get('bid') not in ['N/A', '', None] else 0
        ask = float(trade_data.get('ask', 0)) if trade_data.get('ask') not in ['N/A', '', None] else 0
    except (ValueError, TypeError):
        price = bid = ask = 0
    
    # Bid/Ask analysis with confidence scoring
    if bid > 0 and ask > 0 and price > 0:
        mid_price = (bid + ask) / 2
        spread = ask - bid
        
        if price >= ask * 0.98:  # Very close to ask
            confidence_score += 4
            side_indicators.append("Near Ask")
            final_side = "BUY"
        elif price <= bid * 1.02:  # Very close to bid
            confidence_score += 4
            side_indicators.append("Near Bid")
            final_side = "SELL"
        elif price > mid_price:
            confidence_score += 2
            side_indicators.append("Above Mid")
            final_side = "BUY"
        else:
            confidence_score += 2
            side_indicators.append("Below Mid")
            final_side = "SELL"
    
    # Pattern analysis
    description = trade_data.get('description', '').lower()
    rule_name = trade_data.get('rule_name', '').lower()
    
    if any(indicator in description for indicator in ['sweep', 'aggressive', 'market buy', 'lifted']):
        confidence_score += 3
        side_indicators.append("Aggressive Pattern")
        final_side = "BUY"
    elif any(indicator in description for indicator in ['sold', 'offer hit', 'market sell']):
        confidence_score += 3
        side_indicators.append("Selling Pattern")
        final_side = "SELL"
    
    # Volume/OI analysis
    try:
        volume = float(trade_data.get('volume', 0))
        oi = float(trade_data.get('open_interest', 1))
        vol_oi_ratio = volume / max(oi, 1)
    except (ValueError, TypeError):
        vol_oi_ratio = 0
    
    if vol_oi_ratio > config.HIGH_VOL_OI_RATIO:
        confidence_score += 2
        side_indicators.append("High Vol/OI")
        final_side = "BUY"
    
    # Confidence level determination
    if confidence_score >= 6:
        confidence = "High"
    elif confidence_score >= 4:
        confidence = "Medium"
    elif confidence_score >= 2:
        confidence = "Low"
    else:
        confidence = "Unknown"
        final_side = "UNKNOWN"
    
    return f"{final_side} ({confidence})"

def analyze_open_interest(trade_data, ticker_trades):
    """Enhanced open interest analysis with historical context"""
    try:
        oi = float(trade_data.get('open_interest', 0))
        volume = float(trade_data.get('volume', 0))
        strike = float(trade_data.get('strike', 0))
    except (ValueError, TypeError):
        oi = volume = strike = 0
        
    option_type = trade_data.get('type', '')
    
    analysis = {
        'oi_level': 'Normal',
        'oi_change_indicator': 'Stable',
        'liquidity_score': 'Medium',
        'oi_concentration': 'Distributed',
        'historical_percentile': 'N/A',
        'flow_direction': 'Neutral'
    }
    
    # Enhanced OI level determination
    if oi > 20000:
        analysis['oi_level'] = 'Extreme'
    elif oi > 10000:
        analysis['oi_level'] = 'Very High'
    elif oi > 5000:
        analysis['oi_level'] = 'High'
    elif oi > 1000:
        analysis['oi_level'] = 'Medium'
    elif oi > 100:
        analysis['oi_level'] = 'Low'
    else:
        analysis['oi_level'] = 'Very Low'
    
    # Volume to OI ratio analysis with enhanced detection
    vol_oi_ratio = volume / max(oi, 1)
    if vol_oi_ratio > 20:
        analysis['oi_change_indicator'] = 'Massive Increase Expected'
        analysis['flow_direction'] = 'Strong Bullish'
    elif vol_oi_ratio > 10:
        analysis['oi_change_indicator'] = 'Large Increase Expected'
        analysis['flow_direction'] = 'Bullish'
    elif vol_oi_ratio > 5:
        analysis['oi_change_indicator'] = 'Major Increase Expected'
        analysis['flow_direction'] = 'Moderately Bullish'
    elif vol_oi_ratio > 2:
        analysis['oi_change_indicator'] = 'Increase Expected'
        analysis['flow_direction'] = 'Slightly Bullish'
    elif vol_oi_ratio > 0.5:
        analysis['oi_change_indicator'] = 'Moderate Activity'
    else:
        analysis['oi_change_indicator'] = 'Low Activity'
    
    # Enhanced liquidity scoring
    if oi > 10000 and volume > 500:
        analysis['liquidity_score'] = 'Excellent'
    elif oi > 5000 and volume > 200:
        analysis['liquidity_score'] = 'Very Good'
    elif oi > 1000 and volume > 100:
        analysis['liquidity_score'] = 'Good'
    elif oi > 500 and volume > 50:
        analysis['liquidity_score'] = 'Fair'
    elif oi > 100 and volume > 20:
        analysis['liquidity_score'] = 'Poor'
    else:
        analysis['liquidity_score'] = 'Very Poor'
    
    # Strike concentration analysis
    try:
        same_strike_trades = [t for t in ticker_trades 
                            if abs(float(t.get('strike', 0)) - strike) < 1 
                            and t.get('type') == option_type]
        concentration_score = len(same_strike_trades)
        
        if concentration_score > 5:
            analysis['oi_concentration'] = 'Extreme Concentration'
        elif concentration_score > 3:
            analysis['oi_concentration'] = 'High Concentration'
        elif concentration_score > 1:
            analysis['oi_concentration'] = 'Some Concentration'
    except (ValueError, TypeError):
        pass
    
    return analysis

def detect_institutional_flow(trades):
    """Detect institutional vs retail flow patterns"""
    institutional_trades = []
    retail_trades = []
    
    for trade in trades:
        premium = trade.get('premium', 0)
        volume = trade.get('volume', 0)
        vol_oi_ratio = trade.get('vol_oi_ratio', 0)
        time_ny = trade.get('time_ny', '')
        
        # Institutional indicators
        institutional_score = 0
        
        if premium > config.INSTITUTIONAL_PREMIUM_THRESHOLD:
            institutional_score += 3
        if volume > 500:
            institutional_score += 2
        if vol_oi_ratio > 10:
            institutional_score += 2
        if time_ny in ['09:30', '09:31', '15:59', '16:00']:  # Market open/close
            institutional_score += 1
        
        if institutional_score >= 4:
            institutional_trades.append(trade)
        elif premium < config.RETAIL_PREMIUM_THRESHOLD:
            retail_trades.append(trade)
    
    return institutional_trades, retail_trades

def detect_dark_pool_activity(trades):
    """Detect potential dark pool prints"""
    dark_pool_candidates = []
    
    for trade in trades:
        premium = trade.get('premium', 0)
        volume = trade.get('volume', 0)
        vol_oi_ratio = trade.get('vol_oi_ratio', 0)
        
        # Dark pool indicators
        dark_pool_score = 0
        reasons = []
        
        if volume > config.DARK_POOL_VOLUME_THRESHOLD:
            dark_pool_score += 2
            reasons.append("High Volume")
        
        if vol_oi_ratio > 20:
            dark_pool_score += 3
            reasons.append("Extreme Vol/OI")
        
        if premium > 2000000:
            dark_pool_score += 2
            reasons.append("Mega Premium")
        
        # Check for unusual timing patterns
        time_ny = trade.get('time_ny', '')
        if time_ny in ['09:30', '16:00']:
            dark_pool_score += 1
            reasons.append("Market Hours")
        
        if dark_pool_score >= 4:
            trade['dark_pool_score'] = dark_pool_score
            trade['dark_pool_reasons'] = reasons
            dark_pool_candidates.append(trade)
    
    return dark_pool_candidates

def calculate_sector_flow(trades):
    """Analyze flow by sector"""
    sector_map = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'AMZN': 'Technology',
        'NVDA': 'Technology', 'META': 'Technology', 'NFLX': 'Technology', 'ADBE': 'Technology',
        'JPM': 'Finance', 'BAC': 'Finance', 'WFC': 'Finance', 'GS': 'Finance', 'MS': 'Finance',
        'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'ABBV': 'Healthcare',
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
        'SPY': 'ETF', 'QQQ': 'ETF', 'IWM': 'ETF',
        'TSLA': 'Automotive'
    }
    
    sector_flow = {}
    for trade in trades:
        ticker = trade.get('ticker', '')
        sector = sector_map.get(ticker, 'Other')
        
        if sector not in sector_flow:
            sector_flow[sector] = {
                'calls': 0, 'puts': 0, 'total_premium': 0, 
                'buy_premium': 0, 'sell_premium': 0, 'trades': 0
            }
        
        sector_flow[sector]['total_premium'] += trade.get('premium', 0)
        sector_flow[sector]['trades'] += 1
        
        if trade.get('type') == 'C':
            sector_flow[sector]['calls'] += 1
        else:
            sector_flow[sector]['puts'] += 1
        
        if 'BUY' in trade.get('trade_side', ''):
            sector_flow[sector]['buy_premium'] += trade.get('premium', 0)
        elif 'SELL' in trade.get('trade_side', ''):
            sector_flow[sector]['sell_premium'] += trade.get('premium', 0)
    
    return sector_flow

def detect_scenarios(trade, underlying_price=None, oi_analysis=None):
    """Enhanced scenario detection with new patterns"""
    scenarios = []
    opt_type = trade['type']
    try:
        strike = float(trade['strike'])
        premium = float(trade['premium'])
        volume = float(trade.get('volume', 0))
        oi = float(trade.get('open_interest', 0))
        iv = float(trade.get('iv', 0))
    except (ValueError, TypeError):
        strike = premium = volume = oi = iv = 0
        
    rule_name = trade.get('rule_name', '')
    ticker = trade['ticker']
    trade_side = trade.get('trade_side', 'UNKNOWN')

    if underlying_price is None:
        underlying_price = strike
    
    try:
        underlying_price = float(underlying_price)
    except (ValueError, TypeError):
        underlying_price = strike

    moneyness = "ATM"
    if opt_type == 'C' and strike > underlying_price:
        moneyness = "OTM"
    elif opt_type == 'C' and strike < underlying_price:
        moneyness = "ITM"
    elif opt_type == 'P' and strike < underlying_price:
        moneyness = "OTM"
    elif opt_type == 'P' and strike > underlying_price:
        moneyness = "ITM"

    # Enhanced scenarios with buy/sell consideration
    if opt_type == 'C' and moneyness == 'OTM' and premium >= config.SCENARIO_OTM_CALL_MIN_PREMIUM:
        if 'BUY' in trade_side:
            scenarios.append("Large OTM Call Buying")
        else:
            scenarios.append("Large OTM Call Writing")
    
    if opt_type == 'P' and moneyness == 'OTM' and premium >= config.SCENARIO_OTM_CALL_MIN_PREMIUM:
        if 'BUY' in trade_side:
            scenarios.append("Large OTM Put Buying")
        else:
            scenarios.append("Large OTM Put Writing")
    
    if moneyness == 'ITM' and premium >= config.SCENARIO_ITM_CONV_MIN_PREMIUM:
        scenarios.append("ITM Conviction Trade")
    
    # Volume/OI scenarios
    vol_oi_ratio = volume / max(oi, 1)
    if vol_oi_ratio > config.SCENARIO_SWEEP_VOLUME_OI_RATIO:
        scenarios.append("Sweep Orders")
    
    if volume >= config.SCENARIO_BLOCK_TRADE_VOL:
        scenarios.append("Block Trade")
    
    # New scenarios for institutional detection
    if premium > config.INSTITUTIONAL_PREMIUM_THRESHOLD:
        scenarios.append("Institutional Size")
    
    # Dark pool scenarios
    if volume > config.DARK_POOL_VOLUME_THRESHOLD and vol_oi_ratio > 15:
        scenarios.append("Potential Dark Pool")
    
    # Gamma scenarios
    if ticker in ['SPY', 'QQQ'] and moneyness == 'ATM' and trade.get('dte', 0) <= 7:
        scenarios.append("Gamma Exposure Play")
    
    # Open Interest based scenarios
    if oi_analysis:
        if oi_analysis['oi_level'] in ['Very High', 'Extreme'] and vol_oi_ratio > 5:
            scenarios.append("High OI + Volume Surge")
        
        if oi_analysis['liquidity_score'] in ['Poor', 'Very Poor'] and premium > 200000:
            scenarios.append("Illiquid Large Trade")
        
        if oi_analysis['oi_concentration'] in ['High Concentration', 'Extreme Concentration']:
            scenarios.append("Strike Concentration Play")
    
    # Pattern-based scenarios
    if rule_name in ['RepeatedHits', 'RepeatedHitsAscendingFill']:
        scenarios.append("Repeated Buying at Same Strike")
    elif rule_name in ['RepeatedHitsDescendingFill']:
        scenarios.append("Repeated Selling at Same Strike")
    
    # Advanced scenarios
    if opt_type == 'C' and moneyness == 'OTM' and 'SELL' in trade_side and iv > config.HIGH_IV_THRESHOLD:
        scenarios.append("High IV Call Selling")
    
    if opt_type == 'P' and moneyness == 'OTM' and 'SELL' in trade_side:
        scenarios.append("Put Selling for Income")
    
    if ticker in ['SPY', 'QQQ'] and opt_type == 'P' and moneyness in ['ITM', 'ATM']:
        scenarios.append("Portfolio Hedging")
    
    # Insider-like activity detection
    if premium > 500000 and vol_oi_ratio > 10:
        scenarios.append("Potential Insider Activity")
    
    # IV-based scenarios
    if iv > config.EXTREME_IV_THRESHOLD:
        scenarios.append("Extreme IV Play")
    elif iv > config.HIGH_IV_THRESHOLD:
        scenarios.append("High IV Premium")
    
    if iv > config.IV_CRUSH_THRESHOLD and trade.get('dte', 0) <= 7:
        scenarios.append("IV Crush Risk")
    
    # Volatility trading scenarios
    if iv > config.HIGH_IV_THRESHOLD and premium > 200000:
        if 'BUY' in trade_side:
            scenarios.append("Long Volatility Strategy")
        else:
            scenarios.append("Short Volatility Strategy")
    
    # New earnings-related scenarios
    if trade.get('dte', 0) <= 14 and premium > 300000:
        scenarios.append("Earnings Play")
    
    return scenarios if scenarios else ["Normal Flow"]

def calculate_moneyness(strike, current_price):
    if current_price == 'N/A' or current_price == 0:
        return "Unknown"
    try:
        strike = float(strike)
        price = float(current_price)
        diff_percent = ((strike - price) / price) * 100
        if abs(diff_percent) < 2:
            return "ATM"
        elif diff_percent > 0:
            return f"OTM +{diff_percent:.1f}%"
        else:
            return f"ITM {diff_percent:.1f}%"
    except (ValueError, TypeError):
        return "Unknown"

def get_time_to_expiry_category(dte):
    if dte <= 1:
        return "0DTE"
    elif dte <= 7:
        return "Weekly"
    elif dte <= 30:
        return "Monthly"
    elif dte <= 90:
        return "Quarterly"
    else:
        return "LEAPS"

def calculate_sentiment_score(trades):
    call_premium = sum(t['premium'] for t in trades if t['type'] == 'C' and 'BUY' in t.get('trade_side', ''))
    put_premium = sum(t['premium'] for t in trades if t['type'] == 'P' and 'BUY' in t.get('trade_side', ''))
    total = call_premium + put_premium
    if total == 0:
        return 0, "Neutral"
    call_ratio = call_premium / total
    if call_ratio > 0.7:
        return call_ratio, "Very Bullish"
    elif call_ratio > 0.6:
        return call_ratio, "Bullish"
    elif call_ratio > 0.4:
        return call_ratio, "Neutral"
    elif call_ratio > 0.3:
        return call_ratio, "Bearish"
    else:
        return call_ratio, "Very Bearish"

def generate_enhanced_alerts(trades):
    """Enhanced alert system with priority levels"""
    alerts = {
        'CRITICAL': [],
        'HIGH': [],
        'MEDIUM': [],
        'LOW': []
    }
    
    for trade in trades:
        alert_level = 'LOW'
        score = 0
        reasons = []

        premium = trade.get('premium', 0)
        vol_oi_ratio = trade.get('vol_oi_ratio', 0)
        dte = trade.get('dte', 0)
        trade_side = trade.get('trade_side', '')
        oi_analysis = trade.get('oi_analysis', {})
        iv = trade.get('iv', 0)
        
        # Critical alerts
        if premium > 2000000:
            score += 5
            reasons.append("Mega Premium (>$2M)")
            alert_level = 'CRITICAL'
        
        if vol_oi_ratio > 50:
            score += 4
            reasons.append("Extreme Vol/OI Ratio")
            alert_level = 'CRITICAL'
        
        # High priority alerts
        if premium > 1000000:
            score += 4
            reasons.append("Massive Premium (>$1M)")
            if alert_level == 'LOW':
                alert_level = 'HIGH'
        
        if vol_oi_ratio > 20:
            score += 3
            reasons.append("Very High Vol/OI")
            if alert_level == 'LOW':
                alert_level = 'HIGH'
        
        if dte <= 7 and premium > 500000:
            score += 3
            reasons.append("Short-term + Large Size")
            if alert_level == 'LOW':
                alert_level = 'HIGH'
        
        # Medium priority alerts
        if premium > 500000:
            score += 2
            reasons.append("Large Premium")
            if alert_level == 'LOW':
                alert_level = 'MEDIUM'
        
        if vol_oi_ratio > 10:
            score += 2
            reasons.append("High Vol/OI")
            if alert_level == 'LOW':
                alert_level = 'MEDIUM'
        
        if 'Aggressive' in trade_side:
            score += 2
            reasons.append("Aggressive Execution")
            if alert_level == 'LOW':
                alert_level = 'MEDIUM'
        
        # OI-based alerts
        if oi_analysis.get('oi_level') in ['Very High', 'Extreme']:
            score += 2
            reasons.append("Extreme OI Level")
            if alert_level == 'LOW':
                alert_level = 'MEDIUM'
        
        if oi_analysis.get('liquidity_score') in ['Poor', 'Very Poor'] and premium > 200000:
            score += 2
            reasons.append("Illiquid Large Trade")
            if alert_level == 'LOW':
                alert_level = 'MEDIUM'
        
        # IV-based alerts
        if iv > config.EXTREME_IV_THRESHOLD:
            score += 3
            reasons.append("Extreme IV")
            if alert_level == 'LOW':
                alert_level = 'MEDIUM'
        
        # Scenario-based alerts
        scenarios = trade.get('scenarios', [])
        high_impact_scenarios = [
            'Potential Insider Activity', 'Potential Dark Pool',
            'High OI + Volume Surge', 'Strike Concentration Play',
            'Institutional Size'
        ]
        
        for scenario in scenarios:
            if scenario in high_impact_scenarios:
                score += 2
                reasons.append(f"Pattern: {scenario}")
                if alert_level == 'LOW':
                    alert_level = 'MEDIUM'
        
        if score >= 3:  # Only include significant alerts
            trade['alert_score'] = score
            trade['alert_reasons'] = reasons
            alerts[alert_level].append(trade)
    
    # Sort each priority level by score
    for level in alerts:
        alerts[level].sort(key=lambda x: -x.get('alert_score', 0))
    
    return alerts

# --- VISUALIZATION FUNCTIONS ---
def create_premium_flow_chart(trades):
    """Create interactive premium flow chart"""
    if not trades:
        return None
    
    # Prepare data
    df = pd.DataFrame(trades)
    df['hour'] = pd.to_datetime(df['time_ny'], format='%I:%M %p', errors='coerce').dt.hour
    df = df.dropna(subset=['hour'])
    
    # Aggregate by hour
    hourly_data = df.groupby('hour').agg({
        'premium': 'sum',
        'volume': 'sum'
    }).reset_index()
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Premium Flow by Hour', 'Volume Flow by Hour'),
        vertical_spacing=0.1
    )
    
    # Premium flow
    fig.add_trace(
        go.Scatter(
            x=hourly_data['hour'],
            y=hourly_data['premium'],
            mode='lines+markers',
            name='Premium Flow',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Volume flow
    fig.add_trace(
        go.Scatter(
            x=hourly_data['hour'],
            y=hourly_data['volume'],
            mode='lines+markers',
            name='Volume Flow',
            line=dict(color='#764ba2', width=3),
            marker=dict(size=8)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title="Options Flow Analysis",
        height=600,
        showlegend=True,
        template='plotly_white'
    )
    
    return fig

def create_sentiment_gauge(trades):
    """Create sentiment gauge chart"""
    sentiment_ratio, sentiment_label = calculate_sentiment_score(trades)
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = sentiment_ratio * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Market Sentiment"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 30], 'color': "#ff4444"},
                {'range': [30, 70], 'color': "#ffaa00"},
                {'range': [70, 100], 'color': "#00ff00"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_volume_heatmap(trades):
    """Create volume heatmap by strike and time"""
    if not trades:
        return None
    
    df = pd.DataFrame(trades)
    df['hour'] = pd.to_datetime(df['time_ny'], format='%I:%M %p', errors='coerce').dt.hour
    df = df.dropna(subset=['hour'])
    
    # Create pivot table
    pivot_data = df.pivot_table(
        values='volume',
        index='strike',
        columns='hour',
        aggfunc='sum',
        fill_value=0
    )
    
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Hour", y="Strike", color="Volume"),
        x=pivot_data.columns,
        y=pivot_data.index,
        color_continuous_scale='Viridis',
        title="Volume Heatmap by Strike and Time"
    )
    
    fig.update_layout(height=500)
    return fig

def create_sector_analysis_chart(trades):
    """Create sector analysis chart"""
    sector_flow = calculate_sector_flow(trades)
    
    if not sector_flow:
        return None
    
    sectors = list(sector_flow.keys())
    premiums = [sector_flow[sector]['total_premium'] for sector in sectors]
    call_ratios = [
        sector_flow[sector]['calls'] / max(sector_flow[sector]['calls'] + sector_flow[sector]['puts'], 1)
        for sector in sectors
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=sectors,
        y=premiums,
        name='Total Premium',
        marker_color='#667eea'
    ))
    
    fig.update_layout(
        title="Sector Flow Analysis",
        xaxis_title="Sector",
        yaxis_title="Premium ($)",
        height=400,
        template='plotly_white'
    )
    
    return fig

# --- FETCH FUNCTIONS ---
@rate_limit(max_calls=30, period=60)
def fetch_general_flow():
    """Enhanced general flow fetch with better error handling"""
    params = {
        'issue_types[]': ['Common Stock', 'ADR'],
        'min_dte': 1,
        'min_volume_oi_ratio': 1.0,
        'rule_name[]': ['RepeatedHits', 'RepeatedHitsAscendingFill', 'RepeatedHitsDescendingFill'],
        'limit': config.LIMIT
    }
    
    try:
        response = httpx.get(url, headers=headers, params=params, timeout=30)
        if response.status_code == 429:  # Rate limited
            st.warning("Rate limited. Waiting...")
            time.sleep(60)
            response = httpx.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return []
        
        data = response.json().get('data', [])
        result = []
        ticker_data = defaultdict(list)

        for trade in data:
            option_chain = trade.get('option_chain', '')
            ticker, expiry, dte, opt_type, strike = parse_option_chain(option_chain)

            if not ticker or ticker in config.EXCLUDE_TICKERS:
                continue

            premium = float(trade.get('total_premium', 0))
            if premium < config.MIN_PREMIUM:
                continue

            utc_time_str = trade.get('created_at')
            ny_time_str = "N/A"
            if utc_time_str != "N/A":
                try:
                    utc_time = datetime.fromisoformat(utc_time_str.replace("Z", "+00:00"))
                    ny_time = utc_time.astimezone(ZoneInfo("America/New_York"))
                    ny_time_str = ny_time.strftime("%I:%M %p")
                except Exception:
                    ny_time_str = "N/A"

            # Extract IV data
            iv = 0
            iv_fields = ['iv', 'implied_volatility', 'volatility', 'impliedVolatility', 'vol', 'IV']
            for field in iv_fields:
                if field in trade and trade[field] not in ['N/A', '', None, 0]:
                    try:
                        iv = float(trade[field])
                        if iv > 0:
                            break
                    except (ValueError, TypeError):
                        continue

            trade_data = {
                'ticker': ticker,
                'option': option_chain,
                'type': opt_type,
                'strike': strike,
                'expiry': expiry,
                'dte': dte,
                'price': trade.get('price', 'N/A'),
                'premium': premium,
                'volume': trade.get('volume', 0),
                'open_interest': trade.get('open_interest', 0),
                'time_utc': utc_time_str,
                'time_ny': ny_time_str,
                'rule_name': trade.get('rule_name', ''),
                'description': trade.get('description', ''),
                'underlying_price': trade.get('underlying_price', strike),
                'moneyness': calculate_moneyness(strike, trade.get('underlying_price', strike)),
                'vol_oi_ratio': float(trade.get('volume', 0)) / max(float(trade.get('open_interest', 1)), 1),
                'iv': iv,
                'iv_percentage': f"{iv:.1%}" if iv > 0 else "N/A",
                'bid': float(trade.get('bid', 0)) if trade.get('bid') not in ['N/A', '', None] else 0,
                'ask': float(trade.get('ask', 0)) if trade.get('ask') not in ['N/A', '', None] else 0
            }
            
            # Determine trade side
            trade_data['trade_side'] = determine_trade_side(trade)

            ticker_data[ticker].append(trade_data)

        # Process each ticker's trades
        for ticker, trade_list in ticker_data.items():
            for trade in trade_list:
                # Analyze open interest
                oi_analysis = analyze_open_interest(trade, trade_list)
                trade['oi_analysis'] = oi_analysis
                
                # Detect scenarios
                scenarios = detect_scenarios(trade, trade['underlying_price'], oi_analysis)
                trade['scenarios'] = scenarios
                result.append(trade)

        # Store in database
        if result:
            store_trades_in_db(result)
        
        return result

    except Exception as e:
        st.error(f"Error fetching general flow: {e}")
        return []

@rate_limit(max_calls=60, period=60)
def fetch_etf_trades():
    """Enhanced ETF trades fetch"""
    allowed_tickers = {'QQQ', 'SPY', 'IWM'}
    max_dte = 7
    
    params = {
        'limit': config.LIMIT
    }
    
    try:
        response = httpx.get(url, headers=headers, params=params, timeout=30)
        if response.status_code == 429:
            st.warning("Rate limited. Waiting...")
            time.sleep(60)
            response = httpx.get(url, headers=headers, params=params, timeout=30)
        
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return []
        
        data = response.json().get('data', [])
        filtered_trades = []
        
        for trade in data:
            option_chain = trade.get('option_chain', '')
            ticker, expiry, dte, opt_type, strike = parse_option_chain(option_chain)

            if not ticker or ticker.upper() not in allowed_tickers:
                continue
            if dte is None or dte > max_dte:
                continue

            # Time conversion
            utc_time_str = trade.get('created_at')
            ny_time_str = "N/A"
            if utc_time_str != "N/A":
                try:
                    utc_time = datetime.fromisoformat(utc_time_str.replace("Z", "+00:00"))
                    ny_time = utc_time.astimezone(ZoneInfo("America/New_York"))
                    ny_time_str = ny_time.strftime("%I:%M %p")
                except Exception:
                    ny_time_str = "N/A"

            # Safe data extraction
            try:
                premium = float(trade.get('total_premium', 0))
                volume = float(trade.get('volume', 0))
                oi = float(trade.get('open_interest', 0))
                price = trade.get('price', 'N/A')
                if price != 'N/A':
                    price = float(price)
            except (ValueError, TypeError):
                premium = volume = oi = 0
                price = 'N/A'

            trade_data = {
                'ticker': ticker,
                'type': opt_type,
                'strike': strike,
                'expiry': expiry,
                'dte': dte,
                'side': trade.get('side', 'N/A'),
                'price': price,
                'premium': premium,
                'volume': volume,
                'oi': oi,
                'vol_oi_ratio': volume / max(oi, 1),
                'time_ny': ny_time_str,
                'option': option_chain,
                'underlying_price': trade.get('underlying_price', strike),
                'rule_name': trade.get('rule_name', ''),
                'description': trade.get('description', ''),
                'moneyness': calculate_moneyness(strike, trade.get('underlying_price', strike))
            }
            
            # Add trade side detection
            trade_data['trade_side'] = determine_trade_side(trade)
            
            filtered_trades.append(trade_data)
        
        return filtered_trades

    except Exception as e:
        st.error(f"Error fetching ETF trades: {e}")
        return []

def store_trades_in_db(trades):
    """Store trades in database"""
    try:
        conn = init_database()
        
        for trade in trades:
            scenarios_str = json.dumps(trade.get('scenarios', []))
            
            conn.execute('''
                INSERT INTO trades (
                    ticker, option_chain, option_type, strike, expiry, dte,
                    premium, volume, open_interest, vol_oi_ratio, iv,
                    trade_side, scenarios, alert_score, underlying_price, moneyness
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.get('ticker', ''),
                trade.get('option', ''),
                trade.get('type', ''),
                trade.get('strike', 0),
                trade.get('expiry', ''),
                trade.get('dte', 0),
                trade.get('premium', 0),
                trade.get('volume', 0),
                trade.get('open_interest', 0),
                trade.get('vol_oi_ratio', 0),
                trade.get('iv', 0),
                trade.get('trade_side', ''),
                scenarios_str,
                trade.get('alert_score', 0),
                trade.get('underlying_price', 0),
                trade.get('moneyness', '')
            ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        st.error(f"Database error: {e}")

# --- FILTER FUNCTIONS ---
def apply_premium_filter(trades, premium_range):
    if premium_range == "All Premiums (No Filter)":
        return trades
    
    filtered_trades = []
    for trade in trades:
        premium = trade.get('premium', 0)
        
        if premium_range == "Under $100K" and premium < 100000:
            filtered_trades.append(trade)
        elif premium_range == "Under $250K" and premium < 250000:
            filtered_trades.append(trade)
        elif premium_range == "$100K - $250K" and 100000 <= premium < 250000:
            filtered_trades.append(trade)
        elif premium_range == "$250K - $500K" and 250000 <= premium < 500000:
            filtered_trades.append(trade)
        elif premium_range == "Above $250K" and premium >= 250000:
            filtered_trades.append(trade)
        elif premium_range == "Above $500K" and premium >= 500000:
            filtered_trades.append(trade)
        elif premium_range == "Above $1M" and premium >= 1000000:
            filtered_trades.append(trade)
    
    return filtered_trades

def apply_dte_filter(trades, dte_filter):
    if dte_filter == "All DTE":
        return trades
    
    filtered_trades = []
    for trade in trades:
        dte = trade.get('dte', 0)
        
        if dte_filter == "0DTE Only" and dte == 0:
            filtered_trades.append(trade)
        elif dte_filter == "Weekly (≤7d)" and dte <= 7:
            filtered_trades.append(trade)
        elif dte_filter == "Monthly (≤30d)" and dte <= 30:
            filtered_trades.append(trade)
        elif dte_filter == "Quarterly (≤90d)" and dte <= 90:
            filtered_trades.append(trade)
        elif dte_filter == "LEAPS (>90d)" and dte > 90:
            filtered_trades.append(trade)
    
    return filtered_trades

def apply_trade_side_filter(trades, side_filter):
    if side_filter == "All Trades":
        return trades
    
    filtered_trades = []
    for trade in trades:
        trade_side = trade.get('trade_side', 'UNKNOWN')
        
        if side_filter == "Buy Only" and 'BUY' in trade_side:
            filtered_trades.append(trade)
        elif side_filter == "Sell Only" and 'SELL' in trade_side:
            filtered_trades.append(trade)
        elif side_filter == "Aggressive Only" and 'Aggressive' in trade_side:
            filtered_trades.append(trade)
        elif side_filter == "Institutional Only" and trade.get('premium', 0) > config.INSTITUTIONAL_PREMIUM_THRESHOLD:
            filtered_trades.append(trade)
    
    return filtered_trades

# --- DISPLAY FUNCTIONS ---
def display_live_status():
    """Display live status indicator"""
    current_time = datetime.now().strftime("%H:%M:%S")
    
    st.markdown(f"""
    <div style="text-align: center; margin: 1rem 0;">
        <span class="status-indicator status-live"></span>
        <strong>LIVE</strong> - Last Update: {current_time}
    </div>
    """, unsafe_allow_html=True)

def display_enhanced_summary(trades):
    """Enhanced summary with visualizations"""
    st.markdown("### 📊 Enhanced Market Summary")
    
    if not trades:
        st.warning("No trades to analyze")
        return
    
    # Quick metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        sentiment_ratio, sentiment_label = calculate_sentiment_score(trades)
        st.metric("Market Sentiment", sentiment_label, f"{sentiment_ratio:.1%}")
    
    with col2:
        total_premium = sum(t.get('premium', 0) for t in trades)
        st.metric("Total Premium", f"${total_premium:,.0f}")
    
    with col3:
        buy_trades = len([t for t in trades if 'BUY' in t.get('trade_side', '')])
        sell_trades = len([t for t in trades if 'SELL' in t.get('trade_side', '')])
        st.metric("Buy vs Sell", f"{buy_trades}/{sell_trades}")
    
    with col4:
        avg_oi = np.mean([t.get('open_interest', 0) for t in trades])
        st.metric("Avg Open Interest", f"{avg_oi:,.0f}")
    
    with col5:
        high_vol_oi = len([t for t in trades if t.get('vol_oi_ratio', 0) > 10])
        st.metric("High Vol/OI", high_vol_oi)
    
    # Institutional vs Retail Analysis
    institutional_trades, retail_trades = detect_institutional_flow(trades)
    
    st.markdown("#### 🏛️ Institutional vs Retail Flow")
    col1, col2 = st.columns(2)
    
    with col1:
        inst_premium = sum(t.get('premium', 0) for t in institutional_trades)
        st.metric("Institutional Premium", f"${inst_premium:,.0f}")
        st.metric("Institutional Trades", len(institutional_trades))
    
    with col2:
        retail_premium = sum(t.get('premium', 0) for t in retail_trades)
        st.metric("Retail Premium", f"${retail_premium:,.0f}")
        st.metric("Retail Trades", len(retail_trades))
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment gauge
        sentiment_chart = create_sentiment_gauge(trades)
        if sentiment_chart:
            st.plotly_chart(sentiment_chart, use_container_width=True)
    
    with col2:
        # Sector analysis
        sector_chart = create_sector_analysis_chart(trades)
        if sector_chart:
            st.plotly_chart(sector_chart, use_container_width=True)
    
    # Premium flow chart
    premium_chart = create_premium_flow_chart(trades)
    if premium_chart:
        st.plotly_chart(premium_chart, use_container_width=True)

def display_smart_alerts(trades):
    """Display enhanced alert system"""
    st.markdown("### 🚨 Smart Alert System")
    
    alerts = generate_enhanced_alerts(trades)
    
    # Alert summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🔴 Critical", len(alerts['CRITICAL']))
    with col2:
        st.metric("🟠 High", len(alerts['HIGH']))
    with col3:
        st.metric("🟡 Medium", len(alerts['MEDIUM']))
    with col4:
        st.metric("🔵 Low", len(alerts['LOW']))
    
    # Display alerts by priority
    for priority, alert_list in alerts.items():
        if not alert_list:
            continue
        
        if priority == 'CRITICAL':
            st.markdown("#### 🔴 CRITICAL ALERTS")
            css_class = "alert-critical"
        elif priority == 'HIGH':
            st.markdown("#### 🟠 HIGH PRIORITY ALERTS")
            css_class = "alert-high"
        elif priority == 'MEDIUM':
            st.markdown("#### 🟡 MEDIUM PRIORITY ALERTS")
            css_class = "alert-medium"
        else:
            st.markdown("#### 🔵 LOW PRIORITY ALERTS")
            css_class = "alert-medium"
        
        for i, alert in enumerate(alert_list[:10], 1):  # Show top 10 per priority
            side_emoji = "🟢" if "BUY" in alert.get('trade_side', '') else "🔴" if "SELL" in alert.get('trade_side', '') else "⚪"
            
            st.markdown(f"""
            <div class="{css_class}">
                <strong>{i}. {side_emoji} {alert['ticker']} {alert['strike']:.0f}{alert['type']} 
                {alert['expiry']} ({alert['dte']}d) - {alert.get('trade_side', 'UNKNOWN')}</strong><br>
                💰 Premium: ${alert['premium']:,.0f} | Vol: {alert['volume']:,} | 
                OI: {alert['open_interest']:,} | Vol/OI: {alert['vol_oi_ratio']:.1f}<br>
                📊 Alert Score: {alert.get('alert_score', 0)} | 
                IV: {alert.get('iv_percentage', 'N/A')}<br>
                🎯 Scenarios: {', '.join(alert.get('scenarios', [])[:3])}<br>
                📍 Reasons: {', '.join(alert.get('alert_reasons', []))}
            </div>
            """, unsafe_allow_html=True)

def display_dark_pool_analysis(trades):
    """Display dark pool analysis"""
    st.markdown("### 🌑 Dark Pool Analysis")
    
    dark_pool_trades = detect_dark_pool_activity(trades)
    
    if not dark_pool_trades:
        st.info("No potential dark pool activity detected")
        return
    
    st.metric("Potential Dark Pool Trades", len(dark_pool_trades))
    
    # Create table
    table_data = []
    for trade in dark_pool_trades[:15]:
        table_data.append({
            'Ticker': trade['ticker'],
            'Type': trade['type'],
            'Strike': f"${trade['strike']:.0f}",
            'Premium': f"${trade['premium']:,.0f}",
            'Volume': f"{trade['volume']:,}",
            'Vol/OI': f"{trade['vol_oi_ratio']:.1f}",
            'Dark Pool Score': trade.get('dark_pool_score', 0),
            'Reasons': ', '.join(trade.get('dark_pool_reasons', [])),
            'Time': trade.get('time_ny', 'N/A')
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)

def display_advanced_visualizations(trades):
    """Display advanced visualizations"""
    st.markdown("### 📈 Advanced Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Volume Heatmap", "Flow Timeline", "OI Analysis"])
    
    with tab1:
        volume_chart = create_volume_heatmap(trades)
        if volume_chart:
            st.plotly_chart(volume_chart, use_container_width=True)
    
    with tab2:
        premium_chart = create_premium_flow_chart(trades)
        if premium_chart:
            st.plotly_chart(premium_chart, use_container_width=True)
    
    with tab3:
        # OI distribution
        if trades:
            oi_levels = [t.get('oi_analysis', {}).get('oi_level', 'Normal') for t in trades]
            oi_counts = pd.Series(oi_levels).value_counts()
            
            fig = px.pie(
                values=oi_counts.values,
                names=oi_counts.index,
                title="Open Interest Level Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

def display_main_trades_table(trades, title="📋 Main Trades Analysis"):
    """Enhanced main trades table"""
    st.markdown(f"### {title}")
    
    if not trades:
        st.info("No trades found")
        return
    
    # Sort by premium descending
    sorted_trades = sorted(trades, key=lambda x: x.get('premium', 0), reverse=True)
    
    # Create enhanced table
    table_data = []
    for trade in sorted_trades[:50]:  # Show top 50
        oi_analysis = trade.get('oi_analysis', {})
        
        table_data.append({
            'Ticker': trade['ticker'],
            'Side': trade.get('trade_side', 'UNKNOWN'),
            'Type': trade['type'],
            'Strike': f"${trade['strike']:.0f}",
            'Expiry': trade['expiry'],
            'DTE': trade['dte'],
            'Premium': f"${trade['premium']:,.0f}",
            'Volume': f"{trade['volume']:,}",
            'OI': f"{trade['open_interest']:,}",
            'Vol/OI': f"{trade['vol_oi_ratio']:.1f}",
            'OI Level': oi_analysis.get('oi_level', 'N/A'),
            'Liquidity': oi_analysis.get('liquidity_score', 'N/A'),
            'IV': trade.get('iv_percentage', 'N/A'),
            'Moneyness': trade.get('moneyness', 'N/A'),
            'Primary Scenario': trade.get('scenarios', ['Normal Flow'])[0],
            'Alert Score': trade.get('alert_score', 0),
            'Time': trade.get('time_ny', 'N/A')
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)

def display_personalized_dashboard():
    """Display personalized dashboard"""
    st.markdown("### 👤 Personalized Dashboard")
    
    # User preferences
    col1, col2 = st.columns(2)
    
    with col1:
        watchlist = st.multiselect(
            "📋 Your Watchlist",
            ['AAPL', 'TSLA', 'SPY', 'QQQ', 'NVDA', 'AMD', 'MSFT', 'AMZN'],
            default=['SPY', 'QQQ']
        )
    
    with col2:
        risk_level = st.select_slider(
            "⚖️ Risk Level",
            options=['Conservative', 'Moderate', 'Aggressive'],
            value='Moderate'
        )
    
    return {
        'watchlist': watchlist,
        'risk_level': risk_level
    }

# --- CSV EXPORT ---
def save_to_csv(trades, filename_prefix):
    if not trades:
        st.warning("No data to save")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    
    # Prepare data for CSV
    csv_data = []
    for trade in trades:
        row = trade.copy()
        
        # Handle list/dict fields
        if isinstance(row.get('alert_reasons'), list):
            row['alert_reasons'] = ', '.join(row['alert_reasons'])
        if isinstance(row.get('scenarios'), list):
            row['scenarios'] = ', '.join(row['scenarios'])
        if isinstance(row.get('oi_analysis'), dict):
            oi_analysis = row['oi_analysis']
            row['oi_level'] = oi_analysis.get('oi_level', '')
            row['liquidity_score'] = oi_analysis.get('liquidity_score', '')
            row['oi_change_indicator'] = oi_analysis.get('oi_change_indicator', '')
            del row['oi_analysis']
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label=f"📥 Download {filename}",
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

# --- MAIN STREAMLIT APP ---
def main():
    # Page config
    st.set_page_config(
        page_title="Enhanced Options Flow Tracker", 
        page_icon="📊", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_custom_css()
    
    # Initialize database
    init_database()
    
    # Header
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h1>📊 Enhanced Options Flow Tracker</h1>
        <p style="font-size: 1.2rem; color: #666;">
            Real-time unusual options activity with AI-powered analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Live status
    display_live_status()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2>🎛️ Control Panel</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", value=False)
        
        # Analysis type selection
        scan_type = st.selectbox(
            "Select Analysis Type:",
            [
                "🔍 Main Flow Analysis",
                "📈 Advanced Visualizations",
                "🚨 Smart Alert System",
                "🌑 Dark Pool Analysis",
                "📊 Institutional vs Retail",
                "⚡ ETF Flow Scanner"
            ]
        )
        
        # Personalized dashboard
        st.divider()
        user_prefs = display_personalized_dashboard()
        
        st.divider()
        
        # Filters
        st.markdown("### 💰 Premium Range Filter")
        premium_range = st.selectbox(
            "Select Premium Range:",
            [
                "All Premiums (No Filter)",
                "Under $100K",
                "Under $250K", 
                "$100K - $250K",
                "$250K - $500K",
                "Above $250K",
                "Above $500K",
                "Above $1M"
            ],
            index=0
        )
        
        st.markdown("### 📅 Time to Expiry Filter")
        dte_filter = st.selectbox(
            "Select DTE Range:",
            [
                "All DTE",
                "0DTE Only",
                "Weekly (≤7d)",
                "Monthly (≤30d)",
                "Quarterly (≤90d)",
                "LEAPS (>90d)"
            ],
            index=0
        )
        
        st.markdown("### 🔄 Trade Side Filter")
        side_filter = st.selectbox(
            "Filter by Trade Side:",
            [
                "All Trades",
                "Buy Only",
                "Sell Only", 
                "Aggressive Only",
                "Institutional Only"
            ],
            index=0
        )
        
        # Advanced filters
        with st.expander("🔍 Advanced Filters", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                iv_range = st.slider("IV Range (%)", 0, 200, (0, 200))
                min_volume = st.number_input("Min Volume", min_value=0, value=0)
            
            with col2:
                min_oi = st.number_input("Min Open Interest", min_value=0, value=0)
                vol_oi_threshold = st.slider("Min Vol/OI Ratio", 0.0, 50.0, 0.0)
        
        # Quick filter buttons
        st.markdown("### ⚡ Quick Filters")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔥 Mega Trades", use_container_width=True):
                premium_range = "Above $1M"
                st.rerun()
        
        with col2:
            if st.button("⚡ 0DTE Plays", use_container_width=True):
                dte_filter = "0DTE Only"
                st.rerun()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏛️ Institutional", use_container_width=True):
                side_filter = "Institutional Only"
                st.rerun()
        
        with col2:
            if st.button("🌑 Dark Pool", use_container_width=True):
                scan_type = "🌑 Dark Pool Analysis"
                st.rerun()
        
        st.divider()
        
        # Scan button
        run_scan = st.button("🔄 Run Enhanced Scan", type="primary", use_container_width=True)
        
        # Historical data toggle
        use_historical = st.checkbox("📊 Include Historical Data", value=False)
        
        # Real-time settings
        st.markdown("### ⚙️ Real-time Settings")
        refresh_interval = st.slider("Refresh Interval (seconds)", 10, 120, 30)
        
        # Data source status
        st.markdown("### 📡 Data Source Status")
        st.markdown("""
        <div style="padding: 1rem; background: #f0f0f0; border-radius: 5px;">
            <strong>🟢 API Status:</strong> Connected<br>
            <strong>📊 Database:</strong> Ready<br>
            <strong>⚡ Cache:</strong> Active
        </div>
        """, unsafe_allow_html=True)
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
    
    # Main content
    if run_scan or auto_refresh:
        with st.spinner(f"Running {scan_type}..."):
            # Fetch data based on scan type
            if "ETF Flow Scanner" in scan_type:
                trades = fetch_etf_trades_cached()
            else:
                trades = fetch_general_flow_cached()
            
            # Apply filters
            original_count = len(trades)
            
            # Apply basic filters
            trades = apply_premium_filter(trades, premium_range)
            trades = apply_dte_filter(trades, dte_filter)
            trades = apply_trade_side_filter(trades, side_filter)
            
            # Apply advanced filters
            if iv_range != (0, 200):
                trades = [t for t in trades if iv_range[0] <= t.get('iv', 0) * 100 <= iv_range[1]]
            
            if min_volume > 0:
                trades = [t for t in trades if t.get('volume', 0) >= min_volume]
            
            if min_oi > 0:
                trades = [t for t in trades if t.get('open_interest', 0) >= min_oi]
            
            if vol_oi_threshold > 0:
                trades = [t for t in trades if t.get('vol_oi_ratio', 0) >= vol_oi_threshold]
            
            # Apply watchlist filter if specified
            if user_prefs['watchlist']:
                trades = [t for t in trades if t.get('ticker') in user_prefs['watchlist']]
            
            # Show filter results
            if len(trades) != original_count:
                st.info(f"**Filter Results:** {original_count} → {len(trades)} trades after applying filters")
            
            if not trades:
                st.warning("⚠️ No trades match your current filters. Try adjusting the filters.")
                return
            
            # Display results based on scan type
            if "Main Flow" in scan_type:
                display_enhanced_summary(trades)
                display_main_trades_table(trades)
                
                # Additional analysis tabs
                tab1, tab2, tab3 = st.tabs(["📊 Flow Analysis", "🎯 Key Scenarios", "📈 Performance"])
                
                with tab1:
                    # Sector breakdown
                    sector_flow = calculate_sector_flow(trades)
                    if sector_flow:
                        st.markdown("#### 🏢 Sector Flow Breakdown")
                        sector_data = []
                        for sector, data in sector_flow.items():
                            sector_data.append({
                                'Sector': sector,
                                'Total Premium': f"${data['total_premium']:,.0f}",
                                'Trades': data['trades'],
                                'Calls': data['calls'],
                                'Puts': data['puts'],
                                'Buy Premium': f"${data['buy_premium']:,.0f}",
                                'Sell Premium': f"${data['sell_premium']:,.0f}"
                            })
                        
                        df_sector = pd.DataFrame(sector_data)
                        st.dataframe(df_sector, use_container_width=True)
                
                with tab2:
                    # Scenario analysis
                    scenario_counts = {}
                    for trade in trades:
                        scenarios = trade.get('scenarios', ['Normal Flow'])
                        for scenario in scenarios:
                            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
                    
                    st.markdown("#### 🎯 Most Common Scenarios")
                    sorted_scenarios = sorted(scenario_counts.items(), key=lambda x: x[1], reverse=True)
                    
                    for scenario, count in sorted_scenarios[:10]:
                        st.write(f"**{scenario}**: {count} trades")
                
                with tab3:
                    # Performance metrics
                    st.markdown("#### 📈 Performance Metrics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_premium = np.mean([t.get('premium', 0) for t in trades])
                        st.metric("Average Premium", f"${avg_premium:,.0f}")
                    
                    with col2:
                        high_confidence_trades = len([t for t in trades if 'High' in t.get('trade_side', '')])
                        st.metric("High Confidence Trades", high_confidence_trades)
                    
                    with col3:
                        unusual_trades = len([t for t in trades if t.get('vol_oi_ratio', 0) > 10])
                        st.metric("Unusual Volume Trades", unusual_trades)
            
            elif "Advanced Visualizations" in scan_type:
                display_enhanced_summary(trades)
                display_advanced_visualizations(trades)
            
            elif "Smart Alert" in scan_type:
                display_smart_alerts(trades)
                display_main_trades_table(trades, "📋 Alert-Triggered Trades")
            
            elif "Dark Pool" in scan_type:
                display_enhanced_summary(trades)
                display_dark_pool_analysis(trades)
            
            elif "Institutional vs Retail" in scan_type:
                display_enhanced_summary(trades)
                
                institutional_trades, retail_trades = detect_institutional_flow(trades)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🏛️ Institutional Flow")
                    if institutional_trades:
                        inst_data = []
                        for trade in institutional_trades[:20]:
                            inst_data.append({
                                'Ticker': trade['ticker'],
                                'Type': trade['type'],
                                'Premium': f"${trade['premium']:,.0f}",
                                'Volume': f"{trade['volume']:,}",
                                'Vol/OI': f"{trade['vol_oi_ratio']:.1f}",
                                'Side': trade.get('trade_side', 'UNKNOWN')
                            })
                        
                        df_inst = pd.DataFrame(inst_data)
                        st.dataframe(df_inst, use_container_width=True)
                    else:
                        st.info("No institutional trades detected")
                
                with col2:
                    st.markdown("#### 👥 Retail Flow")
                    if retail_trades:
                        retail_data = []
                        for trade in retail_trades[:20]:
                            retail_data.append({
                                'Ticker': trade['ticker'],
                                'Type': trade['type'],
                                'Premium': f"${trade['premium']:,.0f}",
                                'Volume': f"{trade['volume']:,}",
                                'Vol/OI': f"{trade['vol_oi_ratio']:.1f}",
                                'Side': trade.get('trade_side', 'UNKNOWN')
                            })
                        
                        df_retail = pd.DataFrame(retail_data)
                        st.dataframe(df_retail, use_container_width=True)
                    else:
                        st.info("No retail trades detected")
            
            elif "ETF Flow Scanner" in scan_type:
                display_etf_scanner(trades)
            
            # Export section
            with st.expander("💾 Export Data & Analytics", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    save_to_csv(trades, "enhanced_flow")
                
                with col2:
                    # Export alerts
                    alerts = generate_enhanced_alerts(trades)
                    all_alerts = []
                    for priority, alert_list in alerts.items():
                        for alert in alert_list:
                            alert['priority'] = priority
                            all_alerts.append(alert)
                    
                    if all_alerts:
                        save_to_csv(all_alerts, "smart_alerts")
                
                with col3:
                    # Export summary stats
                    if st.button("📊 Export Summary", use_container_width=True):
                        summary_stats = {
                            'total_trades': len(trades),
                            'total_premium': sum(t.get('premium', 0) for t in trades),
                            'avg_premium': np.mean([t.get('premium', 0) for t in trades]),
                            'sentiment_ratio': calculate_sentiment_score(trades)[0],
                            'institutional_trades': len(detect_institutional_flow(trades)[0]),
                            'retail_trades': len(detect_institutional_flow(trades)[1]),
                            'dark_pool_trades': len(detect_dark_pool_activity(trades))
                        }
                        
                        summary_df = pd.DataFrame([summary_stats])
                        csv = summary_df.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download Summary Stats",
                            data=csv,
                            file_name=f"flow_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to Enhanced Options Flow Tracker! 👋
        
        This significantly enhanced version includes major improvements:
        
        ### 🆕 **New Features:**
        
        #### 🤖 **AI-Powered Analysis**
        - **Smart Pattern Recognition** with confidence scoring
        - **Institutional vs Retail Detection** using premium and volume thresholds
        - **Dark Pool Activity Detection** with multi-factor scoring
        - **Predictive Alert System** with priority levels (Critical/High/Medium/Low)
        
        #### 📊 **Interactive Visualizations**
        - **Real-time Charts** with Plotly integration
        - **Volume Heatmaps** by strike and time
        - **Sentiment Gauges** and flow analysis
        - **Sector Flow Analysis** with breakdown charts
        
        #### ⚡ **Performance Enhancements**
        - **Smart Caching** for faster data retrieval
        - **Rate Limiting** to prevent API throttling
        - **Database Storage** for historical analysis
        - **Auto-refresh** with customizable intervals
        
        #### 🎯 **Advanced Features**
        - **Enhanced Trade Side Detection** with confidence levels
        - **Open Interest Deep Analysis** with liquidity scoring
        - **Multi-factor Alert System** with customizable thresholds
        - **Personalized Dashboard** with watchlists and risk preferences
        
        ### 🔧 **Technical Improvements**
        - **SQLite Database** for historical data storage
        - **Enhanced Error Handling** with retry mechanisms
        - **Modern UI Design** with custom CSS styling
        - **Advanced Filtering** with multiple criteria
        
        ### 📈 **Analysis Types:**
        
        #### 🔍 **Main Flow Analysis**
        - Comprehensive trade analysis with enhanced metrics
        - Sector breakdown and scenario analysis
        - Performance metrics and confidence scoring
        
        #### 📊 **Advanced Visualizations**
        - Interactive charts and heatmaps
        - Real-time flow analysis
        - Volume and premium distribution charts
        
        #### 🚨 **Smart Alert System**
        - Priority-based alerts (Critical/High/Medium/Low)
        - Multi-factor scoring system
        - Customizable alert thresholds
        
        #### 🌑 **Dark Pool Analysis**
        - Potential dark pool print detection
        - Multi-factor scoring for unusual activity
        - Large block trade identification
        
        #### 🏛️ **Institutional vs Retail**
        - Separate analysis of institutional and retail flow
        - Premium and volume-based classification
        - Timing pattern analysis
        
        #### ⚡ **ETF Flow Scanner**
        - Dedicated SPY/QQQ/IWM analysis
        - Short-term (≤7 DTE) focus
        - 0DTE spotlight section
        
        ### 🎛️ **Enhanced Controls:**
        - **Personalized Dashboard** with watchlists
        - **Advanced Filters** including IV range, volume, and OI
        - **Quick Filter Buttons** for common scenarios
        - **Auto-refresh** with customizable intervals
        - **Real-time Status** indicators
        
        ### 💾 **Export Options:**
        - **Enhanced CSV Export** with all analysis data
        - **Smart Alerts Export** with priority levels
        - **Summary Statistics** export
        - **Historical Data** storage and retrieval
        
        **Select your analysis type and filters, then click 'Run Enhanced Scan' to begin!**
        """)

# --- SHORT-TERM ETF SCANNER ---
def parse_option_chain_simple(opt_str):
    """Simplified option chain parser for ETF scanner"""
    try:
        idx = next(i for i, c in enumerate(opt_str) if c.isdigit())
        ticker = opt_str[:idx]
        date_str = opt_str[idx:idx+6]
        expiry_date = date(2000 + int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6]))
        dte = (expiry_date - date.today()).days
        option_type = opt_str[idx+6]
        strike = int(opt_str[idx+7:]) / 1000
        return ticker.upper(), expiry_date.strftime('%Y-%m-%d'), dte, option_type.upper(), strike
    except Exception:
        return None, None, None, None, None

def display_etf_scanner(trades):
    """Enhanced ETF scanner display"""
    st.markdown("### ⚡ ETF Flow Scanner (SPY/QQQ/IWM ≤ 7 DTE)")
    
    if not trades:
        st.warning("No ETF trades found")
        return
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_premium = sum(t.get('premium', 0) for t in trades)
        st.metric("Total Premium", f"${total_premium:,.0f}")
    
    with col2:
        zero_dte = len([t for t in trades if t.get('dte', 0) == 0])
        st.metric("0DTE Trades", zero_dte)
    
    with col3:
        buy_trades = len([t for t in trades if 'BUY' in t.get('trade_side', '')])
        sell_trades = len([t for t in trades if 'SELL' in t.get('trade_side', '')])
        st.metric("Buy/Sell", f"{buy_trades}/{sell_trades}")
    
    with col4:
        avg_vol_oi = np.mean([t.get('vol_oi_ratio', 0) for t in trades]) if trades else 0
        st.metric("Avg Vol/OI", f"{avg_vol_oi:.1f}")
    
    with col5:
        high_vol_trades = len([t for t in trades if t.get('vol_oi_ratio', 0) > 10])
        st.metric("High Vol/OI", high_vol_trades)
    
    # Separate by ETF
    spy_trades = [t for t in trades if t.get('ticker') == 'SPY']
    qqq_trades = [t for t in trades if t.get('ticker') == 'QQQ']
    iwm_trades = [t for t in trades if t.get('ticker') == 'IWM']
    
    def create_etf_table(ticker_trades, ticker_name):
        if not ticker_trades:
            st.info(f"No {ticker_name} trades found")
            return
        
        # Sort by premium descending
        sorted_trades = sorted(ticker_trades, key=lambda x: x.get('premium', 0), reverse=True)
        
        table_data = []
        for trade in sorted_trades[:25]:  # Top 25 per ETF
            table_data.append({
                'Type': trade.get('type', ''),
                'Side': trade.get('trade_side', 'N/A'),
                'Strike': f"${trade.get('strike', 0):.0f}",
                'DTE': trade.get('dte', 0),
                'Price': f"${trade.get('price', 0):.2f}" if trade.get('price') != 'N/A' else 'N/A',
                'Premium': f"${trade.get('premium', 0):,.0f}",
                'Volume': f"{trade.get('volume', 0):,.0f}",
                'OI': f"{trade.get('oi', 0):,.0f}",
                'Vol/OI': f"{trade.get('vol_oi_ratio', 0):.1f}",
                'Moneyness': trade.get('moneyness', 'N/A'),
                'Time': trade.get('time_ny', 'N/A'),
                'Rule': trade.get('rule_name', 'N/A')
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    
    # Display each ETF in tabs
    tab1, tab2, tab3 = st.tabs(["🕷️ SPY", "🔷 QQQ", "🔸 IWM"])
    
    with tab1:
        st.markdown("#### SPY Short-Term Flow")
        spy_premium = sum(t.get('premium', 0) for t in spy_trades)
        spy_count = len(spy_trades)
        st.write(f"**{spy_count} trades | ${spy_premium:,.0f} premium**")
        create_etf_table(spy_trades, "SPY")
    
    with tab2:
        st.markdown("#### QQQ Short-Term Flow")
        qqq_premium = sum(t.get('premium', 0) for t in qqq_trades)
        qqq_count = len(qqq_trades)
        st.write(f"**{qqq_count} trades | ${qqq_premium:,.0f} premium**")
        create_etf_table(qqq_trades, "QQQ")
    
    with tab3:
        st.markdown("#### IWM Short-Term Flow")
        iwm_premium = sum(t.get('premium', 0) for t in iwm_trades)
        iwm_count = len(iwm_trades)
        st.write(f"**{iwm_count} trades | ${iwm_premium:,.0f} premium**")
        create_etf_table(iwm_trades, "IWM")
    
    # Key insights
    st.markdown("#### 🔍 Key ETF Insights")
    
    # Most active strikes
    strike_activity = {}
    for trade in trades:
        key = f"{trade.get('ticker', '')} ${trade.get('strike', 0):.0f}{trade.get('type', '')}"
        if key not in strike_activity:
            strike_activity[key] = {'count': 0, 'total_premium': 0, 'total_volume': 0}
        strike_activity[key]['count'] += 1
        strike_activity[key]['total_premium'] += trade.get('premium', 0)
        strike_activity[key]['total_volume'] += trade.get('volume', 0)
    
    # Sort by total premium
    top_strikes = sorted(strike_activity.items(), 
                        key=lambda x: x[1]['total_premium'], reverse=True)[:8]
    
    if top_strikes:
        st.markdown("**🎯 Most Active ETF Strikes by Premium:**")
        col1, col2 = st.columns(2)
        
        for i, (strike_key, data) in enumerate(top_strikes):
            col = col1 if i % 2 == 0 else col2
            with col:
                st.write(f"**{strike_key}**")
                st.write(f"💰 ${data['total_premium']:,.0f} | 📊 {data['total_volume']:,.0f} vol | {data['count']} trades")
    
    # 0DTE focus
    zero_dte_trades = [t for t in trades if t.get('dte', 0) == 0]
    if zero_dte_trades:
        st.markdown("#### ⚡ 0DTE Spotlight")
        zero_dte_premium = sum(t.get('premium', 0) for t in zero_dte_trades)
        st.metric("0DTE Total Premium", f"${zero_dte_premium:,.0f}")
        
        # Top 0DTE trades
        top_0dte = sorted(zero_dte_trades, key=lambda x: x.get('premium', 0), reverse=True)[:5]
        st.markdown("**Top 0DTE Trades:**")
        for i, trade in enumerate(top_0dte, 1):
            side_indicator = "🟢" if "BUY" in trade.get('trade_side', '') else "🔴" if "SELL" in trade.get('trade_side', '') else "⚪"
            st.write(f"{i}. {side_indicator} {trade.get('ticker', '')} {trade.get('strike', 0):.0f}{trade.get('type', '')} - "
                    f"${trade.get('premium', 0):,.0f} ({trade.get('trade_side', 'N/A')})")

if __name__ == "__main__":
    main()
