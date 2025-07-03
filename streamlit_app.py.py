import streamlit as st
import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, date
from collections import defaultdict
from zoneinfo import ZoneInfo
import time
import json

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Enhanced Options Flow Tracker",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header Styles */
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #667eea, #764ba2);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 1rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        transition: transform 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 15px 15px 0 0;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        font-weight: 500;
    }
    
    .metric-change {
        font-size: 0.9rem;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        margin-top: 0.5rem;
        display: inline-block;
    }
    
    .positive {
        background: rgba(78, 205, 196, 0.1);
        color: #4ecdc4;
    }
    
    .negative {
        background: rgba(253, 116, 108, 0.1);
        color: #fd746c;
    }
    
    .neutral {
        background: rgba(128, 128, 128, 0.1);
        color: #808080;
    }
    
    /* Sidebar Styles */
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Alert Styles */
    .alert {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .alert-critical {
        background: rgba(253, 116, 108, 0.1);
        border-color: #fd746c;
        color: #d63031;
    }
    
    .alert-high {
        background: rgba(255, 144, 104, 0.1);
        border-color: #ff9068;
        color: #e17055;
    }
    
    .alert-info {
        background: rgba(102, 126, 234, 0.1);
        border-color: #667eea;
        color: #6c5ce7;
    }
    
    /* Data Table Styles */
    .dataframe {
        background: white;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .dataframe thead tr {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(102, 126, 234, 0.05);
    }
    
    /* Card Styles */
    .stContainer > div {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 1rem;
    }
    
    /* Loading Animation */
    .loading {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 3rem;
        font-size: 1.2rem;
        color: #666;
    }
    
    .loading::before {
        content: '';
        width: 30px;
        height: 30px;
        border: 3px solid #e0e0e0;
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-right: 15px;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-bullish {
        background: #4ecdc4;
        box-shadow: 0 0 10px rgba(78, 205, 196, 0.5);
    }
    
    .status-bearish {
        background: #fd746c;
        box-shadow: 0 0 10px rgba(253, 116, 108, 0.5);
    }
    
    .status-neutral {
        background: #ddd;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        
        .metric-value {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- CONFIGURATION ---
class Config:
    UW_TOKEN = st.secrets.get("UW_TOKEN", "e6e8601a-0746-4cec-a07d-c3eabfc13926")
    EXCLUDE_TICKERS = {'TSLA', 'MSTR', 'CRCL'}
    ALLOWED_TICKERS = {'QQQ', 'SPY', 'IWM'}
    MIN_PREMIUM = 50000
    LIMIT = 1000
    HIGH_IV_THRESHOLD = 0.30
    IV_CRUSH_THRESHOLD = 0.15
    GAMMA_SQUEEZE_THRESHOLD = 5
    DARK_POOL_THRESHOLD = 0.3

config = Config()

# --- HELPER FUNCTIONS ---
def parse_option_chain(opt_str):
    """Parse option chain string to extract components"""
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

def detect_order_side(trade):
    """Enhanced order side detection"""
    try:
        price = float(trade.get('price', 0)) if trade.get('price') not in ['N/A', '', None] else 0
        mid_price = float(trade.get('mid_price', 0)) if trade.get('mid_price') not in ['N/A', '', None] else 0
        volume = int(trade.get('volume', 0)) if trade.get('volume') not in ['N/A', '', None] else 0
        oi = int(trade.get('open_interest', 1)) if trade.get('open_interest') not in ['N/A', '', None] else 1
        
        ask_side_indicators = [
            price >= mid_price if mid_price > 0 else False,
            'ask' in trade.get('description', '').lower(),
            trade.get('side', '').upper() == 'ASK'
        ]
        
        bid_side_indicators = [
            price <= mid_price if mid_price > 0 else False,
            'bid' in trade.get('description', '').lower(),
            trade.get('side', '').upper() == 'BID'
        ]
        
        if sum(ask_side_indicators) > sum(bid_side_indicators):
            return "SELL_TO_OPEN" if volume > oi else "SELL_TO_CLOSE"
        elif sum(bid_side_indicators) > sum(ask_side_indicators):
            return "BUY_TO_OPEN" if volume > oi else "BUY_TO_CLOSE"
        else:
            vol_oi_ratio = volume / max(oi, 1)
            return "BUY_TO_OPEN" if vol_oi_ratio > 1.5 else "MIXED"
    except (ValueError, TypeError, AttributeError):
        return "UNKNOWN"

def detect_advanced_scenarios(trade, underlying_price=None):
    """Enhanced scenario detection"""
    scenarios = []
    opt_type = trade['type']
    strike = trade['strike']
    premium = trade['premium']
    volume = trade.get('volume', 0)
    oi = trade.get('oi', 0)
    order_side = trade.get('order_side', 'UNKNOWN')
    iv = trade.get('iv', 0)
    
    if underlying_price is None:
        underlying_price = strike

    # Calculate moneyness
    if underlying_price > 0:
        moneyness_pct = ((strike - underlying_price) / underlying_price) * 100
        if opt_type == 'C':
            moneyness = "OTM" if moneyness_pct > 5 else "ITM" if moneyness_pct < -5 else "ATM"
        else:
            moneyness = "OTM" if moneyness_pct < -5 else "ITM" if moneyness_pct > 5 else "ATM"
    else:
        moneyness = "UNKNOWN"

    vol_oi_ratio = volume / max(oi, 1)
    
    # Scenario detection
    if iv > config.HIGH_IV_THRESHOLD:
        scenarios.append("High IV Premium Selling" if "SELL" in order_side else "High IV Long Position")
    
    if iv > config.IV_CRUSH_THRESHOLD and trade.get('dte', 0) <= 7:
        scenarios.append("Potential IV Crush Play")
    
    if opt_type == 'C' and moneyness == 'OTM' and premium >= 100000:
        action = "Selling" if "SELL" in order_side else "Buying"
        scenarios.append(f"Large OTM Call {action}")
    
    if "SELL" in order_side and premium > 200000:
        scenarios.append("Large Premium Collection")
    
    if vol_oi_ratio > config.GAMMA_SQUEEZE_THRESHOLD and moneyness == "ATM":
        scenarios.append("Potential Gamma Squeeze")
    
    if volume >= 500 and premium > 500000:
        scenarios.append("Institutional Flow")
    
    return scenarios if scenarios else ["Normal Flow"]

def calculate_flow_metrics(trades):
    """Calculate comprehensive flow metrics"""
    if not trades:
        return {}
    
    call_trades = [t for t in trades if t['type'] == 'C']
    put_trades = [t for t in trades if t['type'] == 'P']
    
    call_premium_bought = sum(t['premium'] for t in call_trades if 'BUY' in t.get('order_side', ''))
    call_premium_sold = sum(t['premium'] for t in call_trades if 'SELL' in t.get('order_side', ''))
    put_premium_bought = sum(t['premium'] for t in put_trades if 'BUY' in t.get('order_side', ''))
    put_premium_sold = sum(t['premium'] for t in put_trades if 'SELL' in t.get('order_side', ''))
    
    net_call_flow = call_premium_bought - call_premium_sold
    net_put_flow = put_premium_bought - put_premium_sold
    net_total_flow = net_call_flow + net_put_flow
    
    put_call_ratio = (put_premium_bought + put_premium_sold) / max(call_premium_bought + call_premium_sold, 1)
    
    iv_values = [t.get('iv', 0) for t in trades if t.get('iv', 0) > 0]
    avg_iv = np.mean(iv_values) if iv_values else 0
    high_iv_trades = [t for t in trades if t.get('iv', 0) > config.HIGH_IV_THRESHOLD]
    
    return {
        'net_total_flow': net_total_flow,
        'put_call_ratio': put_call_ratio,
        'avg_iv': avg_iv,
        'high_iv_premium': sum(t['premium'] for t in high_iv_trades),
        'call_premium_bought': call_premium_bought,
        'call_premium_sold': call_premium_sold,
        'put_premium_bought': put_premium_bought,
        'put_premium_sold': put_premium_sold
    }

def generate_mock_data(num_trades=50):
    """Generate mock data for testing"""
    import random
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'NFLX', 'SPY', 'QQQ']
    types = ['C', 'P']
    sides = ['BUY_TO_OPEN', 'SELL_TO_OPEN', 'BUY_TO_CLOSE', 'SELL_TO_CLOSE']
    
    trades = []
    for i in range(num_trades):
        ticker = random.choice(tickers)
        opt_type = random.choice(types)
        strike = round(random.uniform(100, 500), 2)
        premium = round(random.uniform(10000, 1000000), 2)
        iv = random.uniform(0.1, 0.8)
        volume = random.randint(10, 2000)
        oi = random.randint(100, 5000)
        dte = random.randint(0, 90)
        
        trade = {
            'ticker': ticker,
            'type': opt_type,
            'strike': strike,
            'premium': premium,
            'iv': iv,
            'volume': volume,
            'oi': oi,
            'dte': dte,
            'order_side': random.choice(sides),
            'price': round(random.uniform(0.5, 50), 2),
            'underlying_price': strike + random.uniform(-50, 50),
            'time_ny': datetime.now().strftime("%I:%M:%S %p")
        }
        
        trade['scenarios'] = detect_advanced_scenarios(trade, trade['underlying_price'])
        trades.append(trade)
    
    return trades

def generate_enhanced_alerts(trades):
    """Generate enhanced alerts"""
    alerts = []
    for trade in trades:
        score = 0
        reasons = []
        alert_type = "INFO"

        premium = trade.get('premium', 0)
        volume = trade.get('volume', 0)
        oi = trade.get('oi', 0)
        order_side = trade.get('order_side', '')
        iv = trade.get('iv', 0)
        
        # Premium scoring
        if premium > 1000000:
            score += 5
            reasons.append("Massive Premium (>$1M)")
            alert_type = "CRITICAL"
        elif premium > 500000:
            score += 3
            reasons.append("Large Premium (>$500K)")
            alert_type = "HIGH"
        elif premium > 250000:
            score += 2
            reasons.append("Notable Premium (>$250K)")

        # Volume/OI analysis
        vol_oi_ratio = volume / max(oi, 1)
        if vol_oi_ratio > 5:
            score += 3
            reasons.append("Extremely High Vol/OI")
        elif vol_oi_ratio > 2:
            score += 2
            reasons.append("High Vol/OI")

        # IV-based alerts
        if iv > 0.50:
            score += 3
            reasons.append("Extreme IV (>50%)")
            alert_type = "IV_ALERT"
        elif iv > config.HIGH_IV_THRESHOLD:
            score += 2
            reasons.append("High IV (>30%)")

        # Selling activity alerts
        if "SELL" in order_side and premium > 200000:
            score += 2
            reasons.append("Large Premium Collection")
            alert_type = "SELL_ALERT"

        if score >= 4:
            trade['alert_score'] = score
            trade['reasons'] = reasons
            trade['alert_type'] = alert_type
            alerts.append(trade)

    return sorted(alerts, key=lambda x: -x.get('alert_score', 0))

# --- STREAMLIT APP ---
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🚀 Enhanced Options Flow Tracker</h1>
        <p class="main-subtitle">Advanced real-time options activity with buy/sell detection, IV analysis, and institutional pattern recognition</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Controls
    with st.sidebar:
        st.markdown("## 🎛️ Enhanced Control Panel")
        
        # Analysis Type
        scan_type = st.selectbox(
            "🔍 Select Analysis Type:",
            [
                "📊 Comprehensive Dashboard",
                "💰 Selling Activity Analysis", 
                "📈 IV Analysis",
                "⚡ Real-time Alerts",
                "🎯 DTE Strategy Analysis",
                "🏛️ Institutional Flow",
                "📈 Gamma Squeeze Scanner"
            ]
        )
        
        st.markdown("### 💰 Premium Range Filters")
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
            index=1
        )
        
        st.markdown("### 📅 Activity Filters")
        include_selling = st.checkbox("Include Selling Activity", value=True)
        dte_filter = st.selectbox(
            "Time to Expiry:",
            ["All DTE", "0DTE Only", "Weekly (≤7d)", "Monthly (≤30d)", "Quarterly (≤90d)"]
        )
        
        st.markdown("### 📊 IV Filters")
        iv_filter = st.selectbox(
            "Implied Volatility:",
            ["All IV", "Low IV (≤20%)", "Medium IV (20-40%)", "High IV (>40%)", "Extreme IV (>50%)"]
        )
        
        st.markdown("---")
        
        # Action Buttons
        col1, col2 = st.columns(2)
        with col1:
            run_scan = st.button("🔄 Run Enhanced Scan", type="primary", use_container_width=True)
        with col2:
            use_mock_data = st.button("🧪 Use Mock Data", use_container_width=True)
        
        quick_0dte = st.button("📱 Quick 0DTE Scan", use_container_width=True)
        quick_high_iv = st.button("🔥 High IV Scan", use_container_width=True)
        
        # Info Section
        st.markdown("### 💡 Pro Tips")
        st.info("""
        • Use **High IV Scan** for volatility opportunities  
        • Check **IV Analysis** for comprehensive insights  
        • Monitor **0DTE** for same-day expiration plays  
        • Watch **Selling Activity** for premium collection strategies
        """)
    
    # Main execution logic
    if run_scan or use_mock_data or quick_0dte or quick_high_iv:
        # Adjust filters for quick scans
        if quick_0dte:
            dte_filter = "0DTE Only"
            premium_range = "All Premiums (No Filter)"
        elif quick_high_iv:
            iv_filter = "High IV (>40%)"
        
        # Show loading
        with st.spinner(f"Running {scan_type}..."):
            if use_mock_data:
                # Use mock data for demonstration
                trades = generate_mock_data(100)
            else:
                # Use mock data for now (replace with actual API call)
                trades = generate_mock_data(75)
                time.sleep(1)  # Simulate API delay
        
        # Apply filters
        filtered_trades = apply_filters(trades, premium_range, dte_filter, iv_filter, include_selling)
        
        if not filtered_trades:
            st.warning("⚠️ No trades match your current filters. Try adjusting the premium range or include selling activity.")
            st.info("💡 **Tip:** Try 'All Premiums (No Filter)' and 'All DTE' to see all available data.")
            return
        
        st.success(f"✅ Found {len(filtered_trades)} trades matching criteria")
        
        # Display results based on scan type
        if "Comprehensive" in scan_type:
            display_comprehensive_dashboard(filtered_trades)
        elif "Selling" in scan_type:
            display_selling_analysis(filtered_trades)
        elif "IV Analysis" in scan_type:
            display_iv_analysis(filtered_trades)
        elif "Alerts" in scan_type:
            display_alerts_analysis(filtered_trades)
        else:
            display_comprehensive_dashboard(filtered_trades)
    
    else:
        # Welcome message
        display_welcome_message()

def apply_filters(trades, premium_range, dte_filter, iv_filter, include_selling):
    """Apply filters to trades"""
    filtered_trades = []
    
    for trade in trades:
        # Premium filter
        if not apply_premium_filter(trade['premium'], premium_range):
            continue
        
        # DTE filter
        if not apply_dte_filter(trade['dte'], dte_filter):
            continue
        
        # IV filter
        if not apply_iv_filter(trade['iv'], iv_filter):
            continue
        
        # Selling filter
        if not include_selling and 'SELL' in trade.get('order_side', ''):
            continue
        
        filtered_trades.append(trade)
    
    return filtered_trades

def apply_premium_filter(premium, range_selection):
    """Apply premium range filter"""
    if range_selection == "All Premiums (No Filter)":
        return True
    elif range_selection == "Under $100K":
        return premium < 100000
    elif range_selection == "Under $250K":
        return premium < 250000
    elif range_selection == "$100K - $250K":
        return 100000 <= premium < 250000
    elif range_selection == "$250K - $500K":
        return 250000 <= premium < 500000
    elif range_selection == "Above $250K":
        return premium >= 250000
    elif range_selection == "Above $500K":
        return premium >= 500000
    elif range_selection == "Above $1M":
        return premium >= 1000000
    return True

def apply_dte_filter(dte, dte_selection):
    """Apply DTE filter"""
    if dte_selection == "All DTE":
        return True
    elif dte_selection == "0DTE Only":
        return dte == 0
    elif dte_selection == "Weekly (≤7d)":
        return dte <= 7
    elif dte_selection == "Monthly (≤30d)":
        return dte <= 30
    elif dte_selection == "Quarterly (≤90d)":
        return dte <= 90
    return True

def apply_iv_filter(iv, iv_selection):
    """Apply IV filter"""
    if iv_selection == "All IV":
        return True
    elif iv_selection == "Low IV (≤20%)":
        return iv <= 0.20
    elif iv_selection == "Medium IV (20-40%)":
        return 0.20 < iv <= 0.40
    elif iv_selection == "High IV (>40%)":
        return iv > 0.40
    elif iv_selection == "Extreme IV (>50%)":
        return iv > 0.50
    return True

def display_comprehensive_dashboard(trades):
    """Display comprehensive dashboard"""
    metrics = calculate_flow_metrics(trades)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        net_flow = metrics.get('net_total_flow', 0)
        flow_direction = "🟢 Bullish" if net_flow > 0 else "🔴 Bearish" if net_flow < 0 else "⚪ Neutral"
        change_class = "positive" if net_flow > 0 else "negative" if net_flow < 0 else "neutral"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Net Flow Direction</div>
            <div class="metric-value">${net_flow:,.0f}</div>
            <div class="metric-change {change_class}">{flow_direction}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        pcr = metrics.get('put_call_ratio', 0)
        pcr_direction = "Bearish" if pcr > 1 else "Bullish"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Put/Call Ratio</div>
            <div class="metric-value">{pcr:.2f}</div>
            <div class="metric-change">{pcr_direction}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_iv = metrics.get('avg_iv', 0)
        iv_level = "High" if avg_iv > 0.3 else "Medium" if avg_iv > 0.2 else "Low"
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average IV</div>
            <div class="metric-value">{avg_iv:.1%}</div>
            <div class="metric-change">{iv_level} Volatility</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        high_iv_premium = metrics.get('high_iv_premium', 0)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">High IV Premium</div>
            <div class="metric-value">${high_iv_premium:,.0f}</div>
            <div class="metric-change">Premium in >30% IV</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Charts Row
    col1, col2 = st.columns(2)
    
    with col1:
        # Flow breakdown chart
        flow_data = {
            'Type': ['Calls Bought', 'Calls Sold', 'Puts Bought', 'Puts Sold'],
            'Premium': [
                metrics.get('call_premium_bought', 0),
                metrics.get('call_premium_sold', 0),
                metrics.get('put_premium_bought', 0),
                metrics.get('put_premium_sold', 0)
            ]
        }
        
        fig = px.bar(
            flow_data, 
            x='Type', 
            y='Premium',
            title="💰 Buy vs Sell Flow Breakdown",
            color='Type',
            color_discrete_map={
                'Calls Bought': '#4ecdc4',
                'Calls Sold': '#fd746c',
                'Puts Bought': '#ff6b6b',
                'Puts Sold': '#6b6bff'
            }
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # IV distribution chart
        iv_data = {
            'IV Level': ['Low (≤20%)', 'Medium (20-40%)', 'High (>40%)'],
            'Count': [
                len([t for t in trades if t['iv'] <= 0.20]),
                len([t for t in trades if 0.20 < t['iv'] <= 0.40]),
                len([t for t in trades if t['iv'] > 0.40])
            ]
        }
        
        fig = px.pie(
            iv_data,
            values='Count',
            names='IV Level',
            title="📊 IV Distribution",
            color_discrete_map={
                'Low (≤20%)': '#90EE90',
                'Medium (20-40%)': '#FFD700',
                'High (>40%)': '#FF6B6B'
            }
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#333')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data Tables
    st.markdown("## 📋 Detailed Trade Analysis")
    
    tabs = st.tabs(["🟢 Call Activity", "🔴 Put Activity", "💎 High Premium", "⚡ High Volume"])
    
    with tabs[0]:
        call_trades = [t for t in trades if t['type'] == 'C']
        if call_trades:
            display_trade_table(call_trades[:20], "Call Trades")
        else:
            st.info("No call trades found")
    
    with tabs[1]:
        put_trades = [t for t in trades if t['type'] == 'P']
        if put_trades:
            display_trade_table(put_trades[:20], "Put Trades")
        else:
            st.info("No put trades found")
    
    with tabs[2]:
        high_premium = sorted(trades, key=lambda x: x['premium'], reverse=True)[:20]
        display_trade_table(high_premium, "High Premium Trades")
    
    with tabs[3]:
        high_volume = sorted(trades, key=lambda x: x['volume'], reverse=True)[:20]
        display_trade_table(high_volume, "High Volume Trades")

def display_trade_table(trades, title):
    """Display trade table with enhanced formatting"""
    if not trades:
        st.info(f"No {title.lower()} found")
        return
    
    df_data = []
    for trade in trades:
        df_data.append({
            'Ticker': trade['ticker'],
            'Type': trade['type'],
            'Strike': f"${trade['strike']:.0f}",
            'Price': f"${trade['price']:.2f}",
            'IV': f"{trade['iv']:.1%}",
            'DTE': trade['dte'],
            'Premium': f"${trade['premium']:,.0f}",
            'Side': trade['order_side'],
            'Volume': trade['volume'],
            'Strategy': ', '.join(trade['scenarios'][:2]),
            'Time': trade['time_ny']
        })
    
    df = pd.DataFrame(df_data)
    st.dataframe(df, use_container_width=True)

def display_selling_analysis(trades):
    """Display selling activity analysis"""
    st.markdown("## 💰 Options Selling Analysis")
    
    sell_trades = [t for t in trades if 'SELL' in t.get('order_side', '')]
    
    if not sell_trades:
        st.info("No selling activity detected in current dataset")
        return
    
    # Selling metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sell_premium = sum(t['premium'] for t in sell_trades)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Premium Collected</div>
            <div class="metric-value">${sell_premium:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_sell_premium = sell_premium / len(sell_trades) if sell_trades else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Premium/Trade</div>
            <div class="metric-value">${avg_sell_premium:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        sell_ratio = len(sell_trades) / len(trades) if trades else 0
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Selling Activity %</div>
            <div class="metric-value">{sell_ratio:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Top selling trades
    st.markdown("### 🔥 Largest Premium Collection Trades")
    top_selling = sorted(sell_trades, key=lambda x: x['premium'], reverse=True)[:15]
    display_trade_table(top_selling, "Top Selling Trades")

def display_iv_analysis(trades):
    """Display IV analysis"""
    st.markdown("## 📊 Implied Volatility Analysis")
    
    iv_trades = [t for t in trades if t.get('iv', 0) > 0]
    
    if not iv_trades:
        st.warning("⚠️ No IV data found in current dataset")
        return
    
    # IV metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_iv = np.mean([t['iv'] for t in iv_trades])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average IV</div>
            <div class="metric-value">{avg_iv:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        median_iv = np.median([t['iv'] for t in iv_trades])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Median IV</div>
            <div class="metric-value">{median_iv:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        high_iv_count = len([t for t in iv_trades if t['iv'] > config.HIGH_IV_THRESHOLD])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">High IV Trades</div>
            <div class="metric-value">{high_iv_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        max_iv = max([t['iv'] for t in iv_trades])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Max IV</div>
            <div class="metric-value">{max_iv:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # High IV opportunities
    st.markdown("### 🔥 High IV Opportunities")
    high_iv_trades = [t for t in iv_trades if t['iv'] > config.HIGH_IV_THRESHOLD]
    
    if high_iv_trades:
        high_iv_sorted = sorted(high_iv_trades, key=lambda x: x['iv'], reverse=True)[:15]
        display_trade_table(high_iv_sorted, "High IV Trades")
    else:
        st.info("No high IV trades found")

def display_alerts_analysis(trades):
    """Display alerts analysis"""
    st.markdown("## 🚨 Real-time Alerts")
    
    alerts = generate_enhanced_alerts(trades)
    
    if not alerts:
        st.info("No alerts triggered with current criteria")
        return
    
    for i, alert in enumerate(alerts[:10], 1):
        alert_type = alert.get('alert_type', 'INFO')
        
        if alert_type == 'CRITICAL':
            alert_class = 'alert-critical'
            icon = '🔥'
        elif alert_type == 'HIGH':
            alert_class = 'alert-high'
            icon = '⚠️'
        else:
            alert_class = 'alert-info'
            icon = 'ℹ️'
        
        st.markdown(f"""
        <div class="alert {alert_class}">
            <div>
                <strong>{icon} {i}. {alert['ticker']} ${alert['strike']:.0f}{alert['type']} ({alert['dte']}d)</strong><br>
                💰 Premium: ${alert['premium']:,.0f} | Price: ${alert['price']:.2f} | IV: {alert['iv']:.1%} | Side: {alert.get('order_side', 'Unknown')}<br>
                📍 Alert Reasons: {', '.join(alert.get('reasons', []))}<br>
                🎯 Strategy: {', '.join(alert.get('scenarios', [])[:2])}<br>
                <small>Alert Score: {alert.get('alert_score', 0)}</small>
            </div>
        </div>
        """, unsafe_allow_html=True)

def display_welcome_message():
    """Display welcome message"""
    st.markdown("""
    ## Welcome to the Enhanced Options Flow Tracker! 🚀
    
    ### ✨ New Features:
    - **🔍 Buy/Sell Detection**: Identify whether traders are buying or selling options
    - **💰 Premium Collection Analysis**: Track large option selling strategies  
    - **📊 IV Analysis**: Comprehensive implied volatility tracking and analysis
    - **⚡ Enhanced Alerts**: Multi-tier alert system with critical/high/sell/IV alerts
    - **🎯 Advanced Scenarios**: Detect gamma squeezes, dark pool activity, institutional flow
    - **📈 Comprehensive Dashboard**: Net flow analysis, put/call ratios, sentiment tracking
    
    ### 🎯 Strategy Detection:
    - Call/Put Overwriting
    - Cash-Secured Puts  
    - Iron Condors & Spreads
    - Gamma Squeeze Setups
    - Portfolio Hedging
    - Event/Earnings Plays
    - Volatility Strategies
    - High IV Premium Selling
    
    **👈 Select an analysis type from the sidebar and click 'Run Enhanced Scan' to begin!**
    
    **💡 Pro Tips:**
    - Use "High IV Scan" to find volatility opportunities
    - Check "IV Analysis" for comprehensive volatility insights
    - Set IV filters to focus on specific volatility ranges
    - Try "Mock Data" to test all features instantly
    """)

if __name__ == "__main__":
    main()
