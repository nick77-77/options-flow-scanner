"""
Options Flow Tracker with Embedded HTML Dashboard
"""

import streamlit as st
import httpx
from datetime import datetime, date
from collections import defaultdict
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

# --- CONFIGURATION ---
class Config:
    UW_TOKEN = st.secrets.get("UW_TOKEN", "e6e8601a-0746-4cec-a07d-c3eabfc13926")
    EXCLUDE_TICKERS = {'TSLA', 'MSTR', 'CRCL'}
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


config = Config()

# --- HELPER FUNCTIONS ---
def parse_option_chain(opt_str):
    try:
        ticker = ''.join([c for c in opt_str if c.isalpha()])[:-1]
        date_start = len(ticker)
        date_str = opt_str[date_start:date_start + 6]
        expiry_date = date(2000 + int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6]))
        dte = (expiry_date - date.today()).days
        option_type = opt_str[date_start + 6].upper()
        strike = int(opt_str[date_start + 7:]) / 1000
        return ticker, expiry_date.strftime('%Y-%m-%d'), dte, option_type, strike
    except Exception:
        return None, None, None, None, None


def calculate_moneyness(strike, current_price):
    if current_price == 'N/A' or current_price == 0:
        return "Unknown"
    try:
        price = float(current_price)
        diff_percent = ((strike - price) / price) * 100
        if abs(diff_percent) < 2:
            return "ATM"
        elif diff_percent > 0:
            return f"OTM +{diff_percent:.1f}%"
        else:
            return f"ITM {diff_percent:.1f}%"
    except:
        return "Unknown"


def detect_scenarios(trade, underlying_price=None):
    scenarios = []
    opt_type = trade['type']
    strike = trade['strike']
    premium = trade['premium']
    volume = trade.get('volume', 0)
    oi = trade.get('oi', 0)
    rule_name = trade.get('rule_name', '')
    ticker = trade['ticker']
    iv = trade.get('iv', 0)

    underlying_price = underlying_price or strike

    moneyness = "ATM"
    if opt_type == 'C' and strike > underlying_price:
        moneyness = "OTM"
    elif opt_type == 'C' and strike < underlying_price:
        moneyness = "ITM"
    elif opt_type == 'P' and strike < underlying_price:
        moneyness = "OTM"
    elif opt_type == 'P' and strike > underlying_price:
        moneyness = "ITM"

    # Scenario Detection
    if opt_type == 'C' and moneyness == 'OTM' and premium >= config.SCENARIO_OTM_CALL_MIN_PREMIUM:
        scenarios.append("Large OTM Call Buying")
    if opt_type == 'P' and moneyness == 'OTM' and premium >= config.SCENARIO_OTM_CALL_MIN_PREMIUM:
        scenarios.append("Large OTM Put Buying")
    if moneyness == 'ITM' and premium >= config.SCENARIO_ITM_CONV_MIN_PREMIUM:
        scenarios.append("ITM Conviction Trade")
    if volume > oi * config.SCENARIO_SWEEP_VOLUME_OI_RATIO:
        scenarios.append("Sweep Orders")
    if volume >= config.SCENARIO_BLOCK_TRADE_VOL:
        scenarios.append("Block Trade")
    if rule_name in ['RepeatedHits', 'RepeatedHitsAscendingFill']:
        scenarios.append("Repeated Buying at Same Strike")
    if opt_type == 'C' and moneyness == 'OTM' and premium < 0 and strike < underlying_price * 1.1:
        scenarios.append("Call Selling at High IV")
    if opt_type == 'P' and moneyness == 'OTM' and premium > 0:
        scenarios.append("Put Selling")
    if ticker == 'SPY' and opt_type == 'P' and moneyness == 'ITM':
        scenarios.append("Hedging Behavior")
    if 'earnings' in trade.get('description', '').lower():
        scenarios.append("Insider-like Activity")
    if iv > config.EXTREME_IV_THRESHOLD:
        scenarios.append("Extreme IV Play")
    elif iv > config.HIGH_IV_THRESHOLD:
        scenarios.append("High IV Premium")
    if iv > config.IV_CRUSH_THRESHOLD and trade.get('dte', 0) <= 7:
        scenarios.append("IV Crush Risk")
    if iv > config.HIGH_IV_THRESHOLD and premium > 200000:
        scenarios.append("Volatility Strategy")

    return scenarios if scenarios else ["Normal Flow"]


def apply_filters(trades, premium_range, dte_filter, iv_filter="All IV Levels"):
    filtered = trades.copy()
    if premium_range != "All Premiums (No Filter)":
        if premium_range == "Under $100K":
            filtered = [t for t in filtered if t['premium'] < 100000]
        elif premium_range == "Under $250K":
            filtered = [t for t in filtered if t['premium'] < 250000]
        elif premium_range == "$100K - $250K":
            filtered = [t for t in filtered if 100000 <= t['premium'] < 250000]
        elif premium_range == "$250K - $500K":
            filtered = [t for t in filtered if 250000 <= t['premium'] < 500000]
        elif premium_range == "Above $250K":
            filtered = [t for t in filtered if t['premium'] >= 250000]
        elif premium_range == "Above $500K":
            filtered = [t for t in filtered if t['premium'] >= 500000]
        elif premium_range == "Above $1M":
            filtered = [t for t in filtered if t['premium'] >= 1000000]

    if dte_filter != "All DTE":
        if dte_filter == "0DTE Only":
            filtered = [t for t in filtered if t['dte'] == 0]
        elif dte_filter == "Weekly (≤7d)":
            filtered = [t for t in filtered if t['dte'] <= 7]
        elif dte_filter == "Monthly (≤30d)":
            filtered = [t for t in filtered if t['dte'] <= 30]
        elif dte_filter == "Quarterly (≤90d)":
            filtered = [t for t in filtered if t['dte'] <= 90]
        elif dte_filter == "LEAPS (>90d)":
            filtered = [t for t in filtered if t['dte'] > 90]

    if iv_filter != "All IV Levels":
        if iv_filter == "High IV Only (>30%)":
            filtered = [t for t in filtered if t.get('iv', 0) > 0.30]
        elif iv_filter == "Extreme IV Only (>50%)":
            filtered = [t for t in filtered if t.get('iv', 0) > 0.50]
        elif iv_filter == "Low IV Only (≤20%)":
            filtered = [t for t in filtered if t.get('iv', 0) <= 0.20]

    return filtered


@st.cache_data(ttl=300)
def fetch_general_flow():
    headers = {
        'Accept': 'application/json, text/plain',
        'Authorization': config.UW_TOKEN
    }
    url = 'https://api.unusualwhales.com/api/option-trades/flow-alerts '
    params = {
        'issue_types[]': ['Common Stock', 'ADR'],
        'min_dte': 1,
        'min_volume_oi_ratio': 1.0,
        'rule_name[]': ['RepeatedHits', 'RepeatedHitsAscendingFill', 'RepeatedHitsDescendingFill'],
        'limit': config.LIMIT
    }

    try:
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

            utc_time_str = trade.get('created_at', "N/A")
            ny_time_str = "N/A"
            if utc_time_str != "N/A":
                try:
                    utc_time = datetime.fromisoformat(utc_time_str.replace("Z", "+00:00"))
                    ny_time = utc_time.astimezone(ZoneInfo("America/New_York"))
                    ny_time_str = ny_time.strftime("%I:%M %p")
                except Exception:
                    pass

            iv = 0
            iv_fields = ['implied_volatility', 'volatility', 'impliedVolatility', 'vol', 'IV', 'iv']
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
                'oi': trade.get('open_interest', 0),
                'time_utc': utc_time_str,
                'time_ny': ny_time_str,
                'rule_name': trade.get('rule_name', ''),
                'description': trade.get('description', ''),
                'underlying_price': trade.get('underlying_price', strike),
                'moneyness': calculate_moneyness(strike, trade.get('underlying_price', strike)),
                'vol_oi_ratio': trade.get('volume', 0) / max(trade.get('open_interest', 1), 1),
                'iv': iv,
                'iv_percentage': f"{iv:.1%}" if iv > 0 else "N/A"
            }

            ticker_data[ticker].append(trade_data)

        for ticker, trade_list in ticker_data.items():
            atm_calls = [t['strike'] for t in trade_list if t['type'] == 'C']
            avg_underlying_price = sum(atm_calls) / len(atm_calls) if atm_calls else None
            for trade in trade_list:
                underlying_price = avg_underlying_price or trade['strike']
                trade['scenarios'] = detect_scenarios(trade, underlying_price)
                result.append(trade)

        return result

    except Exception as e:
        st.error(f"Error fetching general flow: {e}")
        return []


# --- HTML Dashboard Content ---
HTML_DASHBOARD = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Options Flow Tracker Dashboard</title>
  <style>
    body {
      font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: #f9fafb;
      margin: 0;
      padding: 0 2rem 2rem;
      color: #1f2937;
      min-height: 100vh;
    }
    header {
      background: #2563eb;
      color: white;
      padding: 1.6rem 2rem;
      font-weight: 700;
      font-size: 1.8rem;
      text-align: center;
      border-radius: 0 0 12px 12px;
      margin-bottom: 2rem;
      box-shadow: 0 4px 12px rgb(37 99 235 / 0.4);
    }
    main {
      max-width: 1200px;
      margin: 0 auto;
    }
    section {
      background: white;
      padding: 1.8rem 2rem;
      border-radius: 12px;
      box-shadow: 0 4px 15px rgba(0, 0, 0, 0.07);
      margin-bottom: 2.5rem;
    }
    h2 {
      margin-bottom: 1rem;
      color: #1e40af;
      border-bottom: 3px solid #3b82f6;
      padding-bottom: 0.3rem;
      font-weight: 700;
      font-size: 1.6rem;
    }
    p {
      line-height: 1.5;
      font-size: 1rem;
      color: #374151;
      margin-bottom: 1rem;
    }
    ul {
      padding-left: 1.2rem;
      margin-bottom: 1rem;
      color: #4b5563;
    }
    ul li {
      margin-bottom: 0.5rem;
    }
    .grid-charts {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1.5rem;
      margin-top: 1rem;
    }
    .chart-box {
      background: #e0e7ff;
      border-radius: 10px;
      height: 220px;
      display: flex;
      justify-content: center;
      align-items: center;
      color: #3730a3;
      font-weight: 600;
      font-size: 1.1rem;
      user-select: none;
      box-shadow: inset 0 0 12px rgba(59, 130, 246, 0.3);
    }
    button.cta-btn {
      background-color: #2563eb;
      color: white;
      border: none;
      padding: 0.8rem 1.8rem;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      transition: background-color 0.3s ease;
      margin-top: 1rem;
      display: inline-block;
      text-align: center;
    }
    button.cta-btn:hover {
      background-color: #1d4ed8;
    }
    @media (max-width: 480px) {
      body {
        padding: 1rem;
      }
      header {
        font-size: 1.4rem;
        padding: 1rem;
      }
      section {
        padding: 1rem 1.2rem;
      }
    }
  </style>
</head>
<body>
  <header>Options Flow Tracker Dashboard</header>
  <main>
    <section>
      <h2>Overview</h2>
      <p>This Streamlit-based Options Flow Tracker app fetches, analyzes, and visualizes real-time unusual options trading activity from the Unusual Whales API.</p>
      <p>It detects significant trades and analyzes them by premium size, time-to-expiration (DTE), and implied volatility (IV).</p>
    </section>

    <section>
      <h2>Scan Types</h2>
      <p>Our app supports multiple scan types to cover all angles of the options market:</p>
      <ul>
        <li>General flow analysis</li>
        <li>DTE-segregated views</li>
        <li>Implied Volatility (IV) analytics</li>
        <li>Smart alert generation based on customized criteria</li>
      </ul>
      <button class="cta-btn" onclick="alert('Start scan clicked')">Start Scan</button>
    </section>

    <section>
      <h2>Analytics & Trade Scenarios</h2>
      <p>The app parses option chain data to calculate moneyness and detect trade scenarios like large OTM buying, sweeps, block trades, hedging, and volatility plays.</p>
      <div class="grid-charts">
        <div class="chart-box">Call/Put Ratio Pie Chart</div>
        <div class="chart-box">Scenario Distribution Bar Graph</div>
        <div class="chart-box">Implied Volatility Histogram</div>
      </div>
      <button class="cta-btn" onclick="alert('View analytics clicked')">View Analytics</button>
    </section>

    <section>
      <h2>Filters & Customization</h2>
      <p>Focus your analysis with powerful filters:</p>
      <ul>
        <li>Premium Range</li>
        <li>Time-To-Expiration (DTE)</li>
        <li>Implied Volatility (IV) Levels</li>
      </ul>
      <p>These filters help you zero in on high-impact trades.</p>
      <button class="cta-btn" onclick="alert('Apply filters clicked')">Apply Filters</button>
    </section>

    <section>
      <h2>Export Data</h2>
      <p>Export your filtered results to CSV for further offline analysis and record-keeping.</p>
      <button class="cta-btn" onclick="alert('Export to CSV clicked')">Export to CSV</button>
    </section>
  </main>
</body>
</html>
"""


# --- MAIN APP ---
def main():
    st.set_page_config(page_title="Options Flow Tracker", page_icon="📊", layout="wide")
    st.title("📊 Comprehensive Options Flow Tracker")
    st.markdown("### Real-time unusual options activity analysis with enhanced IV pattern recognition")

    view = st.selectbox("Select View", ["Dashboard View", "Advanced View"])

    if view == "Dashboard View":
        components.html(HTML_DASHBOARD, height=1200, scrolling=True)
    else:
        with st.sidebar:
            st.markdown("## 🎛️ Control Panel")
            scan_type = st.selectbox(
                "Select Scan Type:",
                [
                    "🔍 General Flow Scanner",
                    "⏰ DTE Segregated Flow",
                    "📊 IV Analysis",
                    "🚨 Alert Generator"
                ]
            )
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
            iv_filter = st.selectbox(
                "IV Range Filter:",
                [
                    "All IV Levels",
                    "High IV Only (>30%)",
                    "Extreme IV Only (>50%)",
                    "Low IV Only (≤20%)"
                ] if "IV Analysis" in scan_type else ["All IV Levels"],
                disabled="IV Analysis" not in scan_type
            )

            run_scan = st.button("🔄 Run Scan", type="primary", use_container_width=True)

        if run_scan:
            with st.spinner(f"Running {scan_type}..."):
                trades = fetch_general_flow()
                filtered_trades = apply_filters(trades, premium_range, dte_filter, iv_filter)
                st.success(f"Found {len(filtered_trades)} matching trades.")
                # You can add more advanced UI here as before...
                st.dataframe(pd.DataFrame(filtered_trades).head(10))


if __name__ == "__main__":
    main()
