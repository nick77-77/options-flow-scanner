"""
Options Flow Tracker - Modern Dashboard Version
"""

import streamlit as st
import httpx
from datetime import datetime, date
from collections import defaultdict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from zoneinfo import ZoneInfo

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


def generate_alerts(trades):
    alerts = []
    for trade in trades:
        score = 0
        reasons = []
        premium = trade.get('premium', 0)
        if premium > 500000:
            score += 3
            reasons.append("Massive Premium")
        elif premium > 250000:
            score += 2
            reasons.append("Large Premium")
        vol_oi_ratio = trade.get('vol_oi_ratio', 0)
        if vol_oi_ratio > 2:
            score += 2
            reasons.append("High Vol/OI")
        dte = trade.get('dte', 0)
        if dte <= 7 and premium > 200000:
            score += 2
            reasons.append("Short-term + Size")
        moneyness = trade.get('moneyness', '')
        if "ATM" in moneyness or ("OTM" in moneyness and "+5%" not in moneyness):
            score += 1
            reasons.append("Good Strike")
        iv = trade.get('iv', 0)
        if iv > config.EXTREME_IV_THRESHOLD:
            score += 3
            reasons.append("Extreme IV (>50%)")
        elif iv > config.HIGH_IV_THRESHOLD:
            score += 2
            reasons.append("High IV (>30%)")
        if iv > config.IV_CRUSH_THRESHOLD and dte <= 7:
            score += 2
            reasons.append("IV Crush Risk")
        if score >= 4:
            trade['alert_score'] = score
            trade['reasons'] = reasons
            alerts.append(trade)
    return sorted(alerts, key=lambda x: -x.get('alert_score', 0))


def calculate_iv_metrics(trades):
    """Calculate IV-related metrics"""
    iv_trades = [t for t in trades if t.get('iv', 0) > 0]
    if not iv_trades:
        return {}
    iv_values = [t['iv'] for t in iv_trades]
    low_iv = [t for t in iv_trades if t['iv'] <= 0.20]
    medium_iv = [t for t in iv_trades if 0.20 < t['iv'] <= 0.40]
    high_iv = [t for t in iv_trades if t['iv'] > 0.40]
    extreme_iv = [t for t in iv_trades if t['iv'] > 0.50]
    return {
        'total_iv_trades': len(iv_trades),
        'avg_iv': np.mean(iv_values),
        'median_iv': np.median(iv_values),
        'max_iv': np.max(iv_values),
        'min_iv': np.min(iv_values),
        'low_iv_count': len(low_iv),
        'medium_iv_count': len(medium_iv),
        'high_iv_count': len(high_iv),
        'extreme_iv_count': len(extreme_iv),
        'low_iv_premium': sum(t['premium'] for t in low_iv),
        'medium_iv_premium': sum(t['premium'] for t in medium_iv),
        'high_iv_premium': sum(t['premium'] for t in high_iv),
        'extreme_iv_premium': sum(t['premium'] for t in extreme_iv)
    }


def apply_filters(trades, premium_range, dte_filter, iv_filter="All IV Levels"):
    filtered = trades.copy()
    # Apply Premium Range Filter
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

    # Apply DTE Filter
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

    # Apply IV Filter
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
    url = 'https://api.unusualwhales.com/api/option-trades/flow-alerts'
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


def save_to_csv(trades, filename_prefix):
    if not trades:
        st.warning("No data to save")
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    csv_data = []
    for trade in trades:
        row = trade.copy()
        if isinstance(row.get('reasons'), list):
            row['reasons'] = ', '.join(row['reasons'])
        if isinstance(row.get('scenarios'), list):
            row['scenarios'] = ', '.join(row['scenarios'])
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


# --- MAIN APP ---

def main():
    st.set_page_config(page_title="Options Flow Tracker", page_icon="📊", layout="wide")
    
    # Inject custom CSS for better styling
    st.markdown("""
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
        }
        .css-1v3fvcr {
            padding-top: 2rem;
        }
        .stButton button {
            width: 100%;
            background-color: #1E90FF;
            color: white;
            border-radius: 8px;
            padding: 10px;
            font-weight: bold;
        }
        .metric-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("📊 Comprehensive Options Flow Tracker")
    st.markdown("### Real-time unusual options activity analysis with enhanced IV pattern recognition")

    tab1, tab2, tab3, tab4 = st.tabs(["🔍 General Flow", "⏰ DTE Segregated", "📊 IV Analysis", "🚨 Alert Generator"])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            premium_range = st.selectbox("Premium Range", [
                "All Premiums (No Filter)",
                "Under $100K",
                "Under $250K",
                "$100K - $250K",
                "$250K - $500K",
                "Above $250K",
                "Above $500K",
                "Above $1M"
            ], key="general_premium")
        with col2:
            dte_filter = st.selectbox("DTE Range", [
                "All DTE",
                "0DTE Only",
                "Weekly (≤7d)",
                "Monthly (≤30d)",
                "Quarterly (≤90d)",
                "LEAPS (>90d)"
            ], key="general_dte")
        run_scan = st.button("🔄 Run Scan", type="primary", key="run_general")

        if run_scan:
            with st.spinner("Fetching data..."):
                trades = fetch_general_flow()
                filtered = apply_filters(trades, premium_range, dte_filter)
                sentiment_ratio, sentiment_label = calculate_sentiment_score(filtered)
                total_premium = sum(t['premium'] for t in filtered)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Market Sentiment", sentiment_label, f"{sentiment_ratio:.1%} calls")
                with col2:
                    st.metric("Total Premium", f"${total_premium:,.0f}")
                with col3:
                    st.metric("Total Trades", len(filtered))
                with col4:
                    iv_avg = np.mean([t['iv'] for t in filtered if t.get('iv', 0) > 0]) \
                        if any(t.get('iv', 0) > 0 for t in filtered) else "N/A"
                    st.metric("Avg IV", f"{iv_avg:.1%}" if isinstance(iv_avg, float) else "N/A")
                visualize_market_summary(filtered)
                save_to_csv(filtered, "general_flow")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            premium_range = st.selectbox("Premium Range", [
                "All Premiums (No Filter)",
                "Under $100K",
                "Under $250K",
                "$100K - $250K",
                "$250K - $500K",
                "Above $250K",
                "Above $500K",
                "Above $1M"
            ], key="dte_premium")
        with col2:
            dte_filter = st.selectbox("DTE Range", [
                "All DTE",
                "0DTE Only",
                "Weekly (≤7d)",
                "Monthly (≤30d)",
                "Quarterly (≤90d)",
                "LEAPS (>90d)"
            ], key="dte_dte")
        run_scan = st.button("🔄 Run Scan", type="primary", key="run_dte")

        if run_scan:
            with st.spinner("Fetching data..."):
                trades = fetch_general_flow()
                filtered = apply_filters(trades, premium_range, dte_filter)
                display_dte_segregated(filtered)
                save_to_csv(filtered, "dte_segregated_flow")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            premium_range = st.selectbox("Premium Range", [
                "All Premiums (No Filter)",
                "Under $100K",
                "Under $250K",
                "$100K - $250K",
                "$250K - $500K",
                "Above $250K",
                "Above $500K",
                "Above $1M"
            ], key="iv_premium")
        with col2:
            iv_filter = st.selectbox("IV Filter", [
                "All IV Levels",
                "High IV Only (>30%)",
                "Extreme IV Only (>50%)",
                "Low IV Only (≤20%)"
            ])
        run_scan = st.button("🔄 Run Scan", type="primary", key="run_iv")

        if run_scan:
            with st.spinner("Fetching data..."):
                trades = fetch_general_flow()
                filtered = apply_filters(trades, premium_range, "All DTE", iv_filter)
                display_iv_analysis(filtered)
                save_to_csv(filtered, "iv_analysis")

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            premium_range = st.selectbox("Premium Range", [
                "All Premiums (No Filter)",
                "Under $100K",
                "Under $250K",
                "$100K - $250K",
                "$250K - $500K",
                "Above $250K",
                "Above $500K",
                "Above $1M"
            ], key="alert_premium")
        with col2:
            dte_filter = st.selectbox("DTE Range", [
                "All DTE",
                "0DTE Only",
                "Weekly (≤7d)",
                "Monthly (≤30d)",
                "Quarterly (≤90d)",
                "LEAPS (>90d)"
            ], key="alert_dte")
        run_scan = st.button("🔄 Run Scan", type="primary", key="run_alert")

        if run_scan:
            with st.spinner("Fetching data..."):
                trades = fetch_general_flow()
                filtered = apply_filters(trades, premium_range, dte_filter)
                display_alerts(filtered)
                save_to_csv(filtered, "alerts")


if __name__ == "__main__":
    main()
