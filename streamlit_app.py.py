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
    HIGH_IV_THRESHOLD = 0.30  # 30% IV threshold
    EXTREME_IV_THRESHOLD = 0.50  # 50% IV threshold
    IV_CRUSH_THRESHOLD = 0.15  # 15% IV threshold for crush detection
config = Config()

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    html body {
        font-size: 16px;
    }
    .stTextInput input, .stSelectbox select, .stSlider input {
        font-size: 16px !important;
    }
    .stSelectbox label, .stTextInput label {
        font-size: 14px !important;
    }
    .reportview-container .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

dark_mode = st.checkbox("🌙 Dark Mode", key="dark_mode")
if dark_mode:
    st.markdown("""
    <style>
        body {
            background-color: #1e1e1e;
            color: white;
        }
        .stTextInput input, .stSelectbox select, .stSlider input {
            background-color: #2e2e2e;
            color: white;
        }
        .st-expander {
            background-color: #2e2e2e;
        }
    </style>
    """, unsafe_allow_html=True)

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
    if underlying_price is None:
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
    if iv > config.EXTREME_IV_THRESHOLD:
        scenarios.append("Extreme IV Play")
    elif iv > config.HIGH_IV_THRESHOLD:
        scenarios.append("High IV Premium")
    if iv > config.IV_CRUSH_THRESHOLD and trade.get('dte', 0) <= 7:
        scenarios.append("IV Crush Risk")
    return scenarios if scenarios else ["Normal Flow"]

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
    call_premium = sum(t['premium'] for t in trades if t['type'] == 'C')
    put_premium = sum(t['premium'] for t in trades if t['type'] == 'P')
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

def calculate_iv_metrics(trades):
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

# --- FETCH FUNCTION ---
def fetch_general_flow():
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
            utc_time_str = trade.get('created_at')
            ny_time_str = "N/A"
            if utc_time_str != "N/A":
                try:
                    utc_time = datetime.fromisoformat(utc_time_str.replace("Z", "+00:00"))
                    ny_time = utc_time.astimezone(ZoneInfo("America/New_York"))
                    ny_time_str = ny_time.strftime("%I:%M %p")
                except Exception:
                    ny_time_str = "N/A"
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
            if iv == 0:
                try:
                    option_price = float(trade.get('price', 0))
                    underlying_price = float(trade.get('underlying_price', strike))
                    if option_price > 0 and underlying_price > 0 and dte > 0:
                        time_factor = np.sqrt(dte / 365.0)
                        if time_factor > 0:
                            estimated_iv = (option_price / underlying_price) / time_factor
                            if 0.01 <= estimated_iv <= 3.0:
                                iv = estimated_iv
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
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
                underlying_price = avg_underlying_price if avg_underlying_price is not None else trade['strike']
                scenarios = detect_scenarios(trade, underlying_price)
                trade['scenarios'] = scenarios
                result.append(trade)
        return result
    except Exception as e:
        st.error(f"Error fetching general flow: {e}")
        return []

# --- FILTER FUNCTIONS ---
def apply_premium_filter(trades, premium_range):
    if premium_range == "All Premiums (No Filter)": return trades
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
    if dte_filter == "All DTE": return trades
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

def apply_iv_filter(trades, iv_filter):
    if iv_filter == "All IV Levels": return trades
    filtered_trades = []
    for trade in trades:
        iv = trade.get('iv', 0)
        if iv_filter == "High IV Only (>30%)" and iv > 0.30:
            filtered_trades.append(trade)
        elif iv_filter == "Extreme IV Only (>50%)" and iv > 0.50:
            filtered_trades.append(trade)
        elif iv_filter == "Low IV Only (≤20%)" and iv <= 0.20:
            filtered_trades.append(trade)
    return filtered_trades

# --- CSV EXPORT ---
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

# --- DISPLAY FUNCTIONS ---
def display_mobile_friendly_table(trades):
    for trade in trades[:10]:
        st.markdown(f"""
        <div style="border:1px solid #ccc; border-radius:8px; padding:10px; margin-bottom:10px;">
            <strong>{trade['ticker']} {trade['strike']}{trade['type']}</strong><br/>
            Exp: {trade['expiry']} | DTE: {trade['dte']}<br/>
            Premium: ${trade['premium']:,.0f}<br/>
            Scenarios: {', '.join(trade.get('scenarios', []))}
        </div>
        """, unsafe_allow_html=True)

def visualize_market_summary(trades):
    if not trades:
        return
    df = pd.DataFrame(trades)
    df['scenario'] = df['scenarios'].apply(lambda x: x[0] if x else "Normal Flow")
    fig = px.pie(df, names='type', title="Call vs Put Distribution")
    st.plotly_chart(fig, use_container_width=True)

def display_dte_segregated(trades):
    calls_lt7 = [t for t in trades if t['type'] == 'C' and t['dte'] < 7]
    calls_gte7 = [t for t in trades if t['type'] == 'C' and t['dte'] >= 7]
    puts_lt7 = [t for t in trades if t['type'] == 'P' and t['dte'] < 7]
    puts_gte7 = [t for t in trades if t['type'] == 'P' and t['dte'] >= 7]

    def create_trade_df(trades_list, limit=10):
        df_data = []
        for trade in sorted(trades_list, key=lambda x: x.get('premium', 0), reverse=True)[:limit]:
            df_data.append({
                'Ticker': trade['ticker'],
                'Strike': trade['strike'],
                'Expiry': trade['expiry'],
                'DTE': trade['dte'],
                'Price': trade['price'],
                'IV': trade['iv_percentage'],
                'Premium': f"${trade['premium']:,}",
                'Volume': trade['volume'],
                'OI': trade['oi'],
                'Time': trade['time_ny'],
                'Scenarios': ", ".join(trade['scenarios']),
            })
        return pd.DataFrame(df_data) if df_data else pd.DataFrame()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🟢 CALLS (< 7 DTE)")
        df = create_trade_df(calls_lt7)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No calls < 7 DTE")

        st.markdown("#### 🟢 CALLS (≥ 7 DTE)")
        df = create_trade_df(calls_gte7)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No calls ≥ 7 DTE")

    with col2:
        st.markdown("#### 🔴 PUTS (< 7 DTE)")
        df = create_trade_df(puts_lt7)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No puts < 7 DTE")

        st.markdown("#### 🔴 PUTS (≥ 7 DTE)")
        df = create_trade_df(puts_gte7)
        if not df.empty:
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No puts ≥ 7 DTE")

def display_iv_analysis(trades):
    st.markdown("### 📊 Implied Volatility Analysis")
    iv_trades = [t for t in trades if t.get('iv', 0) > 0]
    if not iv_trades:
        st.warning("⚠️ No IV data found in current dataset")
        st.info("The system will attempt to estimate IV from option prices when possible.")
        return
    iv_metrics = calculate_iv_metrics(trades)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average IV", f"{iv_metrics['avg_iv']:.1%}")
    with col2:
        st.metric("Median IV", f"{iv_metrics['median_iv']:.1%}")
    with col3:
        st.metric("High IV Trades", f"{iv_metrics['high_iv_count']}")
    with col4:
        iv_range = iv_metrics['max_iv'] - iv_metrics['min_iv']
        st.metric("IV Range", f"{iv_range:.1%}")

    col1, col2 = st.columns(2)
    with col1:
        iv_dist_data = pd.DataFrame({
            'IV Category': ['Low (≤20%)', 'Medium (20-40%)', 'High (>40%)', 'Extreme (>50%)'],
            'Count': [
                iv_metrics['low_iv_count'],
                iv_metrics['medium_iv_count'], 
                iv_metrics['high_iv_count'],
                iv_metrics['extreme_iv_count']
            ]
        })
        fig = px.bar(
            iv_dist_data, 
            x='IV Category', 
            y='Count',
            title="IV Distribution by Category",
            color='IV Category',
            color_discrete_map={
                'Low (≤20%)': '#90EE90',
                'Medium (20-40%)': '#FFD700', 
                'High (>40%)': '#FF6B6B',
                'Extreme (>50%)': '#8B0000'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        premium_dist_data = pd.DataFrame({
            'IV Category': ['Low (≤20%)', 'Medium (20-40%)', 'High (>40%)', 'Extreme (>50%)'],
            'Premium': [
                iv_metrics['low_iv_premium'],
                iv_metrics['medium_iv_premium'],
                iv_metrics['high_iv_premium'],
                iv_metrics['extreme_iv_premium']
            ]
        })
        fig = px.bar(
            premium_dist_data, 
            x='IV Category', 
            y='Premium',
            title="Premium Distribution by IV Level",
            color='IV Category',
            color_discrete_map={
                'Low (≤20%)': '#90EE90',
                'Medium (20-40%)': '#FFD700',
                'High (>40%)': '#FF6B6B',
                'Extreme (>50%)': '#8B0000'
            }
        )
        fig.update_layout(yaxis_tickformat='$,.0f')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🔥 High IV Opportunities")
    high_iv_trades = [t for t in trades if t.get('iv', 0) > config.HIGH_IV_THRESHOLD]
    if high_iv_trades:
        high_iv_trades.sort(key=lambda x: x.get('iv', 0), reverse=True)
        high_iv_data = []
        for trade in high_iv_trades[:15]:
            high_iv_data.append({
                'Ticker': trade['ticker'],
                'Type': trade['type'],
                'Strike': f"${trade['strike']:.0f}",
                'IV': trade['iv_percentage'],
                'DTE': trade['dte'],
                'Premium': f"${trade['premium']:,.0f}",
                'Volume': trade['volume'],
                'Strategy': ", ".join(trade.get('scenarios', [])[:2]),
                'Time': trade['time_ny']
            })
        st.dataframe(pd.DataFrame(high_iv_data), use_container_width=True)
    else:
        st.info("No high IV trades found in current dataset")

    st.markdown("#### ⚡ IV Crush Risk Analysis")
    iv_crush_candidates = [t for t in trades if t.get('iv', 0) > config.IV_CRUSH_THRESHOLD and t.get('dte', 0) <= 7]
    if iv_crush_candidates:
        iv_crush_data = []
        for trade in sorted(iv_crush_candidates, key=lambda x: x.get('iv', 0), reverse=True)[:10]:
            risk_level = "High" if trade.get('iv', 0) > 0.50 else "Medium"
            iv_crush_data.append({
                'Ticker': trade['ticker'],
                'Type': trade['type'],
                'Strike': f"${trade['strike']:.0f}",
                'IV': trade['iv_percentage'],
                'DTE': trade['dte'],
                'Premium': f"${trade['premium']:,.0f}",
                'Volume': trade['volume'],
                'Risk Level': risk_level,
                'Time': trade['time_ny']
            })
        st.dataframe(pd.DataFrame(iv_crush_data), use_container_width=True)
        st.info("💡 These positions may lose value rapidly if volatility decreases after events")
    else:
        st.info("No IV crush candidates found")

def display_alerts(trades):
    alerts = generate_alerts(trades)
    if not alerts:
        st.info("No high-priority alerts found")
        return
    st.markdown("### 🚨 High Priority Alerts")
    for i, alert in enumerate(alerts[:10], 1):
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{i}. {alert['ticker']} {alert['strike']:.0f}{alert['type']} "
                            f"{alert['expiry']} ({alert['dte']}d)**")
                st.write(f"💰 Premium: ${alert['premium']:,.0f} | Vol: {alert['volume']} | "
                         f"IV: {alert['iv_percentage']} | {alert.get('moneyness', 'N/A')}")
                st.write(f"🎯 Scenarios: {', '.join(alert.get('scenarios', []))}")
                st.write(f"📍 Reasons: {', '.join(alert.get('reasons', []))}")
            with col2:
                st.metric("Alert Score", alert.get('alert_score', 0))
            st.divider()

# --- MAIN UI ---
st.set_page_config(page_title="Options Flow Tracker", page_icon="📊", layout="wide")
st.title("📊 Comprehensive Options Flow Tracker")
st.markdown("### Real-time unusual options activity analysis with enhanced IV pattern recognition")

with st.sidebar:
    scan_type = st.selectbox(
        "Select View:",
        [
            "🔍 General Flow Scanner",
            "⏰ DTE Segregated Flow",
            "📊 IV Analysis",
            "🚨 Alert Generator"
        ],
        index=0
    )

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

    if "IV Analysis" in scan_type:
        st.markdown("### 📊 IV Filter Options")
        iv_filter = st.selectbox(
            "IV Range Filter:",
            [
                "All IV Levels",
                "High IV Only (>30%)",
                "Extreme IV Only (>50%)",
                "Low IV Only (≤20%)"
            ]
        )
        show_iv_crush = st.checkbox("Show IV Crush Candidates", value=True)
        show_volatility_plays = st.checkbox("Show Volatility Strategies", value=True)

    st.markdown("### ⚡ Quick Filters")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔥 High Premium", use_container_width=True):
            premium_range = "Above $500K"
    with col2:
        if st.button("💎 Mega Trades", use_container_width=True):
            premium_range = "Above $1M"

    run_scan = st.button("🔄 Run Scan", type="primary", use_container_width=True)

if run_scan:
    with st.spinner(f"Running {scan_type}..."):
        trades = fetch_general_flow()
        original_count = len(trades)
        trades = apply_premium_filter(trades, premium_range)
        premium_filtered_count = len(trades)
        trades = apply_dte_filter(trades, dte_filter)
        dte_filtered_count = len(trades)
        if "IV Analysis" in scan_type and 'iv_filter' in locals():
            trades = apply_iv_filter(trades, iv_filter)
            iv_filtered_count = len(trades)

        if premium_range != "All Premiums (No Filter)" or dte_filter != "All DTE":
            st.info(f"**Filter Results:** {original_count} → {len(trades)} trades after applying filters")
            if premium_range != "All Premiums (No Filter)":
                st.write(f"📊 Premium Filter ({premium_range}): {premium_filtered_count} trades")
            if dte_filter != "All DTE":
                st.write(f"📅 DTE Filter ({dte_filter}): {dte_filtered_count} trades")
            if "IV Analysis" in scan_type and 'iv_filter' in locals() and iv_filter != "All IV Levels":
                st.write(f"📊 IV Filter ({iv_filter}): {len(trades)} trades")

        if not trades:
            st.warning("⚠️ No trades match your current filters. Try adjusting the premium range or DTE filters.")
            st.info("💡 Tip: Try 'All Premiums (No Filter)' and 'All DTE' to see all available data.")
        else:
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Market Summary", 
                "📋 Raw Data", 
                "📊 Charts", 
                "💾 Export"
            ])
            with tab1:
                sentiment_ratio, sentiment_label = calculate_sentiment_score(trades)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Market Sentiment", sentiment_label, f"{sentiment_ratio:.1%} calls")
                with col2:
                    total_premium = sum(t.get('premium', 0) for t in trades)
                    st.metric("Total Premium", f"${total_premium:,.0f}")
                with col3:
                    st.metric("Total Trades", len(trades))
                with col4:
                    iv_trades = [t for t in trades if t.get('iv', 0) > 0]
                    if iv_trades:
                        avg_iv = np.mean([t['iv'] for t in iv_trades])
                        st.metric("Average IV", f"{avg_iv:.1%}")
                    else:
                        st.metric("Average IV", "N/A")
                visualize_market_summary(trades)

            with tab2:
                st.markdown("### 📋 Top 20 Trades")
                df = pd.DataFrame(trades[:20])
                st.dataframe(df[['ticker', 'type', 'strike', 'expiry', 'dte', 'premium', 'scenarios']], use_container_width=True)

            with tab3:
                st.markdown("### 📈 Chart Section")
                df = pd.DataFrame(trades)
                df['scenario'] = df['scenarios'].apply(lambda x: x[0] if x else "Normal Flow")
                fig = px.histogram(df, x='scenario', title="Scenario Distribution")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            with tab4:
                save_to_csv(trades, "options_flow_data")

else:
    st.markdown("""
    ## Welcome! 👋
    This application combines real-time options flow analysis with advanced pattern recognition and IV analysis.
    ### Available Views:
    - 🔍 **General Flow Scanner** - Overview of all unusual options activity
    - ⏰ **DTE-Segregated Flow** - Trades organized by time to expiration
    - 📊 **IV Analysis** - Comprehensive implied volatility analysis and opportunities
    - 🚨 **Smart Alert System** - High-priority alerts with IV considerations
    ### Premium Range Filters:
    - 💰 **Under $100K** - Smaller retail-focused trades
    - 💰 **$100K - $250K** - Mid-size institutional trades
    - 💰 **$250K - $500K** - Large institutional trades
    - 💰 **Above $500K** - Massive institutional flows
    - 💰 **Above $1M** - Whale trades and mega flows
    ### DTE (Days to Expiry) Filters:
    - ⚡ **0DTE** - Same-day expiration plays
    - 📅 **Weekly (≤7d)** - Short-term directional bets
    - 📅 **Monthly (≤30d)** - Standard monthly options
    - 📅 **Quarterly (≤90d)** - Longer-term strategies
    - 📅 **LEAPS (>90d)** - Long-term investment plays
    ### IV Analysis Features:
    - **High IV Identification** - Spot elevated volatility opportunities
    - **IV Crush Detection** - Identify positions at risk of volatility collapse
    - **Volatility Strategies** - Detect advanced volatility plays
    - **IV Distribution Analysis** - Understand market volatility patterns
    ### Example Trade Format:
    `GOOGL 160 2025-07-25 $2219.76 $4,363,870.0 22 1739 6 10:30 AM Sweep Orders, Block Trade`
    **💡 Pro Tips:**
    - Use **Quick Filters** for instant access to high-value trades
    - Combine **Premium** and **DTE** filters for targeted analysis
    - Check **IV Analysis** with premium filters to find volatility opportunities
    - **Above $1M** trades often indicate institutional positioning
    Select your filters and scan type from the sidebar, then click **Run Scan** to begin!
    """)
