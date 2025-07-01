import streamlit as st
import httpx
from datetime import datetime, date
from collections import defaultdict
import pandas as pd
import plotly.express as px
from zoneinfo import ZoneInfo  # Python 3.9+

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

config = Config()

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
    if opt_type == 'C' and moneyness == 'OTM' and premium < 0 and strike < underlying_price * 1.1:
        scenarios.append("Call Selling at High IV")
    if opt_type == 'P' and moneyness == 'OTM' and premium > 0:
        scenarios.append("Put Selling")
    if ticker == 'SPY' and opt_type == 'P' and moneyness == 'ITM':
        scenarios.append("Hedging Behavior")
    if 'earnings' in trade.get('description', '').lower():
        scenarios.append("Insider-like Activity")

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

        # First pass: collect raw trades
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
                'vol_oi_ratio': trade.get('volume', 0) / max(trade.get('open_interest', 1), 1)
            }

            ticker_data[ticker].append(trade_data)

        # Second pass: enrich with scenario detection
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


# --- VISUALIZATIONS ---
def visualize_market_summary(trades):
    if not trades:
        return
    df = pd.DataFrame(trades)
    df['scenario'] = df['scenarios'].apply(lambda x: x[0] if x else "Normal Flow")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(df, names='type', title="Call vs Put Distribution")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df, x='scenario', title="Scenario Distribution")
        st.plotly_chart(fig, use_container_width=True)


# --- DISPLAY FUNCTIONS ---
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
                         f"{alert.get('moneyness', 'N/A')} | 🎯 Scenarios: {', '.join(alert.get('scenarios', []))}")
                st.write(f"📍 Reasons: {', '.join(alert.get('reasons', []))}")
            with col2:
                st.metric("Alert Score", alert.get('alert_score', 0))
            st.divider()


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


# --- FILTERING ---
def apply_filters(trades):
    st.sidebar.markdown("### 🔍 Filters")
    unique_tickers = sorted(set(t['ticker'] for t in trades))
    selected_ticker = st.sidebar.selectbox("Filter by Ticker", ["All"] + unique_tickers)

    all_scenarios = set(s for t in trades for s in t.get('scenarios', ["Normal Flow"]))
    selected_scenario = st.sidebar.selectbox("Filter by Scenario", ["All"] + list(all_scenarios))

    dte_min = min(t['dte'] for t in trades) if trades else 0
    dte_max = max(t['dte'] for t in trades) if trades else 30
    dte_range = st.sidebar.slider("DTE Range", min_value=dte_min, max_value=dte_max, value=(dte_min, dte_max))

    filtered = trades
    if selected_ticker != "All":
        filtered = [t for t in filtered if t['ticker'] == selected_ticker]
    if selected_scenario != "All":
        filtered = [t for t in filtered if selected_scenario in t.get('scenarios', [])]
    filtered = [t for t in filtered if dte_range[0] <= t['dte'] <= dte_range[1]]

    return filtered


# --- STREAMLIT UI ---
st.set_page_config(page_title="Options Flow Tracker", page_icon="📊", layout="wide")
st.title("📊 Comprehensive Options Flow Tracker")
st.markdown("### Real-time unusual options activity analysis with enhanced pattern recognition")

with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    scan_type = st.selectbox(
        "Select Scan Type:",
        [
            "🔍 General Flow Scanner",
            "⏰ DTE Segregated Flow",
            "🚨 Alert Generator"
        ]
    )
    run_scan = st.button("🔄 Run Scan", type="primary", use_container_width=True)

if run_scan:
    with st.spinner(f"Running {scan_type}..."):
        trades = fetch_general_flow()
        if "General" in scan_type:
            st.markdown("### 📊 Market Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                sentiment_ratio, sentiment_label = calculate_sentiment_score(trades)
                st.metric("Market Sentiment", sentiment_label, f"{sentiment_ratio:.1%} calls")
            with col2:
                total_premium = sum(t.get('premium', 0) for t in trades)
                st.metric("Total Premium", f"${total_premium:,.0f}")
            with col3:
                st.metric("Total Trades", len(trades))

            trades = apply_filters(trades)
            visualize_market_summary(trades)
            with st.expander("💾 Export Data", expanded=False):
                save_to_csv(trades, "general_flow")

        elif "DTE" in scan_type:
            trades = apply_filters(trades)
            display_dte_segregated(trades)
            with st.expander("💾 Export Data", expanded=False):
                save_to_csv(trades, "dte_segregated_flow")

        elif "Alert" in scan_type:
            trades = apply_filters(trades)
            display_alerts(trades)
            with st.expander("💾 Export Data", expanded=False):
                save_to_csv(trades, "alerts")

else:
    st.markdown("""
    ## Welcome! 👋
    
    This application combines real-time options flow analysis with advanced pattern recognition.
    
    ### Available Views:
    - 🔍 General Flow Scanner
    - ⏰ DTE-Segregated Flow
    - 🚨 Smart Alert System
    
    Select a scan type from the sidebar and click **Run Scan** to begin!
    """)
