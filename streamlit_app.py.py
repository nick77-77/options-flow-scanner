# smart_flow_combined.py
import streamlit as st
import httpx
import csv
from datetime import datetime, date
from collections import defaultdict

# --- CONFIGURATION ---
class Config:
    UW_TOKEN = st.secrets["UW_TOKEN"]
    EXCLUDE_TICKERS = {'TSLA', 'MSTR', 'CRCL'}
    MIN_PREMIUM = 100000
    LIMIT = 250
    MIN_VOLUME = 50
    MAX_SPREAD_PERCENT = 5
    MIN_OI_RATIO = 0.5
    ALLOWED_ETF_TICKERS = {'QQQ', 'SPY', 'IWM'}

config = Config()

headers = {
    'Accept': 'application/json, text/plain',
    'Authorization': config.UW_TOKEN
}

params = {
    'issue_types[]': ['Common Stock', 'ADR'],
    'min_dte': 1,
    'min_volume_oi_ratio': 1.0,
    'rule_name[]': ['RepeatedHits', 'RepeatedHitsAscendingFill', 'RepeatedHitsDescendingFill'],
    'limit': config.LIMIT
}

url = 'https://api.unusualwhales.com/api/option-trades/flow-alerts'

# --- UTILITY FUNCTIONS ---
def parse_option_chain(opt_str):
    try:
        ticker = ''.join([c for c in opt_str if c.isalpha()])[:-1]
        date_start = len(ticker)
        date_str = opt_str[date_start:date_start + 6]
        expiry_date = date(2000 + int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6]))
        dte = (expiry_date - date.today()).days
        option_type = opt_str[date_start + 6]
        strike = int(opt_str[date_start + 7:]) / 1000
        return ticker.upper(), expiry_date.strftime('%Y-%m-%d'), dte, option_type.upper(), strike
    except Exception:
        return None, None, None, None, None

# --- FETCH FUNCTIONS ---
def fetch_all_trades():
    try:
        response = httpx.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get('data', [])
    except Exception as e:
        st.write(f"❌ Error fetching data: {e}")
        return []

def fetch_and_filter_etf_trades():
    all_trades = fetch_all_trades()
    filtered = []
    for trade in all_trades:
        ticker, expiry, dte, opt_type, strike = parse_option_chain(trade.get('option_chain', ''))
        if not ticker or ticker not in config.ALLOWED_ETF_TICKERS or dte is None or dte > 7:
            continue
        filtered.append({
            'ticker': ticker,
            'type': opt_type,
            'strike': strike,
            'expiry': expiry,
            'dte': dte,
            'side': trade.get('side', 'N/A'),
            'price': trade.get('price', 'N/A'),
            'premium': trade.get('total_premium', 'N/A'),
            'volume': trade.get('volume', 'N/A'),
            'oi': trade.get('open_interest', 'N/A'),
            'time': trade.get('created_at', 'N/A'),
            'option': trade.get('option_chain', '')
        })
    return filtered

def fetch_and_categorize_flows():
    trades = fetch_all_trades()
    result = []
    for trade in trades:
        ticker, expiry, dte, opt_type, strike = parse_option_chain(trade.get('option_chain', ''))
        if not ticker or ticker in config.EXCLUDE_TICKERS:
            continue
        premium = float(trade.get('total_premium', 0))
        if premium < config.MIN_PREMIUM:
            continue
        result.append({
            'ticker': ticker,
            'option': trade.get('option_chain', ''),
            'type': opt_type,
            'strike': strike,
            'expiry': expiry,
            'dte': dte,
            'price': trade.get('price', 'N/A'),
            'premium': premium,
            'volume': trade.get('volume', 'N/A'),
            'oi': trade.get('open_interest', 'N/A'),
            'time': trade.get('created_at', 'N/A')
        })
    return result

def display_top_flows(trades):
    calls_lt7, calls_gte7 = [], []
    puts_lt7, puts_gte7 = [], []
    for t in trades:
        is_call = t['type'] == 'C'
        is_short_dte = t['dte'] < 7
        if is_call and is_short_dte:
            calls_lt7.append(t)
        elif is_call:
            calls_gte7.append(t)
        elif not is_call and is_short_dte:
            puts_lt7.append(t)
        else:
            puts_gte7.append(t)

    def show_section(title, trade_list):
        st.markdown(f"#### {title}")
        for i, t in enumerate(trade_list[:10], 1):
            st.write(f"{i:02d}. {t['ticker']} {t['expiry']} ({t['dte']} DTE) {t['type']} {t['strike']} | \
                      Price: ${t['price']} | Prem: ${int(t['premium'])} | Vol: {t['volume']} | OI: {t['oi']} | Time: {t['time']}")

    show_section("🟢 CALLS (< 7 DTE)", calls_lt7)
    show_section("🟢 CALLS (≥ 7 DTE)", calls_gte7)
    show_section("🔴 PUTS (< 7 DTE)", puts_lt7)
    show_section("🔴 PUTS (≥ 7 DTE)", puts_gte7)

def save_to_csv(trades, filename='unusual_whales_trades.csv'):
    if not trades:
        st.write("No trades to save.")
        return
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=trades[0].keys())
        writer.writeheader()
        writer.writerows(trades)
    st.success(f"📁 Saved {len(trades)} trades to {filename}")

# --- STREAMLIT UI ---
st.set_page_config(page_title="Combined Flow Tool", page_icon="📊", layout="wide")
st.title("📊 Combined Options Flow Tracker")

if st.button("🔍 Run Full Market Scan"):
    with st.spinner("Fetching and analyzing data..."):
        all_trades = fetch_and_categorize_flows()
        display_top_flows(all_trades)
        save_to_csv(all_trades)

if st.button("📊 Focused Scan: SPY, QQQ, IWM < 7 DTE"):
    with st.spinner("Fetching ETF trades..."):
        etf_trades = fetch_and_filter_etf_trades()
        display_top_flows(etf_trades)
        save_to_csv(etf_trades)

st.markdown("---")
st.caption("Developed with 🧠 ")
