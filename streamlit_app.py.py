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

config = Config()

# --- API SETUP ---
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

# --- HELPER FUNCTIONS ---
def parse_option_chain(opt_str):
    try:
        ticker = ''.join([c for c in opt_str if c.isalpha()])[:-1]
        date_start = len(ticker)
        date_str = opt_str[date_start:date_start + 6]
        expiry_date = date(2000 + int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6]))
        dte = (expiry_date - date.today()).days
        option_type = opt_str[date_start + 6]
        strike = int(opt_str[date_start + 7:]) / 1000
        return ticker, expiry_date.strftime('%Y-%m-%d'), dte, option_type.upper(), strike
    except Exception as e:
        st.write(f"Error parsing option chain {opt_str}: {e}")
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

def fetch_trades():
    st.write("🔄 Fetching unusual options flow...")
    try:
        response = httpx.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            st.write(f"❌ API Error: {response.status_code} - {response.text}")
            return []
        data = response.json()
        trades = data.get('data', [])
        st.write(f"✅ Retrieved {len(trades)} potential trades")
    except Exception as e:
        st.write(f"❌ Error fetching data: {e}")
        return []

    result = []
    filtered_count = 0
    for trade in trades:
        option_chain = trade.get('option_chain', '')
        ticker, expiry, dte, opt_type, strike = parse_option_chain(option_chain)
        if not ticker or ticker in config.EXCLUDE_TICKERS:
            filtered_count += 1
            continue
        premium = float(trade.get('total_premium', 0))
        volume = trade.get('volume', 0)
        if premium < config.MIN_PREMIUM or volume < config.MIN_VOLUME:
            filtered_count += 1
            continue
        current_price = trade.get('underlying_price', 'N/A')
        result.append({
            'ticker': ticker,
            'option': option_chain,
            'type': opt_type,
            'strike': strike,
            'expiry': expiry,
            'dte': dte,
            'dte_category': get_time_to_expiry_category(dte),
            'price': trade.get('price', 'N/A'),
            'premium': premium,
            'volume': volume,
            'oi': trade.get('open_interest', 0),
            'vol_oi_ratio': volume / max(trade.get('open_interest', 1), 1),
            'underlying_price': current_price,
            'moneyness': calculate_moneyness(strike, current_price),
            'time': trade.get('created_at', 'N/A'),
            'sentiment': trade.get('sentiment', 'N/A'),
            'rule': trade.get('rule_name', 'N/A')
        })
    st.write(f"📊 Processed {len(result)} trades (filtered out {filtered_count})")
    return result

def analyze_flow_by_ticker(trades):
    ticker_analysis = defaultdict(lambda: {
        'call_premium': 0, 'put_premium': 0, 'total_volume': 0,
        'trades': [], 'avg_dte': 0, 'sentiment': 'Neutral'
    })
    for trade in trades:
        ticker = trade['ticker']
        ticker_analysis[ticker]['trades'].append(trade)
        ticker_analysis[ticker]['total_volume'] += trade['volume']
        if trade['type'] == 'C':
            ticker_analysis[ticker]['call_premium'] += trade['premium']
        else:
            ticker_analysis[ticker]['put_premium'] += trade['premium']
    for ticker, data in ticker_analysis.items():
        total_premium = data['call_premium'] + data['put_premium']
        if total_premium > 0:
            call_ratio = data['call_premium'] / total_premium
            if call_ratio > 0.65:
                data['sentiment'] = "Bullish"
            elif call_ratio < 0.35:
                data['sentiment'] = "Bearish"
            else:
                data['sentiment'] = "Mixed"
        data['avg_dte'] = sum(t['dte'] for t in data['trades']) / len(data['trades'])
    return dict(ticker_analysis)

def display_summary(trades):
    st.markdown("### 📊 Market Summary")
    sentiment_ratio, sentiment_label = calculate_sentiment_score(trades)
    total_premium = sum(t['premium'] for t in trades)
    st.write(f"💰 **Total Premium:** ${total_premium:,.0f}")
    st.write(f"🎯 **Market Sentiment:** {sentiment_label} ({sentiment_ratio:.1%} calls)")
    st.write(f"📈 **Total Trades:** {len(trades)}")

    ticker_data = analyze_flow_by_ticker(trades)
    top_tickers = sorted(ticker_data.items(),
                         key=lambda x: x[1]['call_premium'] + x[1]['put_premium'],
                         reverse=True)[:10]
    st.markdown("#### 🏆 Top Tickers by Premium")
    for i, (ticker, data) in enumerate(top_tickers, 1):
        total_prem = data['call_premium'] + data['put_premium']
        st.write(f"{i}. {ticker}: ${total_prem:,.0f} | {data['sentiment']} | {len(data['trades'])} trades")

def display_alerts(trades):
    alerts = []
    for trade in trades:
        score = 0
        reasons = []
        if trade['premium'] > 500000:
            score += 3
            reasons.append("Massive Premium")
        elif trade['premium'] > 250000:
            score += 2
            reasons.append("Large Premium")
        if trade['vol_oi_ratio'] > 2:
            score += 2
            reasons.append("High Vol/OI")
        if trade['dte'] <= 7 and trade['premium'] > 200000:
            score += 2
            reasons.append("Short-term + Size")
        if "ATM" in trade['moneyness'] or ("OTM" in trade['moneyness'] and "+5%" not in trade['moneyness']):
            score += 1
            reasons.append("Good Strike")
        if score >= 4:
            trade['alert_score'] = score
            trade['reasons'] = reasons
            alerts.append(trade)
    alerts.sort(key=lambda x: -x['alert_score'])
    for i, alert in enumerate(alerts[:10], 1):
        st.markdown(f"**{i}. {alert['ticker']} {alert['strike']:.0f}{alert['type']} {alert['expiry']} ({alert['dte']}d)**")
        st.write(f"💰 Premium: ${alert['premium']:,.0f} | Vol: {alert['volume']} | {alert['moneyness']}")
        st.write(f"📍 Reasons: {', '.join(alert['reasons'])}")

def display_trading_opportunities(trades):
    categories = defaultdict(lambda: {'calls': [], 'puts': []})
    for trade in trades:
        categories[trade['dte_category']][
            'calls' if trade['type'] == 'C' else 'puts'
        ].append(trade)

    for category in ['0DTE', 'Weekly', 'Monthly', 'Quarterly']:
        if category in categories:
            st.markdown(f"### 📆 {category} Opportunities")
            for label, trades_list in categories[category].items():
                if trades_list:
                    st.write(f"**{'🟢' if label == 'calls' else '🔴'} Top {label.title()}**")
                    for t in sorted(trades_list, key=lambda x: -x['premium'])[:5]:
                        st.write(
                            f"{t['ticker']} {t['strike']:.0f}{t['type']} {t['expiry']} ({t['dte']}d) | "
                            f"Prem: ${t['premium']:,.0f} | Vol: {t['volume']} | {t['moneyness']}"
                        )

def save_enhanced_csv(trades, filename=None):
    if not trades:
        st.write("❌ No trades to save")
        return
    if filename is None:
        filename = f'options_flow_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    fieldnames = [
        'ticker', 'option', 'type', 'strike', 'expiry', 'dte', 'dte_category',
        'price', 'premium', 'volume', 'oi', 'vol_oi_ratio', 'underlying_price',
        'moneyness', 'time', 'sentiment', 'rule', 'alert_score', 'reasons'
    ]
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in trades:
            row = t.copy()
            if isinstance(row.get('reasons'), list):
                row['reasons'] = ', '.join(row['reasons'])
            writer.writerow(row)
    st.success(f"💾 Saved to {filename}")

# --- STREAMLIT UI ---
st.set_page_config(page_title="Smart Flow Scanner", page_icon="📊", layout="wide")
st.title("📊 Smart Options Flow Tracker")

col1, col2 = st.columns([1, 4])
with col1:
    run_button = st.button("🔍 Scan Market", use_container_width=True)

with col2:
    st.markdown("### Market Pulse Dashboard")
    st.markdown("Analyze intelligent options activity to uncover strategic trades.")

st.markdown("---")

if run_button:
    with st.spinner("Fetching market activity..."):
        trades = fetch_trades()

    if not trades:
        st.error("❌ No data retrieved. Check your API token and connection.")
    else:
        with st.expander("📅 Trading Opportunities", expanded=True):
            display_trading_opportunities(trades)

        with st.expander("📈 Summary View", expanded=False):
            display_summary(trades)

        with st.expander("🚨 Alerts View", expanded=False):
            display_alerts(trades)

        with st.expander("💾 Export Data", expanded=False):
            save_enhanced_csv(trades)

st.markdown("---")
st.caption("Developed with 💡 using Streamlit")
