import streamlit as st
import httpx
import csv
from datetime import datetime, date
from collections import defaultdict

# --- CONFIGURATION ---
class Config:
    UW_TOKEN = st.secrets["UW_TOKEN"]
    EXCLUDE_TICKERS = {'TSLA', 'MSTR', 'CRCL'}
    MIN_PREMIUM = 100000  # $100k minimum
    LIMIT = 250
    # Trading filters
    MIN_VOLUME = 50  # Minimum volume for liquidity
    MAX_SPREAD_PERCENT = 5  # Max bid-ask spread %
    MIN_OI_RATIO = 0.5  # Volume/OI ratio for unusual activity
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
url = 'https://api.unusualwhales.com/api/option-trades/flow-alerts '

# --- HELPER FUNCTIONS ---
def parse_option_chain(opt_str):
    """Parse option chain string into components"""
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
        st.markdown(f"<div style='color:red;'>❌ Error parsing option chain {opt_str}: {e}</div>", unsafe_allow_html=True)
        return None, None, None, None, None


def calculate_moneyness(strike, current_price):
    """Calculate how far ITM/OTM the option is"""
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
    """Categorize options by time to expiry"""
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
    """Calculate bullish/bearish sentiment from flow"""
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


# --- MAIN FUNCTIONS ---
def fetch_trades():
    """Fetch and process option flow data"""
    st.markdown("<div style='font-size:18px; color:#00ffff;'>🔄 Fetching unusual options flow...</div>", unsafe_allow_html=True)
    try:
        response = httpx.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            st.markdown(f"<div style='color:red;'>❌ API Error: {response.status_code} - Something went wrong</div>", unsafe_allow_html=True)
            return []
        data = response.json()
        trades = data.get('data', [])
        st.markdown(f"<div style='color:#00ff00;'>✅ Retrieved {len(trades)} potential trades</div>", unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f"<div style='color:red;'>❌ Error fetching data: {e}</div>", unsafe_allow_html=True)
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

    st.markdown(f"<div style='color:#ffff00;'>📊 Processed {len(result)} trades (filtered out {filtered_count})</div>", unsafe_allow_html=True)
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
    if not trades:
        st.markdown("<div style='color:red; font-size:20px;'>❌ No trades found matching criteria</div>", unsafe_allow_html=True)
        return

    st.markdown("---")
    st.markdown(f"<h3 style='color:#00ffff;'>🐋 UNUSUAL OPTIONS FLOW SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</h3>", unsafe_allow_html=True)
    st.markdown("---")

    sentiment_ratio, sentiment_label = calculate_sentiment_score(trades)
    total_premium = sum(t['premium'] for t in trades)

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Premium", f"${total_premium:,.0f}")
    col2.metric("🎯 Market Sentiment", f"{sentiment_label} ({sentiment_ratio:.1%} calls)")
    col3.metric("📊 Total Trades", f"{len(trades)}")

    ticker_data = analyze_flow_by_ticker(trades)
    top_tickers = sorted(ticker_data.items(),
                         key=lambda x: x[1]['call_premium'] + x[1]['put_premium'],
                         reverse=True)[:10]

    st.markdown("<h4>🏆 TOP 10 TICKERS BY PREMIUM</h4>", unsafe_allow_html=True)
    st.markdown("-" * 60)

    for i, (ticker, data) in enumerate(top_tickers, 1):
        total_prem = data['call_premium'] + data['put_premium']
        st.markdown(f"<b>{i:2d}. {ticker:5s}</b> | <span style='color:#00ff00;'>${total_prem:8,.0f}</span> | "
                    f"<span style='color:#ffff00;'>{data['sentiment']:8s}</span> | {len(data['trades'])} trades")


def display_trading_opportunities(trades):
    categories = {}
    for trade in trades:
        cat = trade['dte_category']
        if cat not in categories:
            categories[cat] = {'calls': [], 'puts': []}
        trade_type = 'calls' if trade['type'] == 'C' else 'puts'
        categories[cat][trade_type].append(trade)

    st.markdown("<h4>📅 OPTIONS OPPORTUNITIES BY TIMEFRAME</h4>", unsafe_allow_html=True)

    for category in ['0DTE', 'Weekly', 'Monthly', 'Quarterly']:
        if category not in categories:
            continue

        calls = sorted(categories[category]['calls'], key=lambda x: -x['premium'])[:5]
        puts = sorted(categories[category]['puts'], key=lambda x: -x['premium'])[:5]

        if calls or puts:
            st.markdown(f"<h5 style='color:#ff6600;'>{'🔥' if category == '0DTE' else '📅'} <u><b>{category.upper()} OPPORTUNITIES</b></u></h5>", unsafe_allow_html=True)

            if calls:
                st.markdown("<p style='color:#00ff00;'><b>🟢 TOP CALLS</b></p>", unsafe_allow_html=True)
                st.markdown("-" * 50)
                for i, t in enumerate(calls, 1):
                    st.markdown(f"{i}. <b>{t['ticker']:5s} ${t['strike']:6.0f}C {t['expiry']}</b> "
                               f"({t['dte']}d) | Price: ${t['price']:5s} | "
                               f"Premium: ${t['premium']:8,.0f} | Vol: {t['volume']:4d} | "
                               f"<i>{t['moneyness']:12s}</i>")

            if puts:
                st.markdown("<p style='color:#ff0000;'><b>🔴 TOP PUTS</b></p>", unsafe_allow_html=True)
                st.markdown("-" * 50)
                for i, t in enumerate(puts, 1):
                    st.markdown(f"{i}. <b>{t['ticker']:5s} ${t['strike']:6.0f}P {t['expiry']}</b> "
                               f"({t['dte']}d) | Price: ${t['price']:5s} | "
                               f"Premium: ${t['premium']:8,.0f} | Vol: {t['volume']:4d} | "
                               f"<i>{t['moneyness']:12s}</i>")


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

    if alerts:
        alerts.sort(key=lambda x: -x['alert_score'])
        st.markdown(f"<h4 style='color:#ff0000;'>🚨 HIGH CONVICTION ALERTS ({len(alerts)} trades)</h4>", unsafe_allow_html=True)
        st.markdown("=" * 70)

        for i, alert in enumerate(alerts[:10], 1):
            st.markdown(f"<b>{i:2d}. 🎯 {alert['ticker']} ${alert['strike']:.0f}{alert['type']} "
                        f"{alert['expiry']} ({alert['dte']}d)</b>")
            st.markdown(f"    💰 Premium: ${alert['premium']:,.0f} | Price: ${alert['price']} | "
                        f"Vol: {alert['volume']} | {alert['moneyness']}")
            st.markdown(f"    📍 Reasons: {', '.join(alert['reasons'])}")


# --- STREAMLIT UI ---
st.set_page_config(page_title="Options Flow Scanner", page_icon="🐋", layout="wide")

# Dark mode styling
st.markdown("""
<style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .sidebar .sidebar-content {
        background-color: #111111;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='color:#00ffff;'>🐋 Unusual Whales Options Flow Scanner</h1>", unsafe_allow_html=True)
st.markdown("<p>Powered by <a href='https://unusualwhales.com '>Unusual Whales API</a></p>", unsafe_allow_html=True)

if st.button("Run Options Flow Scanner"):
    trades = fetch_trades()
    if not trades:
        st.markdown("<div style='color:red; font-size:18px;'>❌ No data retrieved. Check your API token and connection.</div>", unsafe_allow_html=True)
    else:
        display_summary(trades)
        display_alerts(trades)
        display_trading_opportunities(trades)
