import streamlit as st
import httpx
import csv
from datetime import datetime, date
from collections import defaultdict
import pandas as pd
from io import StringIO

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
        st.write(f"Error parsing option chain {opt_str}: {e}")
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
    st.info("🔄 Fetching unusual options flow...")
    try:
        response = httpx.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            st.error(f"❌ API Error: {response.status_code} - {response.text}")
            return []
        data = response.json()
        trades = data.get('data', [])
        st.success(f"✅ Retrieved {len(trades)} potential trades")
    except Exception as e:
        st.error(f"❌ Error fetching data: {e}")
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
    st.info(f"📊 Processed {len(result)} trades (filtered out {filtered_count})")
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
        st.error("❌ No trades found matching criteria")
        return

    st.markdown("### 📈 Market Summary")
    sentiment_ratio, sentiment_label = calculate_sentiment_score(trades)
    df_summary = pd.DataFrame({
        'Metric': [
            'Total Premium ($)',
            'Market Sentiment',
            'Total Trades'
        ],
        'Value': [
            f"${sum(t['premium'] for t in trades):,.0f}",
            f"{sentiment_label} ({sentiment_ratio:.1%} calls)",
            len(trades)
        ]
    })
    st.dataframe(df_summary, hide_index=True, use_container_width=True)

    ticker_data = analyze_flow_by_ticker(trades)
    top_tickers = sorted(ticker_data.items(),
                         key=lambda x: x[1]['call_premium'] + x[1]['put_premium'],
                         reverse=True)[:10]

    st.markdown("### 🏆 Top 10 Tickers by Premium")
    top_data = []
    for i, (ticker, data) in enumerate(top_tickers, 1):
        total_prem = data['call_premium'] + data['put_premium']
        top_data.append({
            'Rank': i,
            'Ticker': ticker,
            'Premium ($)': f"${total_prem:,.0f}",
            'Sentiment': data['sentiment'],
            '# Trades': len(data['trades'])
        })

    df_top = pd.DataFrame(top_data)
    st.dataframe(df_top, hide_index=True, use_container_width=True)

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

        alert_data = []
        for i, alert in enumerate(alerts[:10], 1):
            alert_data.append({
                'Alert': f"{i}. 🎯 {alert['ticker']} ${alert['strike']:.0f}{alert['type']} {alert['expiry']} ({alert['dte']}d)",
                'Premium': f"${alert['premium']:,.0f}",
                'Price': f"${alert['price']}",
                'Volume': alert['volume'],
                'Moneyness': alert['moneyness'],
                'Reasons': ', '.join(alert['reasons'])
            })

        df_alerts = pd.DataFrame(alert_data)
        st.markdown("### 🚨 High Conviction Alerts")
        st.dataframe(df_alerts, hide_index=True, use_container_width=True)

def save_enhanced_csv(trades):
    if not trades:
        st.error("❌ No trades to save")
        return ""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'options_flow_{timestamp}.csv'

    fieldnames = [
        'ticker', 'option', 'type', 'strike', 'expiry', 'dte', 'dte_category',
        'price', 'premium', 'volume', 'oi', 'vol_oi_ratio', 'underlying_price',
        'moneyness', 'time', 'sentiment', 'rule', 'alert_score', 'reasons'
    ]

    clean_trades = []
    for trade in trades:
        clean_trade = {}
        for field in fieldnames:
            if field == 'reasons' and field in trade:
                clean_trade[field] = ', '.join(trade[field]) if isinstance(trade[field], list) else ''
            else:
                clean_trade[field] = trade.get(field, '')
        clean_trades.append(clean_trade)

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(clean_trades)
    return output.getvalue()

# --- STREAMLIT UI ---
st.set_page_config(page_title="Options Flow Scanner", page_icon="🐋", layout="wide")

st.title("🐋 Unusual Whales Options Flow Scanner")
st.markdown("Powered by [Unusual Whales API](https://unusualwhales.com )")

if st.button("Run Options Flow Scanner"):
    trades = fetch_trades()
    if not trades:
        st.error("❌ No data retrieved. Check your API token and connection.")
    else:
        with st.expander("📈 Market Summary", expanded=True):
            display_summary(trades)
        with st.expander("🟢/🔴 Trading Opportunities", expanded=True):
            display_trading_opportunities(trades)
        with st.expander("🚨 High Conviction Alerts", expanded=True):
            display_alerts(trades)

        csv_data = save_enhanced_csv(trades)
        st.download_button(
            label="📥 Download Report",
            data=csv_data,
            file_name=f"options_flow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )
