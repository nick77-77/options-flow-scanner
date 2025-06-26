import httpx
import csv
import json
from datetime import datetime, date
from collections import defaultdict
import time


# --- CONFIGURATION ---
class Config:
    UW_TOKEN = "e6e8601a-0746-4cec-a07d-c3eabfc13926"
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

url = 'https://api.unusualwhales.com/api/option-trades/flow-alerts'


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
        print(f"Error parsing option chain {opt_str}: {e}")
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
    print("🔄 Fetching unusual options flow...")

    try:
        response = httpx.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return []

        data = response.json()
        trades = data.get('data', [])
        print(f"✅ Retrieved {len(trades)} potential trades")

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
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

        # Apply trading filters
        if premium < config.MIN_PREMIUM or volume < config.MIN_VOLUME:
            filtered_count += 1
            continue

        # Get current stock price (if available)
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

    print(f"📊 Processed {len(result)} trades (filtered out {filtered_count})")
    return result


def analyze_flow_by_ticker(trades):
    """Analyze flow patterns by ticker"""
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

    # Calculate sentiment for each ticker
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
    """Display market summary and top movers"""
    if not trades:
        print("❌ No trades found matching criteria")
        return

    print(f"\n{'=' * 80}")
    print(f"📈 UNUSUAL OPTIONS FLOW SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}")

    # Overall sentiment
    sentiment_ratio, sentiment_label = calculate_sentiment_score(trades)
    total_premium = sum(t['premium'] for t in trades)

    print(f"💰 Total Premium: ${total_premium:,.0f}")
    print(f"🎯 Market Sentiment: {sentiment_label} ({sentiment_ratio:.1%} calls)")
    print(f"📊 Total Trades: {len(trades)}")

    # Ticker analysis
    ticker_data = analyze_flow_by_ticker(trades)
    top_tickers = sorted(ticker_data.items(),
                         key=lambda x: x[1]['call_premium'] + x[1]['put_premium'],
                         reverse=True)[:10]

    print(f"\n🏆 TOP 10 TICKERS BY PREMIUM")
    print("-" * 60)
    for i, (ticker, data) in enumerate(top_tickers, 1):
        total_prem = data['call_premium'] + data['put_premium']
        print(f"{i:2d}. {ticker:5s} | ${total_prem:8,.0f} | "
              f"{data['sentiment']:8s} | {len(data['trades'])} trades")


def display_trading_opportunities(trades):
    """Display organized trading opportunities"""

    # Separate by DTE categories and type
    categories = {}
    for trade in trades:
        cat = trade['dte_category']
        if cat not in categories:
            categories[cat] = {'calls': [], 'puts': []}

        # Fix the type mapping
        trade_type = 'calls' if trade['type'] == 'C' else 'puts'
        categories[cat][trade_type].append(trade)

    # Sort and display each category
    for category in ['0DTE', 'Weekly', 'Monthly', 'Quarterly']:
        if category not in categories:
            continue

        calls = sorted(categories[category]['calls'], key=lambda x: -x['premium'])[:5]
        puts = sorted(categories[category]['puts'], key=lambda x: -x['premium'])[:5]

        if calls or puts:
            print(f"\n{'🔥' if category == '0DTE' else '📅'} {category.upper()} OPPORTUNITIES")
            print("=" * 70)

            if calls:
                print(f"\n🟢 TOP CALLS")
                print("-" * 50)
                for i, t in enumerate(calls, 1):
                    print(f"{i}. {t['ticker']:5s} ${t['strike']:6.0f}C {t['expiry']} "
                          f"({t['dte']}d) | ${t['price']:5s} | "
                          f"Prem: ${t['premium']:8,.0f} | Vol: {t['volume']:4d} | "
                          f"{t['moneyness']:12s}")

            if puts:
                print(f"\n🔴 TOP PUTS")
                print("-" * 50)
                for i, t in enumerate(puts, 1):
                    print(f"{i}. {t['ticker']:5s} ${t['strike']:6.0f}P {t['expiry']} "
                          f"({t['dte']}d) | ${t['price']:5s} | "
                          f"Prem: ${t['premium']:8,.0f} | Vol: {t['volume']:4d} | "
                          f"{t['moneyness']:12s}")


def display_alerts(trades):
    """Display high-conviction trade alerts"""
    # Filter for highest conviction trades
    alerts = []
    for trade in trades:
        score = 0
        reasons = []

        # High premium activity
        if trade['premium'] > 500000:  # $500k+
            score += 3
            reasons.append("Massive Premium")
        elif trade['premium'] > 250000:  # $250k+
            score += 2
            reasons.append("Large Premium")

        # High volume/OI ratio (unusual activity)
        if trade['vol_oi_ratio'] > 2:
            score += 2
            reasons.append("High Vol/OI")

        # Near-term expiry with size
        if trade['dte'] <= 7 and trade['premium'] > 200000:
            score += 2
            reasons.append("Short-term + Size")

        # ATM or slightly OTM options
        if "ATM" in trade['moneyness'] or ("OTM" in trade['moneyness'] and "+5%" not in trade['moneyness']):
            score += 1
            reasons.append("Good Strike")

        if score >= 4:  # High conviction threshold
            trade['alert_score'] = score
            trade['reasons'] = reasons
            alerts.append(trade)

    if alerts:
        alerts.sort(key=lambda x: -x['alert_score'])
        print(f"\n🚨 HIGH CONVICTION ALERTS ({len(alerts)} trades)")
        print("=" * 70)

        for i, alert in enumerate(alerts[:10], 1):
            print(f"\n{i:2d}. 🎯 {alert['ticker']} ${alert['strike']:.0f}{alert['type']} "
                  f"{alert['expiry']} ({alert['dte']}d)")
            print(f"    💰 Premium: ${alert['premium']:,.0f} | Price: ${alert['price']} | "
                  f"Vol: {alert['volume']} | {alert['moneyness']}")
            print(f"    📍 Reasons: {', '.join(alert['reasons'])}")


def save_enhanced_csv(trades, filename=None):
    """Save trades with enhanced data to CSV"""
    if not trades:
        print("❌ No trades to save")
        return

    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'options_flow_{timestamp}.csv'

    # Enhanced fieldnames for better analysis - include all possible fields
    fieldnames = [
        'ticker', 'option', 'type', 'strike', 'expiry', 'dte', 'dte_category',
        'price', 'premium', 'volume', 'oi', 'vol_oi_ratio', 'underlying_price',
        'moneyness', 'time', 'sentiment', 'rule', 'alert_score', 'reasons'
    ]

    # Clean trades data to only include defined fieldnames
    clean_trades = []
    for trade in trades:
        clean_trade = {}
        for field in fieldnames:
            if field == 'reasons' and field in trade:
                # Convert reasons list to string for CSV
                clean_trade[field] = ', '.join(trade[field]) if isinstance(trade[field], list) else trade.get(field, '')
            else:
                clean_trade[field] = trade.get(field, '')
        clean_trades.append(clean_trade)

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_trades)

    print(f"💾 Saved {len(trades)} trades to {filename}")


def main():
    """Main execution function"""
    print("🐋 Unusual Whales Options Flow Scanner")
    print("=" * 50)

    # Fetch data
    trades = fetch_trades()

    if not trades:
        print("❌ No data retrieved. Check your API token and connection.")
        return

    # Display analysis
    display_summary(trades)
    display_alerts(trades)
    display_trading_opportunities(trades)

    # Save data
    save_enhanced_csv(trades)

    print(f"\n{'=' * 50}")
    print("✅ Analysis complete! Check the CSV file for detailed data.")
    print("💡 Focus on high conviction alerts for potential trades.")


if __name__ == '__main__':
    main()