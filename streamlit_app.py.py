import httpx
import csv
from datetime import datetime, date
from collections import defaultdict

# Rich for beautiful terminal output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint

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

# Initialize rich console
console = Console(color_system="auto")

# --- API SETUP ---
headers = {
    'Accept': 'application/json',
    'Authorization': config.UW_TOKEN
}

params = {
    'issue_types[]': ['Common Stock', 'ADR'],
    'min_dte': 1,
    'min_volume_oi_ratio': 1.0,
    'rule_name[]': ['RepeatedHits', 'RepeatedHitsAscendingFill', 'RepeatedHitsDescendingFill'],
    'limit': config.LIMIT
}

url = 'https://unusualwhales.com/api/options_flow '  # Valid endpoint

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
        console.print(f"[red]Error parsing option chain {opt_str}: {e}[/red]")
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
    console.print("🔄 [bold blue]Fetching unusual options flow...[/bold blue]")
    try:
        response = httpx.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            console.print(f"[red]❌ API Error: {response.status_code} - {response.text}[/red]")
            return []
        data = response.json()
        trades = data.get('data', [])
        console.print(f"[green]✅ Retrieved {len(trades)} potential trades[/green]")
    except Exception as e:
        console.print(f"[red]❌ Error fetching data: {e}[/red]")
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
    console.print(f"[blue]📊 Processed {len(result)} trades (filtered out {filtered_count})[/blue]")
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
        console.print("[red]❌ No trades found matching criteria[/red]")
        return

    console.rule("[bold blue]🐋 Unusual Whales Options Flow Scanner[/bold blue]")
    console.print(Panel.fit(
        "[bold]📈 UNUSUAL OPTIONS FLOW SUMMARY[/bold]\n[dim]" +
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/dim]",
        border_style="blue"
    ))

    sentiment_ratio, sentiment_label = calculate_sentiment_score(trades)
    total_premium = sum(t['premium'] for t in trades)
    console.print(f"[bold]💰 Total Premium:[/bold] ${total_premium:,.0f}")
    console.print(f"[bold]🎯 Market Sentiment:[/bold] {sentiment_label} ({sentiment_ratio:.1%} calls)")
    console.print(f"[bold]📊 Total Trades:[/bold] {len(trades)}")

    ticker_data = analyze_flow_by_ticker(trades)
    top_tickers = sorted(ticker_data.items(),
                         key=lambda x: x[1]['call_premium'] + x[1]['put_premium'],
                         reverse=True)[:10]

    table = Table(title="🏆 Top 10 Tickers by Premium", show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="center")
    table.add_column("Ticker", justify="center")
    table.add_column("Premium ($)", justify="right")
    table.add_column("Sentiment", justify="center")
    table.add_column("Trades", justify="center")

    for i, (ticker, data) in enumerate(top_tickers, 1):
        total_prem = data['call_premium'] + data['put_premium']
        sentiment_color = {
            "Bullish": "green",
            "Bearish": "red",
            "Mixed": "yellow",
            "Neutral": "white"
        }.get(data['sentiment'], "white")
        table.add_row(
            str(i),
            ticker,
            f"[bold]{total_prem:,.0f}[/bold]",
            f"[{sentiment_color}]{data['sentiment']}[/{sentiment_color}]",
            str(len(data['trades']))
        )

    console.print(table)

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
        console.print("\n🚨 [bold red]HIGH CONVICTION ALERTS[/bold red]", end="")
        console.print(f" ([cyan]{len(alerts)}[/cyan] trades)\n")

        for i, alert in enumerate(alerts[:10], 1):
            console.print(f"[bold]{i:2d}. 🎯 {alert['ticker']} ${alert['strike']:.0f}{alert['type']} "
                          f"{alert['expiry']} ({alert['dte']}d)[/bold]")
            console.print(f"    💰 Premium: ${alert['premium']:,.0f} | Price: ${alert['price']} | "
                          f"Vol: {alert['volume']} | {alert['moneyness']}")
            console.print(f"    📍 Reasons: {', '.join(alert['reasons'])}\n")

def display_trading_opportunities(trades):
    categories = {}
    for trade in trades:
        cat = trade['dte_category']
        if cat not in categories:
            categories[cat] = {'calls': [], 'puts': []}
        trade_type = 'calls' if trade['type'] == 'C' else 'puts'
        categories[cat][trade_type].append(trade)

    for category in ['0DTE', 'Weekly', 'Monthly', 'Quarterly']:
        if category not in categories:
            continue
        calls = sorted(categories[category]['calls'], key=lambda x: -x['premium'])[:5]
        puts = sorted(categories[category]['puts'], key=lambda x: -x['premium'])[:5]
        if calls or puts:
            emoji = '🔥' if category == '0DTE' else '📅'
            console.print(f"\n[bold cyan]{emoji} {category.upper()} OPPORTUNITIES[/bold cyan]")
            console.print("=" * 70)

            if calls:
                call_table = Table(title="🟢 Top Call Opportunities", show_header=True, header_style="bold green")
                call_table.add_column("Ticker")
                call_table.add_column("Strike")
                call_table.add_column("Expiry")
                call_table.add_column("DTE")
                call_table.add_column("Premium")
                call_table.add_column("Volume")
                for t in calls:
                    call_table.add_row(
                        t['ticker'],
                        f"${t['strike']:.2f}",
                        t['expiry'],
                        str(t['dte']),
                        f"${t['premium']:,.0f}",
                        str(t['volume'])
                    )
                console.print(call_table)

            if puts:
                put_table = Table(title="🔴 Top Put Opportunities", show_header=True, header_style="bold red")
                put_table.add_column("Ticker")
                put_table.add_column("Strike")
                put_table.add_column("Expiry")
                put_table.add_column("DTE")
                put_table.add_column("Premium")
                put_table.add_column("Volume")
                for t in puts:
                    put_table.add_row(
                        t['ticker'],
                        f"${t['strike']:.2f}",
                        t['expiry'],
                        str(t['dte']),
                        f"${t['premium']:,.0f}",
                        str(t['volume'])
                    )
                console.print(put_table)

def save_enhanced_csv(trades, filename=None):
    if not trades:
        console.print("[red]❌ No trades to save[/red]")
        return
    if filename is None:
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
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clean_trades)
    console.print(f"[green]💾 Saved {len(trades)} trades to {filename}[/green]")

def main():
    console.print("[bold blue]🐋 Unusual Whales Options Flow Scanner[/bold blue]")
    console.print("=" * 50)
    trades = fetch_trades()
    if not trades:
        console.print("[red]❌ No data retrieved. Check your API token and connection.[/red]")
        return
    display_summary(trades)
    display_alerts(trades)
    display_trading_opportunities(trades)
    save_enhanced_csv(trades)
    console.print("\n[green]✅ Analysis complete![/green] Check the CSV file for detailed data.")
    console.print("[bold yellow]💡 Focus on high conviction alerts for potential trades.[/bold yellow]")

if __name__ == '__main__':
    main()
