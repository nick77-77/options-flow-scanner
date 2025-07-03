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
    MIN_PREMIUM = 50000  # Lower threshold for more activity
    LIMIT = 1000
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

def detect_basic_scenarios(trade, underlying_price=None):
    """Simplified scenario detection"""
    scenarios = []
    opt_type = trade['type']
    strike = trade['strike']
    premium = trade['premium']
    volume = trade.get('volume', 0)
    oi = trade.get('oi', 0)
    
    if underlying_price is None:
        underlying_price = strike

    # Basic moneyness
    moneyness = "ATM"
    if opt_type == 'C' and strike > underlying_price:
        moneyness = "OTM"
    elif opt_type == 'C' and strike < underlying_price:
        moneyness = "ITM"
    elif opt_type == 'P' and strike < underlying_price:
        moneyness = "OTM"
    elif opt_type == 'P' and strike > underlying_price:
        moneyness = "ITM"

    # Simple scenarios
    if premium >= 500000:
        scenarios.append("Large Premium")
    elif premium >= 200000:
        scenarios.append("Medium Premium")
    
    if volume >= 1000:
        scenarios.append("High Volume")
    elif volume >= 500:
        scenarios.append("Medium Volume")
    
    if opt_type == 'C' and moneyness == 'OTM':
        scenarios.append("OTM Call")
    elif opt_type == 'C' and moneyness == 'ITM':
        scenarios.append("ITM Call")
    elif opt_type == 'P' and moneyness == 'OTM':
        scenarios.append("OTM Put")
    elif opt_type == 'P' and moneyness == 'ITM':
        scenarios.append("ITM Put")
    
    if volume > oi * 2:
        scenarios.append("Fresh Interest")
    
    return scenarios if scenarios else ["Standard Trade"]

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

def calculate_sentiment_score(trades):
    call_premium = sum(t['premium'] for t in trades if t['type'] == 'C')
    put_premium = sum(t['premium'] for t in trades if t['type'] == 'P')
    total = call_premium + put_premium
    if total == 0:
        return 0, "Neutral"
    call_ratio = call_premium / total
    if call_ratio > 0.65:
        return call_ratio, "Bullish"
    elif call_ratio > 0.35:
        return call_ratio, "Neutral"
    else:
        return call_ratio, "Bearish"

# --- SIMPLIFIED FETCH FUNCTION ---
def fetch_simple_flow():
    params = {
        'issue_types[]': ['Common Stock', 'ADR'],
        'min_dte': 0,
        'min_volume_oi_ratio': 0.1,
        'limit': config.LIMIT
    }
    try:
        response = httpx.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return []
        
        data = response.json().get('data', [])
        result = []

        for trade in data:
            option_chain = trade.get('option_chain', '')
            ticker, expiry, dte, opt_type, strike = parse_option_chain(option_chain)

            if not ticker or ticker in config.EXCLUDE_TICKERS:
                continue

            try:
                premium = float(trade.get('total_premium', 0))
                volume = int(trade.get('volume', 0))
                oi = int(trade.get('open_interest', 0))
                price = float(trade.get('price', 0)) if trade.get('price') not in ['N/A', '', None] else 0
                underlying_price = float(trade.get('underlying_price', strike)) if trade.get('underlying_price') not in ['N/A', '', None] else strike
            except (ValueError, TypeError):
                continue

            # Time parsing
            utc_time_str = trade.get('created_at', 'N/A')
            ny_time_str = "N/A"
            if utc_time_str and utc_time_str != "N/A":
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
                'dte_category': get_time_to_expiry_category(dte),
                'price': price,
                'premium': premium,
                'volume': volume,
                'oi': oi,
                'time_ny': ny_time_str,
                'underlying_price': underlying_price,
                'moneyness': calculate_moneyness(strike, underlying_price),
                'vol_oi_ratio': volume / max(oi, 1)
            }
            
            # Add simple scenarios
            scenarios = detect_basic_scenarios(trade_data, underlying_price)
            trade_data['scenarios'] = scenarios
            result.append(trade_data)

        return result

    except Exception as e:
        st.error(f"Error fetching flow: {e}")
        return []

# --- SIMPLE VISUALIZATIONS ---
def create_simple_charts(trades):
    if not trades:
        return
    
    df = pd.DataFrame(trades)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Call vs Put pie chart
        type_counts = df['type'].value_counts()
        fig = px.pie(values=type_counts.values, names=type_counts.index, 
                     title="📊 Calls vs Puts",
                     color_discrete_map={'C': '#00ff00', 'P': '#ff0000'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Premium by ticker
        ticker_premium = df.groupby('ticker')['premium'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=ticker_premium.index, y=ticker_premium.values,
                     title="💰 Top Tickers by Premium",
                     labels={'x': 'Ticker', 'y': 'Total Premium'})
        st.plotly_chart(fig, use_container_width=True)

# --- SIMPLE DISPLAY FUNCTIONS ---
def display_calls_and_puts(trades):
    """Clean call/put separation"""
    call_trades = [t for t in trades if t['type'] == 'C']
    put_trades = [t for t in trades if t['type'] == 'P']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🟢 CALL OPTIONS")
        if call_trades:
            call_df = pd.DataFrame([{
                'Ticker': t['ticker'],
                'Strike': f"${t['strike']:.0f}",
                'Price': f"${t['price']:.2f}" if t['price'] > 0 else 'N/A',
                'Premium': f"${t['premium']:,.0f}",
                'Volume': t['volume'],
                'DTE': t['dte'],
                'Category': t.get('dte_category', 'Unknown'),
                'Moneyness': t['moneyness'],
                'Time': t['time_ny']
            } for t in sorted(call_trades, key=lambda x: x['premium'], reverse=True)[:25]])
            st.dataframe(call_df, use_container_width=True, height=600)
        else:
            st.info("No call trades found")
    
    with col2:
        st.markdown("### 🔴 PUT OPTIONS")
        if put_trades:
            put_df = pd.DataFrame([{
                'Ticker': t['ticker'],
                'Strike': f"${t['strike']:.0f}",
                'Price': f"${t['price']:.2f}" if t['price'] > 0 else 'N/A',
                'Premium': f"${t['premium']:,.0f}",
                'Volume': t['volume'],
                'DTE': t['dte'],
                'Category': t.get('dte_category', 'Unknown'),
                'Moneyness': t['moneyness'],
                'Time': t['time_ny']
            } for t in sorted(put_trades, key=lambda x: x['premium'], reverse=True)[:25]])
            st.dataframe(put_df, use_container_width=True, height=600)
        else:
            st.info("No put trades found")

def display_top_trades(trades):
    """Show top trades overall"""
    st.markdown("### 🏆 Top Trades by Premium")
    
    if trades:
        top_df = pd.DataFrame([{
            'Ticker': t['ticker'],
            'Type': '🟢 CALL' if t['type'] == 'C' else '🔴 PUT',
            'Strike': f"${t['strike']:.0f}",
            'Price': f"${t['price']:.2f}" if t['price'] > 0 else 'N/A',
            'Premium': f"${t['premium']:,.0f}",
            'Volume': t['volume'],
            'DTE': t['dte'],
            'Category': t.get('dte_category', 'Unknown'),
            'Moneyness': t['moneyness'],
            'Scenarios': ', '.join(t['scenarios'][:2]),
            'Time': t['time_ny']
        } for t in sorted(trades, key=lambda x: x['premium'], reverse=True)[:20]])
        st.dataframe(top_df, use_container_width=True)
    else:
        st.info("No trades found")

# --- STREAMLIT UI ---
st.set_page_config(page_title="Simple Options Flow", page_icon="📈", layout="wide")
st.title("📈 Simple Options Flow Tracker")
st.markdown("### Clean, straightforward options activity monitoring")

with st.sidebar:
    st.markdown("## ⚙️ Simple Controls")
    
    # Premium filter
    premium_filter = st.selectbox(
        "Premium Range:",
        ["All", "Under $100K", "Under $250K", "$100K+", "$250K+", "$500K+"]
    )
    
    # DTE filter
    dte_filter = st.selectbox(
        "Days to Expiry:",
        ["All", "0DTE", "Weekly (≤7d)", "Monthly (≤30d)", "Quarterly (≤90d)", "LEAPS (>90d)"]
    )
    
    # View selection
    view_type = st.selectbox(
        "View Type:",
        ["📊 Calls & Puts", "🏆 Top Trades", "📈 Charts & Summary", "⏰ DTE Categories"]
    )
    
    run_scan = st.button("🔄 Scan Options Flow", type="primary", use_container_width=True)

# Main execution
if run_scan:
    with st.spinner("Scanning options flow..."):
        trades = fetch_simple_flow()
        
        if not trades:
            st.error("No trades found. Check API connection.")
            st.stop()
        
        # Apply filters
        filtered_trades = []
        for trade in trades:
            # Premium filter
            premium_ok = True
            if premium_filter == "Under $100K" and trade['premium'] >= 100000:
                premium_ok = False
            elif premium_filter == "Under $250K" and trade['premium'] >= 250000:
                premium_ok = False
            elif premium_filter == "$100K+" and trade['premium'] < 100000:
                premium_ok = False
            elif premium_filter == "$250K+" and trade['premium'] < 250000:
                premium_ok = False
            elif premium_filter == "$500K+" and trade['premium'] < 500000:
                premium_ok = False
            
            # DTE filter
            dte_ok = True
            if dte_filter == "0DTE" and trade['dte'] != 0:
                dte_ok = False
            elif dte_filter == "Weekly (≤7d)" and trade['dte'] > 7:
                dte_ok = False
            elif dte_filter == "Monthly (≤30d)" and trade['dte'] > 30:
                dte_ok = False
            elif dte_filter == "Quarterly (≤90d)" and trade['dte'] > 90:
                dte_ok = False
            elif dte_filter == "LEAPS (>90d)" and trade['dte'] <= 90:
                dte_ok = False
            
            if premium_ok and dte_ok:
                filtered_trades.append(trade)
        
        # Display results
        st.success(f"Found {len(filtered_trades)} trades (Premium: {premium_filter}, DTE: {dte_filter})")
        
        if filtered_trades:
            # Quick stats
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                call_count = len([t for t in filtered_trades if t['type'] == 'C'])
                st.metric("📞 Calls", call_count)
            
            with col2:
                put_count = len([t for t in filtered_trades if t['type'] == 'P'])
                st.metric("📉 Puts", put_count)
            
            with col3:
                total_premium = sum(t['premium'] for t in filtered_trades)
                st.metric("💰 Total Premium", f"${total_premium:,.0f}")
            
            with col4:
                sentiment_ratio, sentiment = calculate_sentiment_score(filtered_trades)
                st.metric("📊 Sentiment", sentiment)
            
            st.divider()
            
            # Show selected view
            if view_type == "📊 Calls & Puts":
                display_calls_and_puts(filtered_trades)
            elif view_type == "🏆 Top Trades":
                display_top_trades(filtered_trades)
            elif view_type == "📈 Charts & Summary":
                create_simple_charts(filtered_trades)
                st.markdown("---")
                display_top_trades(filtered_trades)
            elif view_type == "⏰ DTE Categories":
                # DTE Category view
                st.markdown("### ⏰ Trades by Time to Expiry")
                
                # Group trades by DTE category
                dte_groups = {}
                for trade in filtered_trades:
                    category = trade.get('dte_category', 'Unknown')
                    if category not in dte_groups:
                        dte_groups[category] = []
                    dte_groups[category].append(trade)
                
                # Display each category
                for category in ["0DTE", "Weekly", "Monthly", "Quarterly", "LEAPS"]:
                    if category in dte_groups:
                        trades_in_category = dte_groups[category]
                        
                        # Category header with stats
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.markdown(f"#### {category} ({len(trades_in_category)} trades)")
                        with col2:
                            total_premium = sum(t['premium'] for t in trades_in_category)
                            st.metric("Total Premium", f"${total_premium:,.0f}")
                        with col3:
                            call_ratio = len([t for t in trades_in_category if t['type'] == 'C']) / len(trades_in_category)
                            st.metric("Call %", f"{call_ratio:.0%}")
                        
                        # Top trades in this category
                        category_df = pd.DataFrame([{
                            'Ticker': t['ticker'],
                            'Type': '🟢 CALL' if t['type'] == 'C' else '🔴 PUT',
                            'Strike': f"${t['strike']:.0f}",
                            'Price': f"${t['price']:.2f}" if t['price'] > 0 else 'N/A',
                            'Premium': f"${t['premium']:,.0f}",
                            'Volume': t['volume'],
                            'DTE': t['dte'],
                            'Time': t['time_ny']
                        } for t in sorted(trades_in_category, key=lambda x: x['premium'], reverse=True)[:10]])
                        
                        st.dataframe(category_df, use_container_width=True)
                        st.markdown("---")
            
            # Simple export
            st.markdown("---")
            with st.expander("💾 Export Data"):
                if filtered_trades:
                    csv_data = []
                    for trade in filtered_trades:
                        row = trade.copy()
                        row['scenarios'] = ', '.join(row.get('scenarios', []))
                        csv_data.append(row)
                    
                    df = pd.DataFrame(csv_data)
                    csv = df.to_csv(index=False)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    st.download_button(
                        label=f"📥 Download CSV ({len(filtered_trades)} trades)",
                        data=csv,
                        file_name=f"simple_options_flow_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        else:
            st.warning("No trades match your filters. Try expanding the criteria.")

else:
    st.markdown("""
    ## Welcome! 👋
    
    This is a simplified options flow tracker focused on **clean, easy-to-read data**.
    
    ### What You Get:
    - **📞 Clear Call/Put Separation** - Side-by-side tables
    - **💰 Contract Prices** - See what traders actually paid
    - **🎯 Simple Filters** - Premium ranges and time filters
    - **📊 Clean Charts** - Basic visualizations without clutter
    
    ### Views Available:
    - **📊 Calls & Puts** - Side-by-side comparison
    - **🏆 Top Trades** - Biggest premium trades
    - **📈 Charts & Summary** - Visual overview
    
    **Click "Scan Options Flow" to start!**
    """)
