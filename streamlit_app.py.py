import streamlit as st
import httpx
import csv
from datetime import datetime, date
from collections import defaultdict
import pandas as pd

# --- CONFIGURATION ---
class Config:
    UW_TOKEN = st.secrets.get("UW_TOKEN", "e6e8601a-0746-4cec-a07d-c3eabfc13926")
    EXCLUDE_TICKERS = {'TSLA', 'MSTR', 'CRCL'}
    ALLOWED_TICKERS = {'QQQ', 'SPY', 'IWM'}
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

url = 'https://api.unusualwhales.com/api/option-trades/flow-alerts'

# --- HELPER FUNCTIONS ---
def parse_option_chain(opt_str):
    """Parse option chain string to extract components"""
    try:
        # Find first digit to separate ticker from date
        idx = next(i for i, c in enumerate(opt_str) if c.isdigit())
        ticker = opt_str[:idx].upper()
        
        date_str = opt_str[idx:idx+6]
        expiry_date = date(2000 + int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6]))
        dte = (expiry_date - date.today()).days
        option_type = opt_str[idx+6].upper()
        strike = int(opt_str[idx+7:]) / 1000
        
        return ticker, expiry_date.strftime('%Y-%m-%d'), dte, option_type, strike
    except Exception as e:
        return None, None, None, None, None

def calculate_moneyness(strike, current_price):
    """Calculate moneyness of option"""
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
    """Categorize time to expiry"""
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
    """Calculate overall market sentiment based on call/put premium"""
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

# --- MAIN FETCH FUNCTIONS ---
def fetch_general_flow():
    """Fetch general unusual options flow"""
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
        
        data = response.json()
        trades = data.get('data', [])
        
        result = []
        for trade in trades:
            option_chain = trade.get('option_chain', '')
            ticker, expiry, dte, opt_type, strike = parse_option_chain(option_chain)
            
            if not ticker or ticker in config.EXCLUDE_TICKERS:
                continue
                
            premium = float(trade.get('total_premium', 0))
            volume = trade.get('volume', 0)
            
            if premium < config.MIN_PREMIUM or volume < config.MIN_VOLUME:
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
                'rule': trade.get('rule_name', 'N/A'),
                'side': trade.get('side', 'N/A')
            })
            
        return result
        
    except Exception as e:
        st.error(f"Error fetching general flow: {e}")
        return []

def fetch_etf_flow():
    """Fetch ETF-specific flow (SPY, QQQ, IWM)"""
    params = {
        'limit': 500
    }
    
    try:
        response = httpx.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return []
            
        data = response.json()
        trades = data.get('data', [])
        
        result = []
        for trade in trades:
            option_chain = trade.get('option_chain', '')
            ticker, expiry, dte, opt_type, strike = parse_option_chain(option_chain)
            
            if not ticker or ticker not in config.ALLOWED_TICKERS or dte > 7:
                continue
                
            result.append({
                'ticker': ticker,
                'option': option_chain,
                'type': opt_type,
                'strike': strike,
                'expiry': expiry,
                'dte': dte,
                'side': trade.get('side', 'N/A'),
                'price': trade.get('price', 'N/A'),
                'premium': trade.get('total_premium', 'N/A'),
                'volume': trade.get('volume', 'N/A'),
                'oi': trade.get('open_interest', 'N/A'),
                'time': trade.get('created_at', 'N/A')
            })
            
        return result
        
    except Exception as e:
        st.error(f"Error fetching ETF flow: {e}")
        return []

def fetch_dte_segregated_flow():
    """Fetch flow segregated by DTE <7 and >=7"""
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
            
        data = response.json()
        trades = data.get('data', [])
        
        result = []
        for trade in trades:
            option_chain = trade.get('option_chain', '')
            ticker, expiry, dte, opt_type, strike = parse_option_chain(option_chain)
            
            if not ticker or ticker in config.EXCLUDE_TICKERS:
                continue
                
            premium = float(trade.get('total_premium', 0))
            if premium < config.MIN_PREMIUM:
                continue
                
            result.append({
                'ticker': ticker,
                'option': option_chain,
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
        
    except Exception as e:
        st.error(f"Error fetching DTE segregated flow: {e}")
        return []

# --- ANALYSIS FUNCTIONS ---
def analyze_flow_by_ticker(trades):
    """Analyze flow grouped by ticker"""
    ticker_analysis = defaultdict(lambda: {
        'call_premium': 0, 'put_premium': 0, 'total_volume': 0,
        'trades': [], 'avg_dte': 0, 'sentiment': 'Neutral'
    })
    
    for trade in trades:
        ticker = trade['ticker']
        ticker_analysis[ticker]['trades'].append(trade)
        ticker_analysis[ticker]['total_volume'] += trade.get('volume', 0)
        
        if trade['type'] == 'C':
            ticker_analysis[ticker]['call_premium'] += trade.get('premium', 0)
        else:
            ticker_analysis[ticker]['put_premium'] += trade.get('premium', 0)
    
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
        
        if data['trades']:
            data['avg_dte'] = sum(t.get('dte', 0) for t in data['trades']) / len(data['trades'])
    
    return dict(ticker_analysis)

def generate_alerts(trades):
    """Generate high-priority alerts based on trade characteristics"""
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

# --- DISPLAY FUNCTIONS ---
def display_general_summary(trades):
    """Display general market summary"""
    if not trades:
        st.warning("No trades to display")
        return
        
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
    
    # Top tickers
    ticker_data = analyze_flow_by_ticker(trades)
    top_tickers = sorted(ticker_data.items(),
                        key=lambda x: x[1]['call_premium'] + x[1]['put_premium'],
                        reverse=True)[:10]
    
    st.markdown("#### 🏆 Top Tickers by Premium")
    ticker_df = []
    for ticker, data in top_tickers:
        total_prem = data['call_premium'] + data['put_premium']
        ticker_df.append({
            'Ticker': ticker,
            'Premium': f"${total_prem:,.0f}",
            'Sentiment': data['sentiment'],
            'Trades': len(data['trades']),
            'Avg DTE': f"{data['avg_dte']:.1f}"
        })
    
    if ticker_df:
        st.dataframe(pd.DataFrame(ticker_df), use_container_width=True)

def display_etf_flow(trades):
    """Display ETF-specific flow"""
    if not trades:
        st.warning("No ETF trades found")
        return
        
    st.markdown("### 📈 ETF Flow (SPY, QQQ, IWM - DTE ≤ 7)")
    
    # Create DataFrame
    df_data = []
    for trade in trades:
        df_data.append({
            'Ticker': trade['ticker'],
            'Type': trade['type'],
            'Strike': trade['strike'],
            'Expiry': trade['expiry'],
            'DTE': trade['dte'],
            'Side': trade['side'],
            'Price': trade['price'],
            'Premium': f"${trade['premium']:,}" if isinstance(trade['premium'], (int, float)) else trade['premium'],
            'Volume': trade['volume'],
            'OI': trade['oi'],
            'Time': trade['time']
        })
    
    if df_data:
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)

def display_dte_segregated(trades):
    """Display trades segregated by DTE"""
    if not trades:
        st.warning("No trades to display")
        return
        
    # Segregate trades
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
                'Time': trade['time']
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
    """Display high-priority alerts"""
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
                        f"{alert.get('moneyness', 'N/A')}")
                st.write(f"📍 Reasons: {', '.join(alert.get('reasons', []))}")
            
            with col2:
                st.metric("Alert Score", alert.get('alert_score', 0))
            
            st.divider()

def save_to_csv(trades, filename_prefix):
    """Save trades to CSV"""
    if not trades:
        st.warning("No data to save")
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    
    # Flatten the data for CSV
    csv_data = []
    for trade in trades:
        row = trade.copy()
        if isinstance(row.get('reasons'), list):
            row['reasons'] = ', '.join(row['reasons'])
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

# --- STREAMLIT UI ---
st.set_page_config(
    page_title="Comprehensive Options Flow Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Comprehensive Options Flow Tracker")
st.markdown("### Real-time unusual options activity analysis across multiple strategies")

# Sidebar for controls
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    
    scan_type = st.selectbox(
        "Select Scan Type:",
        [
            "🔍 General Flow Scanner",
            "📈 ETF Flow (SPY/QQQ/IWM)",
            "⏰ DTE Segregated Flow",
            "🚨 Alert Generator"
        ]
    )
    
    st.markdown("---")
    
    if st.button("🔄 Run Scan", use_container_width=True, type="primary"):
        st.session_state.run_scan = True
        st.session_state.scan_type = scan_type

# Main content area
if hasattr(st.session_state, 'run_scan') and st.session_state.run_scan:
    with st.spinner(f"Running {st.session_state.scan_type}..."):
        
        if st.session_state.scan_type == "🔍 General Flow Scanner":
            trades = fetch_general_flow()
            if trades:
                display_general_summary(trades)
                with st.expander("💾 Export Data", expanded=False):
                    save_to_csv(trades, "general_flow")
                    
        elif st.session_state.scan_type == "📈 ETF Flow (SPY/QQQ/IWM)":
            trades = fetch_etf_flow()
            if trades:
                display_etf_flow(trades)
                with st.expander("💾 Export Data", expanded=False):
                    save_to_csv(trades, "etf_flow")
                    
        elif st.session_state.scan_type == "⏰ DTE Segregated Flow":
            trades = fetch_dte_segregated_flow()
            if trades:
                display_dte_segregated(trades)
                with st.expander("💾 Export Data", expanded=False):
                    save_to_csv(trades, "dte_segregated_flow")
                    
        elif st.session_state.scan_type == "🚨 Alert Generator":
            trades = fetch_general_flow()
            if trades:
                display_alerts(trades)
                with st.expander("💾 Export Data", expanded=False):
                    save_to_csv(trades, "alerts")
    
    st.session_state.run_scan = False

else:
    # Welcome screen
    st.markdown("""
    ## Welcome to the Comprehensive Options Flow Tracker! 👋
    
    This application combines multiple options flow analysis strategies:
    
    ### 🔍 **General Flow Scanner**
    - Comprehensive unusual options activity analysis
    - Market sentiment tracking
    - Top ticker identification by premium flow
    
    ### 📈 **ETF Flow (SPY/QQQ/IWM)**
    - Focused on major market ETFs
    - Short-term expiration analysis (≤7 DTE)
    - Perfect for day trading setups
    
    ### ⏰ **DTE Segregated Flow**
    - Separates calls and puts by time to expiration
    - < 7 DTE vs ≥ 7 DTE analysis
    - Helps identify short-term vs longer-term positioning
    
    ### 🚨 **Alert Generator**
    - High-priority trade alerts based on multiple criteria
    - Scoring system for trade significance
    - Focus on the most important opportunities
    
    **Select a scan type from the sidebar and click 'Run Scan' to get started!**
    """)

st.markdown("---")
st.caption("🚀 Powered by Unusual Whales API | Built with Streamlit")
