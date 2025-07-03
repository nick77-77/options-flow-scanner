import streamlit as st
import httpx
from datetime import datetime, date
from collections import defaultdict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from zoneinfo import ZoneInfo  # Python 3.9+
import numpy as np

# --- CONFIGURATION ---
class Config:
    UW_TOKEN = st.secrets.get("UW_TOKEN", "e6e8601a-0746-4cec-a07d-c3eabfc13926")
    EXCLUDE_TICKERS = {'TSLA', 'MSTR', 'CRCL'}
    ALLOWED_TICKERS = {'QQQ', 'SPY', 'IWM'}
    MIN_PREMIUM = 50000  # Lowered to catch more activity
    LIMIT = 1000  # Increased for better analysis
    SCENARIO_OTM_CALL_MIN_PREMIUM = 100000
    SCENARIO_ITM_CONV_MIN_PREMIUM = 50000
    SCENARIO_SWEEP_VOLUME_OI_RATIO = 2
    SCENARIO_BLOCK_TRADE_VOL = 100
    DARK_POOL_THRESHOLD = 0.3  # 30% of average volume
    GAMMA_SQUEEZE_THRESHOLD = 5  # Volume/OI ratio for gamma events

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

def detect_order_side(trade):
    """Enhanced order side detection"""
    try:
        # Safely convert values to float, default to 0 if conversion fails
        price = float(trade.get('price', 0)) if trade.get('price') not in ['N/A', '', None] else 0
        mid_price = float(trade.get('mid_price', 0)) if trade.get('mid_price') not in ['N/A', '', None] else 0
        volume = int(trade.get('volume', 0)) if trade.get('volume') not in ['N/A', '', None] else 0
        oi = int(trade.get('open_interest', 1)) if trade.get('open_interest') not in ['N/A', '', None] else 1
        
        # Check for selling indicators
        ask_side_indicators = [
            price >= mid_price if mid_price > 0 else False,
            'ask' in trade.get('description', '').lower(),
            trade.get('side', '').upper() == 'ASK'
        ]
        
        bid_side_indicators = [
            price <= mid_price if mid_price > 0 else False,
            'bid' in trade.get('description', '').lower(),
            trade.get('side', '').upper() == 'BID'
        ]
        
        # Analyze order flow patterns
        if sum(ask_side_indicators) > sum(bid_side_indicators):
            return "SELL_TO_OPEN" if volume > oi else "SELL_TO_CLOSE"
        elif sum(bid_side_indicators) > sum(ask_side_indicators):
            return "BUY_TO_OPEN" if volume > oi else "BUY_TO_CLOSE"
        else:
            # Fallback to volume/OI analysis
            vol_oi_ratio = volume / max(oi, 1)
            return "BUY_TO_OPEN" if vol_oi_ratio > 1.5 else "MIXED"
    except (ValueError, TypeError, AttributeError):
        return "UNKNOWN"

def detect_advanced_scenarios(trade, underlying_price=None, market_context=None):
    """Enhanced scenario detection with selling patterns"""
    scenarios = []
    opt_type = trade['type']
    strike = trade['strike']
    premium = trade['premium']
    volume = trade.get('volume', 0)
    oi = trade.get('oi', 0)
    rule_name = trade.get('rule_name', '')
    ticker = trade['ticker']
    order_side = trade.get('order_side', 'UNKNOWN')
    
    if underlying_price is None:
        underlying_price = strike

    # Calculate moneyness
    if underlying_price > 0:
        moneyness_pct = ((strike - underlying_price) / underlying_price) * 100
        if opt_type == 'C':
            if moneyness_pct > 5:
                moneyness = "OTM"
            elif moneyness_pct < -5:
                moneyness = "ITM"
            else:
                moneyness = "ATM"
        else:  # Put
            if moneyness_pct < -5:
                moneyness = "OTM"
            elif moneyness_pct > 5:
                moneyness = "ITM"
            else:
                moneyness = "ATM"
    else:
        moneyness = "UNKNOWN"

    vol_oi_ratio = volume / max(oi, 1)
    
    # Original scenarios with buy/sell context
    if opt_type == 'C' and moneyness == 'OTM' and premium >= config.SCENARIO_OTM_CALL_MIN_PREMIUM:
        action = "Selling" if "SELL" in order_side else "Buying"
        scenarios.append(f"Large OTM Call {action}")
    
    if opt_type == 'P' and moneyness == 'OTM' and premium >= config.SCENARIO_OTM_CALL_MIN_PREMIUM:
        action = "Selling" if "SELL" in order_side else "Buying"
        scenarios.append(f"Large OTM Put {action}")
    
    # Enhanced selling scenarios
    if "SELL" in order_side:
        if opt_type == 'C' and moneyness == 'OTM':
            scenarios.append("Call Overwriting Strategy")
        if opt_type == 'P' and moneyness == 'OTM':
            scenarios.append("Cash-Secured Put Strategy")
        if premium > 200000:
            scenarios.append("Large Premium Collection")
    
    # Gamma squeeze detection
    if vol_oi_ratio > config.GAMMA_SQUEEZE_THRESHOLD and moneyness == "ATM":
        scenarios.append("Potential Gamma Squeeze")
    
    # Dark pool activity
    if volume > 1000 and vol_oi_ratio < 0.5:
        scenarios.append("Dark Pool Activity")
    
    # Institutional patterns
    if volume >= 500 and premium > 500000:
        scenarios.append("Institutional Flow")
    
    # Earnings/Event plays
    if trade.get('dte', 0) <= 7 and premium > 100000:
        scenarios.append("Event/Earnings Play")
    
    # Volatility plays
    if opt_type == 'C' and opt_type == 'P' and ticker in market_context.get('high_iv_tickers', []):
        scenarios.append("Volatility Strategy")
    
    # Hedging patterns
    if ticker in ['SPY', 'QQQ', 'IWM'] and opt_type == 'P' and premium > 300000:
        scenarios.append("Portfolio Hedging")
    
    # Iron Condor/Spread detection
    if rule_name in ['RepeatedHits', 'RepeatedHitsAscendingFill']:
        scenarios.append("Multi-Leg Strategy")
    
    return scenarios if scenarios else ["Normal Flow"]

def calculate_flow_metrics(trades):
    """Calculate comprehensive flow metrics"""
    if not trades:
        return {}
    
    call_trades = [t for t in trades if t['type'] == 'C']
    put_trades = [t for t in trades if t['type'] == 'P']
    
    call_premium_bought = sum(t['premium'] for t in call_trades if 'BUY' in t.get('order_side', ''))
    call_premium_sold = sum(t['premium'] for t in call_trades if 'SELL' in t.get('order_side', ''))
    put_premium_bought = sum(t['premium'] for t in put_trades if 'BUY' in t.get('order_side', ''))
    put_premium_sold = sum(t['premium'] for t in put_trades if 'SELL' in t.get('order_side', ''))
    
    total_premium = call_premium_bought + call_premium_sold + put_premium_bought + put_premium_sold
    
    # Net flow calculations
    net_call_flow = call_premium_bought - call_premium_sold
    net_put_flow = put_premium_bought - put_premium_sold
    net_total_flow = net_call_flow + net_put_flow
    
    # Put/Call ratios
    put_call_ratio_volume = len(put_trades) / max(len(call_trades), 1)
    put_call_ratio_premium = (put_premium_bought + put_premium_sold) / max(call_premium_bought + call_premium_sold, 1)
    
    return {
        'total_premium': total_premium,
        'net_call_flow': net_call_flow,
        'net_put_flow': net_put_flow,
        'net_total_flow': net_total_flow,
        'put_call_ratio_volume': put_call_ratio_volume,
        'put_call_ratio_premium': put_call_ratio_premium,
        'call_premium_bought': call_premium_bought,
        'call_premium_sold': call_premium_sold,
        'put_premium_bought': put_premium_bought,
        'put_premium_sold': put_premium_sold,
        'bullish_flow': call_premium_bought + put_premium_sold,
        'bearish_flow': put_premium_bought + call_premium_sold
    }

def generate_enhanced_alerts(trades):
    """Enhanced alert system with selling detection"""
    alerts = []
    for trade in trades:
        score = 0
        reasons = []
        alert_type = "INFO"

        premium = trade.get('premium', 0)
        volume = trade.get('volume', 0)
        oi = trade.get('oi', 0)
        order_side = trade.get('order_side', '')
        
        # Premium scoring
        if premium > 1000000:
            score += 5
            reasons.append("Massive Premium (>$1M)")
            alert_type = "CRITICAL"
        elif premium > 500000:
            score += 3
            reasons.append("Large Premium (>$500K)")
            alert_type = "HIGH"
        elif premium > 250000:
            score += 2
            reasons.append("Notable Premium (>$250K)")

        # Volume/OI analysis
        vol_oi_ratio = volume / max(oi, 1)
        if vol_oi_ratio > 5:
            score += 3
            reasons.append("Extremely High Vol/OI")
        elif vol_oi_ratio > 2:
            score += 2
            reasons.append("High Vol/OI")

        # Selling activity alerts
        if "SELL" in order_side and premium > 200000:
            score += 2
            reasons.append("Large Premium Collection")
            alert_type = "SELL_ALERT"

        # Unusual patterns
        if len(trade.get('scenarios', [])) > 2:
            score += 1
            reasons.append("Multiple Patterns")

        # Time decay plays
        if trade.get('dte', 0) <= 3 and premium > 100000:
            score += 2
            reasons.append("Short-Term High Premium")

        if score >= 4:
            trade['alert_score'] = score
            trade['reasons'] = reasons
            trade['alert_type'] = alert_type
            alerts.append(trade)

    return sorted(alerts, key=lambda x: -x.get('alert_score', 0))

# --- FETCH FUNCTION ---
def fetch_enhanced_flow():
    """Enhanced data fetching with better analysis"""
    params = {
        'issue_types[]': ['Common Stock', 'ADR'],
        'min_dte': 0,  # Include 0DTE
        'min_volume_oi_ratio': 0.1,  # Much lower threshold to catch more activity
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
        ticker_context = defaultdict(list)

        for trade in data:
            option_chain = trade.get('option_chain', '')
            ticker, expiry, dte, opt_type, strike = parse_option_chain(option_chain)

            if not ticker or ticker in config.EXCLUDE_TICKERS:
                continue

            premium = float(trade.get('total_premium', 0))
            # Removed minimum premium filter here - let UI filters handle it
            # if premium < config.MIN_PREMIUM:
            #     continue

            # Enhanced time parsing
            utc_time_str = trade.get('created_at', 'N/A')
            ny_time_str = "N/A"
            if utc_time_str and utc_time_str != "N/A":
                try:
                    utc_time = datetime.fromisoformat(utc_time_str.replace("Z", "+00:00"))
                    ny_time = utc_time.astimezone(ZoneInfo("America/New_York"))
                    ny_time_str = ny_time.strftime("%I:%M:%S %p")
                except Exception:
                    ny_time_str = "N/A"

            # Safe numeric conversions
            try:
                premium = float(trade.get('total_premium', 0))
                volume = int(trade.get('volume', 0))
                oi = int(trade.get('open_interest', 0))
                underlying_price = float(trade.get('underlying_price', strike)) if trade.get('underlying_price') not in ['N/A', '', None] else strike
                price = float(trade.get('price', 0)) if trade.get('price') not in ['N/A', '', None] else 0
                mid_price = float(trade.get('mid_price', 0)) if trade.get('mid_price') not in ['N/A', '', None] else 0
                bid = float(trade.get('bid', 0)) if trade.get('bid') not in ['N/A', '', None] else 0
                ask = float(trade.get('ask', 0)) if trade.get('ask') not in ['N/A', '', None] else 0
            except (ValueError, TypeError):
                continue  # Skip trades with invalid numeric data

            trade_data = {
                'ticker': ticker,
                'option': option_chain,
                'type': opt_type,
                'strike': strike,
                'expiry': expiry,
                'dte': dte,
                'price': price,
                'premium': premium,
                'volume': volume,
                'oi': oi,
                'time_utc': utc_time_str,
                'time_ny': ny_time_str,
                'rule_name': trade.get('rule_name', ''),
                'description': trade.get('description', ''),
                'underlying_price': underlying_price,
                'mid_price': mid_price,
                'bid': bid,
                'ask': ask,
                'vol_oi_ratio': volume / max(oi, 1),
                'side': trade.get('side', ''),
            }
            
            # Detect order side
            trade_data['order_side'] = detect_order_side(trade_data)
            ticker_context[ticker].append(trade_data)

        # Second pass: add scenarios with context
        market_context = {'high_iv_tickers': list(ticker_context.keys())}
        
        for ticker, trade_list in ticker_context.items():
            for trade in trade_list:
                scenarios = detect_advanced_scenarios(trade, trade['underlying_price'], market_context)
                trade['scenarios'] = scenarios
                
                # Enhanced moneyness calculation
                if trade['underlying_price'] and trade['underlying_price'] > 0:
                    try:
                        underlying = float(trade['underlying_price'])
                        strike = float(trade['strike'])
                        pct_diff = ((strike - underlying) / underlying) * 100
                        
                        if abs(pct_diff) < 2:
                            trade['moneyness'] = "ATM"
                        elif pct_diff > 0:
                            trade['moneyness'] = f"OTM +{pct_diff:.1f}%"
                        else:
                            trade['moneyness'] = f"ITM {pct_diff:.1f}%"
                    except (ValueError, TypeError, ZeroDivisionError):
                        trade['moneyness'] = "Unknown"
                else:
                    trade['moneyness'] = "Unknown"
                
                result.append(trade)

        return result

    except Exception as e:
        st.error(f"Error fetching enhanced flow: {e}")
        return []

# --- ENHANCED VISUALIZATIONS ---
def create_enhanced_dashboard(trades):
    """Create comprehensive dashboard"""
    if not trades:
        return
    
    metrics = calculate_flow_metrics(trades)
    df = pd.DataFrame(trades)
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        net_flow = metrics.get('net_total_flow', 0)
        flow_direction = "🟢 Bullish" if net_flow > 0 else "🔴 Bearish" if net_flow < 0 else "⚪ Neutral"
        st.metric("Net Flow Direction", flow_direction, f"${net_flow:,.0f}")
    
    with col2:
        pcr = metrics.get('put_call_ratio_premium', 0)
        st.metric("Put/Call Ratio", f"{pcr:.2f}", "Premium Based")
    
    with col3:
        total_premium = metrics.get('total_premium', 0)
        st.metric("Total Premium", f"${total_premium:,.0f}")
    
    with col4:
        bullish_flow = metrics.get('bullish_flow', 0)
        bearish_flow = metrics.get('bearish_flow', 0)
        sentiment = "Bullish" if bullish_flow > bearish_flow else "Bearish"
        st.metric("Market Sentiment", sentiment, f"{abs(bullish_flow - bearish_flow):,.0f}")

    # Enhanced charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Buy vs Sell flow
        buy_trades = df[df['order_side'].str.contains('BUY', na=False)]
        sell_trades = df[df['order_side'].str.contains('SELL', na=False)]
        
        flow_data = pd.DataFrame({
            'Type': ['Calls Bought', 'Calls Sold', 'Puts Bought', 'Puts Sold'],
            'Premium': [
                metrics.get('call_premium_bought', 0),
                metrics.get('call_premium_sold', 0),
                metrics.get('put_premium_bought', 0),
                metrics.get('put_premium_sold', 0)
            ],
            'Color': ['#00ff00', '#ff6b6b', '#ff0000', '#6b6bff']
        })
        
        fig = px.bar(flow_data, x='Type', y='Premium', color='Type',
                     title="Buy vs Sell Flow Breakdown",
                     color_discrete_sequence=flow_data['Color'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Scenario distribution
        scenario_counts = {}
        for trade in trades:
            for scenario in trade.get('scenarios', []):
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        
        if scenario_counts:
            scenario_df = pd.DataFrame(list(scenario_counts.items()), 
                                     columns=['Scenario', 'Count'])
            fig = px.pie(scenario_df, values='Count', names='Scenario',
                        title="Strategy Distribution")
            st.plotly_chart(fig, use_container_width=True)

def display_selling_analysis(trades):
    """Dedicated selling analysis section"""
    st.markdown("### 💰 Options Selling Analysis")
    
    sell_trades = [t for t in trades if 'SELL' in t.get('order_side', '')]
    buy_trades = [t for t in trades if 'BUY' in t.get('order_side', '')]
    
    if not sell_trades:
        st.info("No selling activity detected in current dataset")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sell_premium = sum(t['premium'] for t in sell_trades)
        st.metric("Total Premium Collected", f"${sell_premium:,.0f}")
    
    with col2:
        avg_sell_premium = sell_premium / len(sell_trades) if sell_trades else 0
        st.metric("Avg Premium/Trade", f"${avg_sell_premium:,.0f}")
    
    with col3:
        sell_ratio = len(sell_trades) / len(trades) if trades else 0
        st.metric("Selling Activity %", f"{sell_ratio:.1%}")
    
    # Top selling trades
    st.markdown("#### 🔥 Largest Premium Collection Trades")
    sell_df_data = []
    for trade in sorted(sell_trades, key=lambda x: x['premium'], reverse=True)[:10]:
        sell_df_data.append({
            'Ticker': trade['ticker'],
            'Type': trade['type'],
            'Strike': trade['strike'],
            'Expiry': trade['expiry'],
            'DTE': trade['dte'],
            'Premium': f"${trade['premium']:,}",
            'Strategy': ", ".join(trade['scenarios'][:2]),
            'Moneyness': trade.get('moneyness', 'Unknown'),
            'Time': trade['time_ny']
        })
    
    if sell_df_data:
        st.dataframe(pd.DataFrame(sell_df_data), use_container_width=True)

# --- MAIN STREAMLIT APP ---
st.set_page_config(page_title="Enhanced Options Flow Tracker", page_icon="🚀", layout="wide")
st.title("🚀 Enhanced Options Flow Tracker")
st.markdown("### Advanced real-time options activity with buy/sell detection and institutional pattern recognition")

with st.sidebar:
    st.markdown("## 🎛️ Enhanced Control Panel")
    
    scan_type = st.selectbox(
        "Select Analysis Type:",
        [
            "📊 Comprehensive Dashboard",
            "💰 Selling Activity Analysis", 
            "⚡ Real-time Alerts",
            "🎯 DTE Strategy Analysis",
            "🏛️ Institutional Flow",
            "📈 Gamma Squeeze Scanner"
        ]
    )
    
    # Filters
    st.markdown("### Premium Range Filters")
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
        index=1  # Default to "Under $100K"
    )
    
    st.markdown("### Activity Filters")
    include_selling = st.checkbox("Include Selling Activity", value=True)
    dte_filter = st.selectbox(
        "Time to Expiry:",
        ["All DTE", "0DTE Only", "Weekly (≤7d)", "Monthly (≤30d)", "Quarterly (≤90d)"],
        index=0
    )
    
    run_scan = st.button("🔄 Run Enhanced Scan", type="primary", use_container_width=True)
    
    if st.button("📱 Quick 0DTE Scan", use_container_width=True):
        premium_range = "All Premiums (No Filter)"
        dte_filter = "0DTE Only"
        run_scan = True

# Main execution
if run_scan:
    with st.spinner(f"Running {scan_type}..."):
        trades = fetch_enhanced_flow()
        
        # Apply premium range filters
        def apply_premium_filter(premium, range_selection):
            if range_selection == "All Premiums (No Filter)":
                return True
            elif range_selection == "Under $100K":
                return premium < 100000
            elif range_selection == "Under $250K":
                return premium < 250000
            elif range_selection == "$100K - $250K":
                return 100000 <= premium < 250000
            elif range_selection == "$250K - $500K":
                return 250000 <= premium < 500000
            elif range_selection == "Above $250K":
                return premium >= 250000
            elif range_selection == "Above $500K":
                return premium >= 500000
            elif range_selection == "Above $1M":
                return premium >= 1000000
            return True
        
        # Apply DTE filters
        def apply_dte_filter(dte, dte_selection):
            if dte_selection == "All DTE":
                return True
            elif dte_selection == "0DTE Only":
                return dte == 0
            elif dte_selection == "Weekly (≤7d)":
                return dte <= 7
            elif dte_selection == "Monthly (≤30d)":
                return dte <= 30
            elif dte_selection == "Quarterly (≤90d)":
                return dte <= 90
            return True
        
        # Apply filters with safe comparisons
        filtered_trades = []
        for trade in trades:
            try:
                # Safe numeric conversions for filtering
                trade_premium = float(trade.get('premium', 0))
                trade_dte = int(trade.get('dte', 0))
                trade_order_side = str(trade.get('order_side', ''))
                
                # Apply filters
                if (apply_premium_filter(trade_premium, premium_range) and 
                    apply_dte_filter(trade_dte, dte_filter)):
                    if include_selling or 'BUY' in trade_order_side:
                        filtered_trades.append(trade)
            except (ValueError, TypeError):
                continue  # Skip trades with invalid data
        
        st.success(f"Found {len(filtered_trades)} trades matching criteria (Premium: {premium_range}, DTE: {dte_filter})")
        
        if "Comprehensive" in scan_type:
            create_enhanced_dashboard(filtered_trades)
            
            # Additional detailed tables
            st.markdown("### 📋 Detailed Trade Analysis")
            
            tabs = st.tabs(["🟢 Call Activity", "🔴 Put Activity", "💎 High Premium", "⚡ High Volume"])
            
            with tabs[0]:
                call_trades = [t for t in filtered_trades if t['type'] == 'C']
                if call_trades:
                    call_df = pd.DataFrame([{
                        'Ticker': t['ticker'], 'Strike': t['strike'], 'Expiry': t['expiry'],
                        'DTE': t['dte'], 'Premium': f"${t['premium']:,}", 'Side': t.get('order_side', 'Unknown'),
                        'Scenarios': ", ".join(t['scenarios'][:2]), 'Time': t['time_ny']
                    } for t in sorted(call_trades, key=lambda x: x['premium'], reverse=True)[:20]])
                    st.dataframe(call_df, use_container_width=True)
            
            with tabs[1]:
                put_trades = [t for t in filtered_trades if t['type'] == 'P']
                if put_trades:
                    put_df = pd.DataFrame([{
                        'Ticker': t['ticker'], 'Strike': t['strike'], 'Expiry': t['expiry'],
                        'DTE': t['dte'], 'Premium': f"${t['premium']:,}", 'Side': t.get('order_side', 'Unknown'),
                        'Scenarios': ", ".join(t['scenarios'][:2]), 'Time': t['time_ny']
                    } for t in sorted(put_trades, key=lambda x: x['premium'], reverse=True)[:20]])
                    st.dataframe(put_df, use_container_width=True)
            
            with tabs[2]:
                high_premium = sorted(filtered_trades, key=lambda x: x['premium'], reverse=True)[:15]
                if high_premium:
                    hp_df = pd.DataFrame([{
                        'Ticker': t['ticker'], 'Type': t['type'], 'Strike': t['strike'],
                        'Premium': f"${t['premium']:,}", 'Side': t.get('order_side', 'Unknown'),
                        'Strategy': ", ".join(t['scenarios'][:2])
                    } for t in high_premium])
                    st.dataframe(hp_df, use_container_width=True)
            
            with tabs[3]:
                high_volume = sorted(filtered_trades, key=lambda x: x['volume'], reverse=True)[:15]
                if high_volume:
                    hv_df = pd.DataFrame([{
                        'Ticker': t['ticker'], 'Type': t['type'], 'Volume': t['volume'],
                        'Premium': f"${t['premium']:,}", 'Vol/OI': f"{t['vol_oi_ratio']:.1f}",
                        'Strategy': ", ".join(t['scenarios'][:2])
                    } for t in high_volume])
                    st.dataframe(hv_df, use_container_width=True)
        
        elif "Selling" in scan_type:
            display_selling_analysis(filtered_trades)
        
        elif "Alerts" in scan_type:
            alerts = generate_enhanced_alerts(filtered_trades)
            if alerts:
                st.markdown("### 🚨 Enhanced Alert System")
                for i, alert in enumerate(alerts[:15], 1):
                    alert_type = alert.get('alert_type', 'INFO')
                    icon = "🔥" if alert_type == "CRITICAL" else "⚠️" if alert_type == "HIGH" else "💰" if alert_type == "SELL_ALERT" else "ℹ️"
                    
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{icon} {i}. {alert['ticker']} ${alert['strike']:.0f}{alert['type']} "
                                      f"{alert['expiry']} ({alert['dte']}d)**")
                            st.write(f"💰 Premium: ${alert['premium']:,.0f} | Side: {alert.get('order_side', 'Unknown')} | "
                                   f"Vol: {alert['volume']} | {alert.get('moneyness', 'N/A')}")
                            st.write(f"🎯 Strategies: {', '.join(alert.get('scenarios', [])[:3])}")
                            st.write(f"📍 Alert Reasons: {', '.join(alert.get('reasons', []))}")
                        with col2:
                            st.metric("Alert Score", alert.get('alert_score', 0))
                            st.write(f"**{alert_type}**")
                        st.divider()
            else:
                st.info("No alerts triggered with current criteria")
        
        # Export functionality
        with st.expander("💾 Export Enhanced Data", expanded=False):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if filtered_trades:
                # Enhanced CSV export
                csv_data = []
                for trade in filtered_trades:
                    row = trade.copy()
                    row['scenarios'] = ', '.join(row.get('scenarios', []))
                    row['reasons'] = ', '.join(row.get('reasons', [])) if 'reasons' in row else ''
                    csv_data.append(row)
                
                df = pd.DataFrame(csv_data)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label=f"📥 Download Enhanced Flow Data ({len(filtered_trades)} trades)",
                    data=csv,
                    file_name=f"enhanced_options_flow_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No data to export")

else:
    st.markdown("""
    ## Welcome to the Enhanced Options Flow Tracker! 🚀
    
    ### New Features:
    - **🔍 Buy/Sell Detection**: Identify whether traders are buying or selling options
    - **💰 Premium Collection Analysis**: Track large option selling strategies  
    - **⚡ Enhanced Alerts**: Multi-tier alert system with critical/high/sell alerts
    - **🎯 Advanced Scenarios**: Detect gamma squeezes, dark pool activity, institutional flow
    - **📊 Comprehensive Dashboard**: Net flow analysis, put/call ratios, sentiment tracking
    - **🏛️ Institutional Patterns**: Identify large block trades and sophisticated strategies
    
    ### Strategy Detection:
    - Call/Put Overwriting
    - Cash-Secured Puts  
    - Iron Condors & Spreads
    - Gamma Squeeze Setups
    - Portfolio Hedging
    - Event/Earnings Plays
    - Volatility Strategies
    
    **Select an analysis type and click Run Enhanced Scan to begin!**
    """)
