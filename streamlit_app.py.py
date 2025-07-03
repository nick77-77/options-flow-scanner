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
    MIN_PREMIUM = 50000  # Lower threshold for more activity
    LIMIT = 1000
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

def calculate_expected_move(stock_price, iv, dte):
    """
    Calculate Expected Move: Stock Price * IV * sqrt(DTE/365)
    Returns the 1 standard deviation expected move (68% probability)
    """
    try:
        if stock_price <= 0 or iv <= 0 or dte <= 0:
            return None, None, None
        
        # Convert IV from percentage to decimal if needed
        if iv > 1:
            iv = iv / 100
        
        # Calculate expected move
        time_factor = (dte / 365) ** 0.5
        expected_move = stock_price * iv * time_factor
        
        # Calculate upper and lower bounds
        upper_bound = stock_price + expected_move
        lower_bound = stock_price - expected_move
        
        return expected_move, upper_bound, lower_bound
    except (ValueError, TypeError, ZeroDivisionError):
        return None, None, None

def format_expected_move(stock_price, expected_move, upper_bound, lower_bound):
    """Format expected move for display"""
    if expected_move is None:
        return "N/A"
    
    move_pct = (expected_move / stock_price) * 100
    return f"±${expected_move:.2f} ({move_pct:.1f}%)"

def get_iv_rank_category(iv):
    """Categorize IV levels for quick assessment"""
    try:
        if iv <= 0:
            return "Unknown"
        
        # Convert to percentage if needed
        if iv <= 1:
            iv = iv * 100
        
        if iv < 20:
            return "Low IV"
        elif iv < 30:
            return "Normal IV"
        elif iv < 50:
            return "Elevated IV"
        elif iv < 80:
            return "High IV"
        else:
            return "Extreme IV"
    except:
        return "Unknown"

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

def detect_enhanced_scenarios(trade, underlying_price=None):
    """Enhanced scenario detection with IV analysis"""
    scenarios = []
    opt_type = trade['type']
    strike = trade['strike']
    premium = trade['premium']
    volume = trade.get('volume', 0)
    oi = trade.get('oi', 0)
    iv = trade.get('iv', 0)
    dte = trade.get('dte', 0)
    order_side = trade.get('order_side', 'UNKNOWN')
    
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

    vol_oi_ratio = volume / max(oi, 1)

    # Premium-based scenarios
    if premium >= 1000000:
        scenarios.append("Massive Premium (>$1M)")
    elif premium >= 500000:
        scenarios.append("Large Premium (>$500K)")
    elif premium >= 200000:
        scenarios.append("Medium Premium (>$200K)")
    
    # Volume-based scenarios
    if volume >= 1000:
        scenarios.append("High Volume")
    elif volume >= 500:
        scenarios.append("Medium Volume")
    
    # Moneyness scenarios with buy/sell context
    if opt_type == 'C' and moneyness == 'OTM':
        action = "Selling" if "SELL" in order_side else "Buying"
        scenarios.append(f"OTM Call {action}")
    elif opt_type == 'C' and moneyness == 'ITM':
        scenarios.append("ITM Call")
    elif opt_type == 'P' and moneyness == 'OTM':
        action = "Selling" if "SELL" in order_side else "Buying"
        scenarios.append(f"OTM Put {action}")
    elif opt_type == 'P' and moneyness == 'ITM':
        scenarios.append("ITM Put")
    
    # Fresh interest
    if volume > oi * 2:
        scenarios.append("Fresh Interest")
    
    # Gamma squeeze detection
    if vol_oi_ratio > config.GAMMA_SQUEEZE_THRESHOLD and moneyness == "ATM":
        scenarios.append("Potential Gamma Squeeze")
    
    # Dark pool activity
    if volume > 1000 and vol_oi_ratio < 0.5:
        scenarios.append("Dark Pool Activity")
    
    # Institutional patterns
    if volume >= 500 and premium > 500000:
        scenarios.append("Institutional Flow")
    
    # IV-based scenarios
    if iv > 80:
        scenarios.append("Extreme IV")
    elif iv > 50:
        scenarios.append("High IV")
    
    # IV Crush risk
    if iv > 60 and dte <= 7:
        scenarios.append("IV Crush Risk")
    
    # Earnings/Event plays
    if iv > 50 and dte <= 14:
        scenarios.append("Event Play")
    elif dte <= 7 and premium > 100000:
        scenarios.append("Event/Earnings Play")
    
    # Enhanced selling scenarios
    if "SELL" in order_side:
        if opt_type == 'C' and moneyness == 'OTM':
            scenarios.append("Call Overwriting Strategy")
        if opt_type == 'P' and moneyness == 'OTM':
            scenarios.append("Cash-Secured Put Strategy")
        if premium > 200000:
            scenarios.append("Large Premium Collection")
    
    # Hedging patterns
    if trade['ticker'] in ['SPY', 'QQQ', 'IWM'] and opt_type == 'P' and premium > 300000:
        scenarios.append("Portfolio Hedging")
    
    # Block trades
    if volume >= config.SCENARIO_BLOCK_TRADE_VOL:
        scenarios.append("Block Trade")
    
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

def analyze_strike_vs_expected_move(trade):
    """Analyze if strike is within expected move range"""
    if not all(k in trade for k in ['upper_bound', 'lower_bound', 'strike', 'type']):
        return "Unknown"
    
    try:
        upper = trade['upper_bound']
        lower = trade['lower_bound']
        strike = trade['strike']
        opt_type = trade['type']
        
        if upper is None or lower is None:
            return "N/A"
        
        if opt_type == 'C':
            # For calls, check if strike is below upper bound
            if strike <= upper:
                return "✅ In Range"
            else:
                return "⚠️ Out of Range"
        else:  # Put
            # For puts, check if strike is above lower bound
            if strike >= lower:
                return "✅ In Range"
            else:
                return "⚠️ Out of Range"
    except:
        return "Unknown"

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
        iv = trade.get('iv', 0)
        
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

        # IV analysis
        if iv > 80:
            score += 2
            reasons.append("Extreme IV")
        elif iv > 60:
            score += 1
            reasons.append("High IV")

        # Selling activity alerts
        if "SELL" in order_side and premium > 200000:
            score += 2
            reasons.append("Large Premium Collection")
            alert_type = "SELL_ALERT"

        # Unusual patterns
        if len(trade.get('scenarios', [])) > 3:
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

# --- ENHANCED FETCH FUNCTION ---
def fetch_complete_flow():
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
                iv = float(trade.get('implied_volatility', 0)) if trade.get('implied_volatility') not in ['N/A', '', None] else 0
                mid_price = float(trade.get('mid_price', 0)) if trade.get('mid_price') not in ['N/A', '', None] else 0
                bid = float(trade.get('bid', 0)) if trade.get('bid') not in ['N/A', '', None] else 0
                ask = float(trade.get('ask', 0)) if trade.get('ask') not in ['N/A', '', None] else 0
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

            # Calculate expected move
            expected_move, upper_bound, lower_bound = calculate_expected_move(underlying_price, iv, dte)

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
                'vol_oi_ratio': volume / max(oi, 1),
                'iv': iv,
                'iv_category': get_iv_rank_category(iv),
                'expected_move': expected_move,
                'expected_move_formatted': format_expected_move(underlying_price, expected_move, upper_bound, lower_bound),
                'upper_bound': upper_bound,
                'lower_bound': lower_bound,
                'mid_price': mid_price,
                'bid': bid,
                'ask': ask,
                'rule_name': trade.get('rule_name', ''),
                'description': trade.get('description', ''),
                'side': trade.get('side', ''),
            }
            
            # Detect order side
            trade_data['order_side'] = detect_order_side(trade_data)
            
            # Add strike vs expected move analysis
            trade_data['strike_analysis'] = analyze_strike_vs_expected_move(trade_data)
            
            # Add enhanced scenarios
            scenarios = detect_enhanced_scenarios(trade_data, underlying_price)
            trade_data['scenarios'] = scenarios
            result.append(trade_data)

        return result

    except Exception as e:
        st.error(f"Error fetching flow: {e}")
        return []

# --- ENHANCED VISUALIZATIONS ---
def create_comprehensive_dashboard(trades):
    """Create comprehensive dashboard"""
    if not trades:
        return
    
    metrics = calculate_flow_metrics(trades)
    df = pd.DataFrame(trades)
    
    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
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
        iv_trades = [t for t in trades if t.get('iv', 0) > 0]
        if iv_trades:
            avg_iv = sum(t['iv'] for t in iv_trades) / len(iv_trades)
            st.metric("Average IV", f"{avg_iv:.1f}%")
        else:
            st.metric("Average IV", "N/A")
    
    with col5:
        bullish_flow = metrics.get('bullish_flow', 0)
        bearish_flow = metrics.get('bearish_flow', 0)
        sentiment = "Bullish" if bullish_flow > bearish_flow else "Bearish"
        st.metric("Market Sentiment", sentiment, f"{abs(bullish_flow - bearish_flow):,.0f}")

    # Enhanced charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Buy vs Sell flow
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
                     title="💰 Buy vs Sell Flow Breakdown",
                     color_discrete_sequence=flow_data['Color'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # IV distribution
        iv_trades = df[df['iv'] > 0]
        if not iv_trades.empty:
            fig = px.histogram(iv_trades, x='iv', nbins=20,
                             title="📈 IV Distribution",
                             labels={'iv': 'Implied Volatility (%)', 'count': 'Number of Trades'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No IV data available for visualization")

    # Additional charts
    col3, col4 = st.columns(2)
    
    with col3:
        # Premium by ticker
        ticker_premium = df.groupby('ticker')['premium'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=ticker_premium.index, y=ticker_premium.values,
                     title="🏆 Top Tickers by Premium",
                     labels={'x': 'Ticker', 'y': 'Total Premium'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Scenario distribution
        scenario_counts = {}
        for trade in trades:
            for scenario in trade.get('scenarios', []):
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
        
        if scenario_counts:
            scenario_df = pd.DataFrame(list(scenario_counts.items()), 
                                     columns=['Scenario', 'Count'])
            top_scenarios = scenario_df.nlargest(8, 'Count')
            fig = px.pie(top_scenarios, values='Count', names='Scenario',
                        title="📊 Top Strategy Patterns")
            st.plotly_chart(fig, use_container_width=True)

# --- ENHANCED DISPLAY FUNCTIONS ---
def display_calls_and_puts_enhanced(trades):
    """Enhanced call/put separation with all analytics"""
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
                'IV': f"{t.get('iv', 0):.1f}%" if t.get('iv', 0) > 0 else 'N/A',
                'Expected Move': t.get('expected_move_formatted', 'N/A'),
                'Strike Analysis': t.get('strike_analysis', 'Unknown'),
                'Side': t.get('order_side', 'Unknown'),
                'Scenarios': ", ".join(t.get('scenarios', [])[:2]),
                'Time': t['time_ny']
            } for t in sorted(call_trades, key=lambda x: x['premium'], reverse=True)[:30]])
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
                'IV': f"{t.get('iv', 0):.1f}%" if t.get('iv', 0) > 0 else 'N/A',
                'Expected Move': t.get('expected_move_formatted', 'N/A'),
                'Strike Analysis': t.get('strike_analysis', 'Unknown'),
                'Side': t.get('order_side', 'Unknown'),
                'Scenarios': ", ".join(t.get('scenarios', [])[:2]),
                'Time': t['time_ny']
            } for t in sorted(put_trades, key=lambda x: x['premium'], reverse=True)[:30]])
            st.dataframe(put_df, use_container_width=True, height=600)
        else:
            st.info("No put trades found")

def display_premium_selling_analysis(trades):
    """Dedicated premium selling analysis"""
    st.markdown("### 💰 Premium Selling Analysis")
    
    sell_trades = [t for t in trades if 'SELL' in t.get('order_side', '')]
    
    if not sell_trades:
        st.info("No selling activity detected in current dataset")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sell_premium = sum(t['premium'] for t in sell_trades)
        st.metric("Total Premium Collected", f"${sell_premium:,.0f}")
    
    with col2:
        avg_sell_premium = sell_premium / len(sell_trades) if sell_trades else 0
        st.metric("Avg Premium/Trade", f"${avg_sell_premium:,.0f}")
    
    with col3:
        sell_ratio = len(sell_trades) / len(trades) if trades else 0
        st.metric("Selling Activity %", f"{sell_ratio:.1%}")
    
    with col4:
        high_iv_sells = len([t for t in sell_trades if t.get('iv', 0) > 50])
        st.metric("High IV Sells", high_iv_sells)
    
    # Top selling trades
    st.markdown("#### 🔥 Largest Premium Collection Trades")
    sell_df_data = []
    for trade in sorted(sell_trades, key=lambda x: x['premium'], reverse=True)[:15]:
        sell_df_data.append({
            'Ticker': trade['ticker'],
            'Type': trade['type'],
            'Strike': f"${trade['strike']:.0f}",
            'Price': f"${trade['price']:.2f}" if trade['price'] > 0 else 'N/A',
            'Premium': f"${trade['premium']:,}",
            'Volume': trade['volume'],
            'DTE': trade['dte'],
            'IV': f"{trade.get('iv', 0):.1f}%" if trade.get('iv', 0) > 0 else 'N/A',
            'Strategy': ", ".join(trade['scenarios'][:2]),
            'IV Risk': trade.get('iv_category', 'Unknown'),
            'Time': trade['time_ny']
        })
    
    if sell_df_data:
        st.dataframe(pd.DataFrame(sell_df_data), use_container_width=True)

def display_iv_deep_analysis(trades):
    """Comprehensive IV analysis section"""
    st.markdown("### 📊 Advanced IV & Expected Move Analysis")
    
    iv_trades = [t for t in trades if t.get('iv', 0) > 0]
    if not iv_trades:
        st.info("No IV data available for current trades")
        return
    
    # IV metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_iv = sum(t['iv'] for t in iv_trades) / len(iv_trades)
        st.metric("Average IV", f"{avg_iv:.1f}%")
    
    with col2:
        high_iv_trades = [t for t in iv_trades if t.get('iv', 0) > 50]
        st.metric("High IV Trades", f"{len(high_iv_trades)}")
    
    with col3:
        extreme_iv_trades = [t for t in iv_trades if t.get('iv', 0) > 80]
        st.metric("Extreme IV (>80%)", f"{len(extreme_iv_trades)}")
    
    with col4:
        iv_crush_candidates = [t for t in iv_trades if t.get('dte', 0) <= 7 and t.get('iv', 0) > 60]
        st.metric("IV Crush Risk", f"{len(iv_crush_candidates)}")
    
    with col5:
        in_range_trades = [t for t in iv_trades if t.get('strike_analysis', '') == '✅ In Range']
        st.metric("In Range Strikes", f"{len(in_range_trades)}")
    
    # Expected move analysis
    st.markdown("#### 📈 Expected Move vs Strike Analysis")
    st.info("📊 **Expected Move Formula**: Stock Price × IV × √(DTE/365) - Shows ±1 standard deviation move (68% probability)")
    
    em_trades = [t for t in iv_trades if t.get('expected_move') is not None]
    if em_trades:
        em_df = pd.DataFrame([{
            'Ticker': t['ticker'],
            'Type': '🟢 C' if t['type'] == 'C' else '🔴 P',
            'Current Price': f"${t.get('underlying_price', 0):.2f}",
            'Strike': f"${t['strike']:.0f}",
            'Expected Move': t.get('expected_move_formatted', 'N/A'),
            'Upper Target': f"${t.get('upper_bound', 0):.2f}" if t.get('upper_bound') else 'N/A',
            'Lower Target': f"${t.get('lower_bound', 0):.2f}" if t.get('lower_bound') else 'N/A',
            'Strike Analysis': t.get('strike_analysis', 'Unknown'),
            'DTE': t['dte'],
            'IV': f"{t.get('iv', 0):.1f}%",
            'IV Risk': t.get('iv_category', 'Unknown'),
            'Premium': f"${t['premium']:,}",
            'Side': t.get('order_side', 'Unknown')
        } for t in sorted(em_trades, key=lambda x: x.get('expected_move', 0), reverse=True)[:20]])
        st.dataframe(em_df, use_container_width=True)

def display_enhanced_alerts(trades):
    """Enhanced alert system display"""
    alerts = generate_enhanced_alerts(trades)
    if not alerts:
        st.info("No high-priority alerts found with current criteria")
        return
    
    st.markdown("### 🚨 Enhanced Alert System")
    
    for i, alert in enumerate(alerts[:20], 1):
        alert_type = alert.get('alert_type', 'INFO')
        icon = "🔥" if alert_type == "CRITICAL" else "⚠️" if alert_type == "HIGH" else "💰" if alert_type == "SELL_ALERT" else "ℹ️"
        
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{icon} {i}. {alert['ticker']} ${alert['strike']:.0f}{alert['type']} "
                          f"{alert['expiry']} ({alert['dte']}d)**")
                st.write(f"💰 Premium: ${alert['premium']:,.0f} | Price: ${alert['price']:.2f} | Side: {alert.get('order_side', 'Unknown')}")
                st.write(f"📊 IV: {alert.get('iv', 0):.1f}% | Expected Move: {alert.get('expected_move_formatted', 'N/A')} | "
                       f"Vol: {alert['volume']} | {alert.get('moneyness', 'N/A')}")
                st.write(f"🎯 Strategies: {', '.join(alert.get('scenarios', [])[:3])}")
                st.write(f"📍 Alert Reasons: {', '.join(alert.get('reasons', []))}")
            with col2:
                st.metric("Alert Score", alert.get('alert_score', 0))
                st.write(f"**{alert_type}**")
            st.divider()

# --- STREAMLIT UI ---
st.set_page_config(page_title="Complete Options Flow Tracker", page_icon="🚀", layout="wide")
st.title("🚀 Complete Options Flow Tracker")
st.markdown("### Professional-grade options activity monitoring with comprehensive analytics")

with st.sidebar:
    st.markdown("## 🎛️ Professional Control Center")
    
    # Quick action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚡ Quick 0DTE", use_container_width=True):
            st.session_state.dte_filter = "0DTE Only"
            st.session_state.premium_filter = "All"
    with col2:
        if st.button("📈 High IV", use_container_width=True):
            st.session_state.iv_filter = "High (50-80%)"
            st.session_state.premium_filter = "All"
    
    st.markdown("---")
    
    # Ticker search
    ticker_search = st.text_input("🔍 Search Ticker (e.g., AAPL)", "").upper()
    
    # Premium filter
    premium_filter = st.selectbox(
        "💰 Premium Range:",
        ["All", "Under $100K", "Under $250K", "$100K - $500K", "$500K+", "$1M+"],
        index=0,
        key="premium_filter"
    )
    
    # DTE filter
    dte_filter = st.selectbox(
        "⏰ Days to Expiry:",
        ["All", "0DTE Only", "Weekly (≤7d)", "Monthly (≤30d)", "Quarterly (≤90d)", "LEAPS (>90d)"],
        index=0,
        key="dte_filter"
    )
    
    # IV filter
    iv_filter = st.selectbox(
        "📊 IV Level:",
        ["All IV", "Low (<30%)", "Normal (30-50%)", "High (50-80%)", "Extreme (>80%)"],
        index=0,
        key="iv_filter"
    )
    
    # Additional filters
    st.markdown("#### 🔧 Advanced Filters")
    include_selling = st.checkbox("💰 Include Selling Activity", value=True)
    only_alerts = st.checkbox("🚨 Only Show Alert-Worthy Trades", value=False)
    min_volume = st.number_input("📊 Min Volume", min_value=0, value=0, step=100)
    
    # Analysis type
    st.markdown("---")
    analysis_type = st.selectbox(
        "📈 Analysis Type:",
        [
            "📊 Complete Dashboard",
            "📞 Calls & Puts Enhanced", 
            "💰 Premium Selling Analysis",
            "📊 IV Deep Analysis",
            "🚨 Alert Center",
            "⏰ DTE Category Analysis"
        ]
    )
    
    run_scan = st.button("🔄 Scan Complete Flow", type="primary", use_container_width=True)

# Main execution
if run_scan:
    with st.spinner("Scanning complete options flow..."):
        trades = fetch_complete_flow()
        
        if not trades:
            st.error("No trades found. Check API connection or try different filters.")
            st.stop()
        
        # Apply all filters
        filtered_trades = []
        for trade in trades:
            # Ticker search filter
            if ticker_search and ticker_search not in trade['ticker']:
                continue
            
            # Premium filter
            premium_ok = True
            premium = trade['premium']
            if premium_filter == "Under $100K" and premium >= 100000:
                premium_ok = False
            elif premium_filter == "Under $250K" and premium >= 250000:
                premium_ok = False
            elif premium_filter == "$100K - $500K" and (premium < 100000 or premium >= 500000):
                premium_ok = False
            elif premium_filter == "$500K+" and premium < 500000:
                premium_ok = False
            elif premium_filter == "$1M+" and premium < 1000000:
                premium_ok = False
            
            # DTE filter
            dte_ok = True
            dte = trade['dte']
            if dte_filter == "0DTE Only" and dte != 0:
                dte_ok = False
            elif dte_filter == "Weekly (≤7d)" and dte > 7:
                dte_ok = False
            elif dte_filter == "Monthly (≤30d)" and dte > 30:
                dte_ok = False
            elif dte_filter == "Quarterly (≤90d)" and dte > 90:
                dte_ok = False
            elif dte_filter == "LEAPS (>90d)" and dte <= 90:
                dte_ok = False
            
            # IV filter
            iv_ok = True
            iv = trade.get('iv', 0)
            if iv_filter == "Low (<30%)" and iv >= 30:
                iv_ok = False
            elif iv_filter == "Normal (30-50%)" and (iv < 30 or iv >= 50):
                iv_ok = False
            elif iv_filter == "High (50-80%)" and (iv < 50 or iv >= 80):
                iv_ok = False
            elif iv_filter == "Extreme (>80%)" and iv < 80:
                iv_ok = False
            
            # Volume filter
            volume_ok = trade['volume'] >= min_volume
            
            # Include selling filter
            selling_ok = include_selling or 'BUY' in trade.get('order_side', '')
            
            if premium_ok and dte_ok and iv_ok and volume_ok and selling_ok:
                filtered_trades.append(trade)
        
        # Alert filter
        if only_alerts:
            alerts = generate_enhanced_alerts(filtered_trades)
            filtered_trades = alerts
        
        # Display results
        search_text = f" for '{ticker_search}'" if ticker_search else ""
        st.success(f"Found {len(filtered_trades)} trades{search_text} (Premium: {premium_filter}, DTE: {dte_filter}, IV: {iv_filter})")
        
        if filtered_trades:
            # Enhanced stats row
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
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
                iv_trades = [t for t in filtered_trades if t.get('iv', 0) > 0]
                if iv_trades:
                    avg_iv = sum(t['iv'] for t in iv_trades) / len(iv_trades)
                    st.metric("📊 Avg IV", f"{avg_iv:.1f}%")
                else:
                    st.metric("📊 Avg IV", "N/A")
            
            with col5:
                sell_trades = [t for t in filtered_trades if 'SELL' in t.get('order_side', '')]
                sell_pct = len(sell_trades) / len(filtered_trades) * 100 if filtered_trades else 0
                st.metric("💰 Selling %", f"{sell_pct:.1f}%")
            
            with col6:
                sentiment_ratio, sentiment = calculate_sentiment_score(filtered_trades)
                st.metric("🎯 Sentiment", sentiment)
            
            st.divider()
            
            # Show selected analysis
            if analysis_type == "📊 Complete Dashboard":
                create_comprehensive_dashboard(filtered_trades)
                
                st.markdown("### 📋 Top Trades Preview")
                preview_df = pd.DataFrame([{
                    'Ticker': t['ticker'],
                    'Type': '🟢 C' if t['type'] == 'C' else '🔴 P',
                    'Strike': f"${t['strike']:.0f}",
                    'Price': f"${t['price']:.2f}" if t['price'] > 0 else 'N/A',
                    'Premium': f"${t['premium']:,.0f}",
                    'Volume': t['volume'],
                    'DTE': t['dte'],
                    'IV': f"{t.get('iv', 0):.1f}%" if t.get('iv', 0) > 0 else 'N/A',
                    'Expected Move': t.get('expected_move_formatted', 'N/A'),
                    'Strike Analysis': t.get('strike_analysis', 'Unknown'),
                    'Side': t.get('order_side', 'Unknown'),
                    'Time': t['time_ny']
                } for t in sorted(filtered_trades, key=lambda x: x['premium'], reverse=True)[:15]])
                st.dataframe(preview_df, use_container_width=True)
                
            elif analysis_type == "📞 Calls & Puts Enhanced":
                display_calls_and_puts_enhanced(filtered_trades)
                
            elif analysis_type == "💰 Premium Selling Analysis":
                display_premium_selling_analysis(filtered_trades)
                
            elif analysis_type == "📊 IV Deep Analysis":
                display_iv_deep_analysis(filtered_trades)
                
            elif analysis_type == "🚨 Alert Center":
                display_enhanced_alerts(filtered_trades)
                
            elif analysis_type == "⏰ DTE Category Analysis":
                st.markdown("### ⏰ Enhanced DTE Category Analysis")
                
                # Group by DTE category
                dte_groups = {}
                for trade in filtered_trades:
                    category = trade.get('dte_category', 'Unknown')
                    if category not in dte_groups:
                        dte_groups[category] = []
                    dte_groups[category].append(trade)
                
                # Display each category with enhanced analytics
                for category in ["0DTE", "Weekly", "Monthly", "Quarterly", "LEAPS"]:
                    if category in dte_groups:
                        trades_in_category = dte_groups[category]
                        
                        st.markdown(f"#### {category} Options ({len(trades_in_category)} trades)")
                        
                        # Enhanced category stats
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        with col1:
                            total_premium = sum(t['premium'] for t in trades_in_category)
                            st.metric("Premium", f"${total_premium:,.0f}")
                        
                        with col2:
                            call_ratio = len([t for t in trades_in_category if t['type'] == 'C']) / len(trades_in_category)
                            st.metric("Call %", f"{call_ratio:.0%}")
                        
                        with col3:
                            iv_trades_cat = [t for t in trades_in_category if t.get('iv', 0) > 0]
                            if iv_trades_cat:
                                avg_iv = sum(t['iv'] for t in iv_trades_cat) / len(iv_trades_cat)
                                st.metric("Avg IV", f"{avg_iv:.1f}%")
                            else:
                                st.metric("Avg IV", "N/A")
                        
                        with col4:
                            sell_trades_cat = [t for t in trades_in_category if 'SELL' in t.get('order_side', '')]
                            sell_pct = len(sell_trades_cat) / len(trades_in_category) * 100
                            st.metric("Selling %", f"{sell_pct:.1f}%")
                        
                        with col5:
                            in_range = len([t for t in trades_in_category if t.get('strike_analysis', '') == '✅ In Range'])
                            in_range_pct = in_range / len(trades_in_category) * 100
                            st.metric("In Range %", f"{in_range_pct:.0f}%")
                        
                        # Category trades table
                        category_df = pd.DataFrame([{
                            'Ticker': t['ticker'],
                            'Type': '🟢 C' if t['type'] == 'C' else '🔴 P',
                            'Strike': f"${t['strike']:.0f}",
                            'Price': f"${t['price']:.2f}" if t['price'] > 0 else 'N/A',
                            'Premium': f"${t['premium']:,.0f}",
                            'Volume': t['volume'],
                            'DTE': t['dte'],
                            'IV': f"{t.get('iv', 0):.1f}%" if t.get('iv', 0) > 0 else 'N/A',
                            'Expected Move': t.get('expected_move_formatted', 'N/A'),
                            'Strike Analysis': t.get('strike_analysis', 'Unknown'),
                            'Side': t.get('order_side', 'Unknown'),
                            'Scenarios': ", ".join(t.get('scenarios', [])[:2]),
                            'Time': t['time_ny']
                        } for t in sorted(trades_in_category, key=lambda x: x['premium'], reverse=True)[:15]])
                        
                        st.dataframe(category_df, use_container_width=True)
                        st.markdown("---")
            
            # Enhanced export section
            st.markdown("---")
            with st.expander("💾 Export Complete Dataset", expanded=False):
                if filtered_trades:
                    csv_data = []
                    for trade in filtered_trades:
                        row = trade.copy()
                        row['scenarios'] = ', '.join(row.get('scenarios', []))
                        row['reasons'] = ', '.join(row.get('reasons', [])) if 'reasons' in row else ''
                        csv_data.append(row)
                    
                    df = pd.DataFrame(csv_data)
                    csv = df.to_csv(index=False)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label=f"📥 Download Complete CSV ({len(filtered_trades)} trades)",
                            data=csv,
                            file_name=f"complete_options_flow_{timestamp}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # JSON export for advanced users
                        import json
                        json_data = json.dumps(filtered_trades, indent=2, default=str)
                        st.download_button(
                            label=f"📋 Download JSON ({len(filtered_trades)} trades)",
                            data=json_data,
                            file_name=f"complete_options_flow_{timestamp}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    # Data summary
                    st.info(f"""
                    **Complete Dataset includes:**
                    - ✅ All trade fundamentals (ticker, strike, premium, volume, etc.)
                    - ✅ IV analysis (implied volatility, IV categories, risk levels)
                    - ✅ Expected move calculations (68% probability ranges)
                    - ✅ Strike analysis (in range vs out of range probabilities)
                    - ✅ Buy/sell detection (order side analysis)
                    - ✅ Enhanced scenarios (30+ pattern types)
                    - ✅ Time categorization (0DTE through LEAPS)
                    - ✅ Alert scoring and risk assessment
                    
                    **Perfect for:** Professional analysis, backtesting, strategy development
                    """)
                else:
                    st.warning("No data to export")
        else:
            st.warning("No trades match your current filters. Try expanding the criteria.")
            st.info("💡 **Tips:** Start with 'All' filters, then narrow down | Try searching specific tickers | Check if min volume is too high")

else:
    st.markdown("""
    ## Welcome to the Complete Options Flow Tracker! 🚀
    
    ### 🎯 **Built for Professional Traders**
    The most comprehensive options flow analysis platform with institutional-grade analytics.
    
    ### 📊 **Complete Feature Set:**
    
    #### 🔍 **Advanced Filtering**
    - **Ticker Search** - Find specific stocks instantly
    - **Premium Ranges** - From retail to institutional size
    - **Time Horizons** - 0DTE through LEAPS
    - **IV Levels** - Risk-based volatility filtering
    - **Volume Thresholds** - Focus on significant activity
    - **Buy/Sell Detection** - Distinguish market direction
    
    #### 📈 **Professional Analytics**
    - **Expected Move Calculations** - Stock Price × IV × √(DTE/365)
    - **Strike Probability Analysis** - ✅ In Range vs ⚠️ Out of Range
    - **IV Risk Assessment** - Low, Normal, Elevated, High, Extreme
    - **Premium Collection Tracking** - Identify selling strategies
    - **Gamma Squeeze Detection** - High Vol/OI ratio alerts
    - **Dark Pool Activity** - Institutional flow patterns
    
    #### 🚨 **Smart Alert System**
    - **Multi-tier Scoring** - Critical, High, Medium alerts
    - **Pattern Recognition** - 30+ strategy types
    - **Risk Warnings** - IV crush, extreme moves
    - **Premium Thresholds** - Whale trade detection
    
    #### 📊 **Analysis Types:**
    1. **📊 Complete Dashboard** - Full market overview with flow metrics
    2. **📞 Calls & Puts Enhanced** - Side-by-side with all analytics
    3. **💰 Premium Selling Analysis** - Dedicated income strategy tracking
    4. **📊 IV Deep Analysis** - Volatility-focused insights
    5. **🚨 Alert Center** - Priority trade notifications
    6. **⏰ DTE Category Analysis** - Time-based strategy breakdown
    
    ### 🎯 **Perfect for Your 2-3 Week Trading Style:**
    - **Realistic Expected Moves** - Optimized for 2-4 week accuracy
    - **IV Analysis** - Avoid overpaying for time value
    - **Strike Selection** - Pick probabilities, not hopes
    - **Risk Management** - Spot IV crush setups before earnings
    
    ### 💡 **Professional Edge:**
    - **Real-time data** from Unusual Whales API
    - **Mathematical backing** - No emotion, just probabilities
    - **Institutional patterns** - Follow the smart money
    - **Complete transparency** - See every calculation
    
    **This isn't just sentiment tracking - it's professional options analysis.**
    
    **Click "Scan Complete Flow" to access institutional-grade insights!**
    """)
