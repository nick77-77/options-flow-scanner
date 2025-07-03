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
    HIGH_IV_THRESHOLD = 0.30  # 30% IV threshold for high volatility
    IV_CRUSH_THRESHOLD = 0.15  # 15% IV threshold for potential crush opportunities

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
    """Enhanced scenario detection with selling patterns and IV analysis"""
    scenarios = []
    opt_type = trade['type']
    strike = trade['strike']
    premium = trade['premium']
    volume = trade.get('volume', 0)
    oi = trade.get('oi', 0)
    rule_name = trade.get('rule_name', '')
    ticker = trade['ticker']
    order_side = trade.get('order_side', 'UNKNOWN')
    iv = trade.get('iv', 0)
    
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
    
    # IV-based scenarios
    if iv > config.HIGH_IV_THRESHOLD:
        if "SELL" in order_side:
            scenarios.append("High IV Premium Selling")
        else:
            scenarios.append("High IV Long Position")
    
    if iv > config.IV_CRUSH_THRESHOLD and trade.get('dte', 0) <= 7:
        scenarios.append("Potential IV Crush Play")
    
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
    if iv > config.HIGH_IV_THRESHOLD and premium > 100000:
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
    
    # IV metrics
    iv_values = [t.get('iv', 0) for t in trades if t.get('iv', 0) > 0]
    avg_iv = np.mean(iv_values) if iv_values else 0
    high_iv_trades = [t for t in trades if t.get('iv', 0) > config.HIGH_IV_THRESHOLD]
    
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
        'bearish_flow': put_premium_bought + call_premium_sold,
        'avg_iv': avg_iv,
        'high_iv_count': len(high_iv_trades),
        'high_iv_premium': sum(t['premium'] for t in high_iv_trades)
    }

def calculate_iv_metrics(trades):
    """Calculate detailed IV metrics"""
    if not trades:
        return {}
    
    iv_trades = [t for t in trades if t.get('iv', 0) > 0]
    if not iv_trades:
        return {}
    
    iv_values = [t['iv'] for t in iv_trades]
    
    # Categorize by IV levels
    low_iv = [t for t in iv_trades if t['iv'] <= 0.20]
    medium_iv = [t for t in iv_trades if 0.20 < t['iv'] <= 0.40]
    high_iv = [t for t in iv_trades if t['iv'] > 0.40]
    
    # Premium by IV category
    low_iv_premium = sum(t['premium'] for t in low_iv)
    medium_iv_premium = sum(t['premium'] for t in medium_iv)
    high_iv_premium = sum(t['premium'] for t in high_iv)
    
    # IV selling vs buying
    iv_selling = [t for t in iv_trades if 'SELL' in t.get('order_side', '')]
    iv_buying = [t for t in iv_trades if 'BUY' in t.get('order_side', '')]
    
    return {
        'total_iv_trades': len(iv_trades),
        'avg_iv': np.mean(iv_values),
        'median_iv': np.median(iv_values),
        'max_iv': np.max(iv_values),
        'min_iv': np.min(iv_values),
        'low_iv_count': len(low_iv),
        'medium_iv_count': len(medium_iv),
        'high_iv_count': len(high_iv),
        'low_iv_premium': low_iv_premium,
        'medium_iv_premium': medium_iv_premium,
        'high_iv_premium': high_iv_premium,
        'iv_selling_count': len(iv_selling),
        'iv_buying_count': len(iv_buying),
        'iv_selling_premium': sum(t['premium'] for t in iv_selling),
        'iv_buying_premium': sum(t['premium'] for t in iv_buying),
        'avg_iv_selling': np.mean([t['iv'] for t in iv_selling]) if iv_selling else 0,
        'avg_iv_buying': np.mean([t['iv'] for t in iv_buying]) if iv_buying else 0
    }

def display_iv_analysis(trades):
    """Comprehensive IV analysis section with fallback for missing IV data"""
    st.markdown("### 📊 Implied Volatility Analysis")
    
    # First, let's check what IV data we actually have
    iv_trades = [t for t in trades if t.get('iv', 0) > 0]
    estimated_iv_trades = [t for t in trades if t.get('iv', 0) > 0 and t.get('iv', 0) < 10]  # Reasonable IV range
    
    if not iv_trades:
        st.warning("⚠️ No IV data found in current dataset")
        st.info("""
        **Possible reasons for missing IV data:**
        - API doesn't include IV in this endpoint
        - IV data might be in a different field name
        - Data source limitations
        
        **Alternative Analysis:**
        We can analyze volatility patterns using price movements and option premiums.
        """)
        
        # Alternative volatility analysis without IV
        st.markdown("#### 📈 Alternative Volatility Analysis")
        
        # Analyze by premium levels as proxy for volatility
        high_premium_trades = [t for t in trades if t.get('premium', 0) > 100000]
        medium_premium_trades = [t for t in trades if 50000 <= t.get('premium', 0) <= 100000]
        low_premium_trades = [t for t in trades if t.get('premium', 0) < 50000]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("High Premium Trades", len(high_premium_trades), "Likely High Vol")
        with col2:
            st.metric("Medium Premium Trades", len(medium_premium_trades), "Moderate Vol")
        with col3:
            st.metric("Low Premium Trades", len(low_premium_trades), "Lower Vol")
        
        # Premium distribution chart
        premium_dist = pd.DataFrame({
            'Premium Level': ['High (>$100K)', 'Medium ($50K-$100K)', 'Low (<$50K)'],
            'Count': [len(high_premium_trades), len(medium_premium_trades), len(low_premium_trades)]
        })
        
        if premium_dist['Count'].sum() > 0:
            fig = px.pie(premium_dist, values='Count', names='Premium Level',
                        title="Premium Distribution (Volatility Proxy)")
            st.plotly_chart(fig, use_container_width=True)
        
        # Show high premium trades as volatility proxy
        st.markdown("#### 🔥 High Premium Trades (Volatility Proxy)")
        if high_premium_trades:
            high_premium_data = []
            for trade in sorted(high_premium_trades, key=lambda x: x.get('premium', 0), reverse=True)[:10]:
                # Calculate rough volatility indicators
                vol_indicator = "Very High" if trade.get('premium', 0) > 200000 else "High"
                
                high_premium_data.append({
                    'Ticker': trade['ticker'],
                    'Type': trade['type'],
                    'Strike': f"${trade['strike']:.0f}",
                    'Premium': f"${trade['premium']:,.0f}",
                    'Vol Indicator': vol_indicator,
                    'DTE': trade['dte'],
                    'Volume': trade['volume'],
                    'Strategy': ", ".join(trade.get('scenarios', [])[:2]),
                    'Time': trade['time_ny']
                })
            
            st.dataframe(pd.DataFrame(high_premium_data), use_container_width=True)
        else:
            st.info("No high premium trades found")
        
        return
    
    # If we have IV data, proceed with full analysis
    iv_metrics = calculate_iv_metrics(trades)
    
    # IV Overview Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average IV", f"{iv_metrics['avg_iv']:.1%}")
    
    with col2:
        st.metric("Median IV", f"{iv_metrics['median_iv']:.1%}")
    
    with col3:
        st.metric("High IV Trades", f"{iv_metrics['high_iv_count']}")
    
    with col4:
        iv_range = iv_metrics['max_iv'] - iv_metrics['min_iv']
        st.metric("IV Range", f"{iv_range:.1%}")
    
    # IV Distribution Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # IV Distribution by Category
        iv_dist_data = pd.DataFrame({
            'IV Category': ['Low (≤20%)', 'Medium (20-40%)', 'High (>40%)'],
            'Trade Count': [
                iv_metrics['low_iv_count'],
                iv_metrics['medium_iv_count'], 
                iv_metrics['high_iv_count']
            ],
            'Premium': [
                iv_metrics['low_iv_premium'],
                iv_metrics['medium_iv_premium'],
                iv_metrics['high_iv_premium']
            ]
        })
        
        fig = px.bar(iv_dist_data, x='IV Category', y='Trade Count',
                     title="IV Distribution by Category",
                     color='IV Category',
                     color_discrete_map={
                         'Low (≤20%)': '#90EE90',
                         'Medium (20-40%)': '#FFD700', 
                         'High (>40%)': '#FF6B6B'
                     })
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Premium Distribution by IV
        fig = px.bar(iv_dist_data, x='IV Category', y='Premium',
                     title="Premium Distribution by IV Level",
                     color='IV Category',
                     color_discrete_map={
                         'Low (≤20%)': '#90EE90',
                         'Medium (20-40%)': '#FFD700',
                         'High (>40%)': '#FF6B6B'
                     })
        fig.update_layout(yaxis_tickformat='$,.0f')
        st.plotly_chart(fig, use_container_width=True)
    
    # IV Selling vs Buying Analysis
    st.markdown("#### 💰 IV Selling vs Buying Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selling_ratio = iv_metrics['iv_selling_count'] / iv_metrics['total_iv_trades']
        st.metric("IV Selling Activity", f"{selling_ratio:.1%}")
    
    with col2:
        avg_iv_selling = iv_metrics['avg_iv_selling']
        st.metric("Avg IV (Selling)", f"{avg_iv_selling:.1%}")
    
    with col3:
        avg_iv_buying = iv_metrics['avg_iv_buying']
        st.metric("Avg IV (Buying)", f"{avg_iv_buying:.1%}")
    
    # IV Selling vs Buying Chart
    sell_buy_data = pd.DataFrame({
        'Activity': ['Selling', 'Buying'],
        'Count': [iv_metrics['iv_selling_count'], iv_metrics['iv_buying_count']],
        'Premium': [iv_metrics['iv_selling_premium'], iv_metrics['iv_buying_premium']],
        'Avg IV': [iv_metrics['avg_iv_selling'], iv_metrics['avg_iv_buying']]
    })
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Premium: Selling vs Buying', 'Average IV: Selling vs Buying'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Bar(x=sell_buy_data['Activity'], y=sell_buy_data['Premium'],
               name='Premium', marker_color=['#FF6B6B', '#90EE90']),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=sell_buy_data['Activity'], y=sell_buy_data['Avg IV'],
               name='Avg IV', marker_color=['#FFD700', '#87CEEB']),
        row=1, col=2
    )
    
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # High IV Opportunities
    st.markdown("#### 🔥 High IV Opportunities")
    
    high_iv_trades = [t for t in trades if t.get('iv', 0) > config.HIGH_IV_THRESHOLD]
    
    if high_iv_trades:
        # Sort by IV descending
        high_iv_trades.sort(key=lambda x: x.get('iv', 0), reverse=True)
        
        high_iv_data = []
        for trade in high_iv_trades[:15]:  # Top 15 high IV trades
            high_iv_data.append({
                'Ticker': trade['ticker'],
                'Type': trade['type'],
                'Strike': f"${trade['strike']:.0f}",
                'IV': f"{trade.get('iv', 0):.1%}",
                'DTE': trade['dte'],
                'Premium': f"${trade['premium']:,.0f}",
                'Side': trade.get('order_side', 'Unknown'),
                'Volume': trade['volume'],
                'Strategy': ", ".join(trade.get('scenarios', [])[:2]),
                'Time': trade['time_ny']
            })
        
        st.dataframe(pd.DataFrame(high_iv_data), use_container_width=True)
    else:
        st.info("No high IV trades found in current dataset")
    
    # IV Crush Candidates
    st.markdown("#### ⚡ IV Crush Candidates")
    
    iv_crush_candidates = [
        t for t in trades 
        if t.get('iv', 0) > config.IV_CRUSH_THRESHOLD 
        and t.get('dte', 0) <= 7
        and 'BUY' in t.get('order_side', '')
    ]
    
    if iv_crush_candidates:
        iv_crush_data = []
        for trade in sorted(iv_crush_candidates, key=lambda x: x.get('iv', 0), reverse=True)[:10]:
            iv_crush_data.append({
                'Ticker': trade['ticker'],
                'Type': trade['type'],
                'Strike': f"${trade['strike']:.0f}",
                'IV': f"{trade.get('iv', 0):.1%}",
                'DTE': trade['dte'],
                'Premium': f"${trade['premium']:,.0f}",
                'Volume': trade['volume'],
                'Risk Level': "High" if trade.get('iv', 0) > 0.50 else "Medium",
                'Time': trade['time_ny']
            })
        
        st.dataframe(pd.DataFrame(iv_crush_data), use_container_width=True)
        st.info("💡 **IV Crush Risk**: These positions may lose value rapidly if volatility decreases after events")
    else:
        st.info("No IV crush candidates found")

def generate_enhanced_alerts(trades):
    """Enhanced alert system with selling detection and IV alerts"""
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

        # IV-based alerts
        if iv > 0.50:
            score += 3
            reasons.append("Extreme IV (>50%)")
            alert_type = "IV_ALERT"
        elif iv > config.HIGH_IV_THRESHOLD:
            score += 2
            reasons.append("High IV (>30%)")

        # Selling activity alerts
        if "SELL" in order_side and premium > 200000:
            score += 2
            reasons.append("Large Premium Collection")
            alert_type = "SELL_ALERT"

        # IV crush risk
        if iv > config.IV_CRUSH_THRESHOLD and trade.get('dte', 0) <= 7 and 'BUY' in order_side:
            score += 2
            reasons.append("IV Crush Risk")
            alert_type = "IV_CRUSH_ALERT"

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
    """Enhanced data fetching with better analysis and IV data"""
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
                
                # IV data extraction - check multiple possible field names
                iv = 0
                iv_fields = ['iv', 'implied_volatility', 'volatility', 'impliedVolatility', 'vol', 'IV', 'ivol']
                for field in iv_fields:
                    if field in trade and trade[field] not in ['N/A', '', None, 0]:
                        try:
                            iv = float(trade[field])
                            if iv > 0:
                                break
                        except (ValueError, TypeError):
                            continue
                
                # If no IV found, try to estimate from bid-ask spread and other factors
                if iv == 0:
                    try:
                        # Simple IV estimation based on option price and moneyness
                        option_price = price if price > 0 else mid_price
                        if option_price > 0 and underlying_price > 0 and dte > 0:
                            # Very rough IV estimation (not precise, but gives some indication)
                            moneyness = abs(strike - underlying_price) / underlying_price
                            time_factor = np.sqrt(dte / 365.0)
                            if time_factor > 0:
                                # Rough approximation: IV ≈ option_price / (underlying_price * time_factor)
                                estimated_iv = (option_price / underlying_price) / time_factor
                                # Cap at reasonable levels
                                if 0.01 <= estimated_iv <= 3.0:
                                    iv = estimated_iv
                    except (ValueError, TypeError, ZeroDivisionError):
                        pass
                    
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
                'iv': iv,
                'iv_percentage': f"{iv:.1%}" if iv > 0 else "N/A"
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
    """Create comprehensive dashboard with IV metrics"""
    if not trades:
        return
    
    metrics = calculate_flow_metrics(trades)
    iv_metrics = calculate_iv_metrics(trades)
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
        avg_iv = iv_metrics.get('avg_iv', 0)
        st.metric("Average IV", f"{avg_iv:.1%}", "Market Volatility")
    
    with col4:
        high_iv_premium = iv_metrics.get('high_iv_premium', 0)
        st.metric("High IV Premium", f"${high_iv_premium:,.0f}", "Premium in >40% IV")

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
        # IV Distribution
        if iv_metrics:
            iv_dist_data = pd.DataFrame({
                'IV Level': ['Low (≤20%)', 'Medium (20-40%)', 'High (>40%)'],
                'Count': [
                    iv_metrics.get('low_iv_count', 0),
                    iv_metrics.get('medium_iv_count', 0),
                    iv_metrics.get('high_iv_count', 0)
                ]
            })
            
            fig = px.pie(iv_dist_data, values='Count', names='IV Level',
                        title="IV Distribution",
                        color_discrete_map={
                            'Low (≤20%)': '#90EE90',
                            'Medium (20-40%)': '#FFD700',
                            'High (>40%)': '#FF6B6B'
                        })
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No IV data available for visualization")

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
            'Strike': f"${trade['strike']:.0f}",
            'Price': f"${trade['price']:.2f}" if trade['price'] != 'N/A' and trade['price'] > 0 else 'N/A',
            'IV': trade['iv_percentage'],
            'Expiry': trade['expiry'],
            'DTE': trade['dte'],
            'Premium': f"${trade['premium']:,}",
            'Volume': trade['volume'],
            'Strategy': ", ".join(trade['scenarios'][:2]),
            'Moneyness': trade.get('moneyness', 'Unknown'),
            'Time': trade['time_ny']
        })
    
    if sell_df_data:
        st.dataframe(pd.DataFrame(sell_df_data), use_container_width=True)

# --- MAIN STREAMLIT APP ---
st.set_page_config(page_title="Enhanced Options Flow Tracker", page_icon="🚀", layout="wide")
st.title("🚀 Enhanced Options Flow Tracker")
st.markdown("### Advanced real-time options activity with buy/sell detection, IV analysis, and institutional pattern recognition")

with st.sidebar:
    st.markdown("## 🎛️ Enhanced Control Panel")
    
    scan_type = st.selectbox(
        "Select Analysis Type:",
        [
            "📊 Comprehensive Dashboard",
            "💰 Selling Activity Analysis", 
            "📈 IV Analysis",
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
    
    # IV Filters
    st.markdown("### IV Filters")
    iv_filter = st.selectbox(
        "Implied Volatility:",
        ["All IV", "Low IV (≤20%)", "Medium IV (20-40%)", "High IV (>40%)", "Extreme IV (>50%)"],
        index=0
    )
    
    run_scan = st.button("🔄 Run Enhanced Scan", type="primary", use_container_width=True)
    
    if st.button("📱 Quick 0DTE Scan", use_container_width=True):
        premium_range = "All Premiums (No Filter)"
        dte_filter = "0DTE Only"
        run_scan = True
    
    if st.button("🔥 High IV Scan", use_container_width=True):
        iv_filter = "High IV (>40%)"
        run_scan = True

# Main execution
if run_scan:
    with st.spinner(f"Running {scan_type}..."):
        trades = fetch_enhanced_flow()
        
        if not trades:
            st.error("No trades fetched from API. Check your API token or connection.")
            st.stop()
        
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
        
        # Apply IV filters
        def apply_iv_filter(iv, iv_selection):
            if iv_selection == "All IV":
                return True
            elif iv_selection == "Low IV (≤20%)":
                return iv <= 0.20
            elif iv_selection == "Medium IV (20-40%)":
                return 0.20 < iv <= 0.40
            elif iv_selection == "High IV (>40%)":
                return iv > 0.40
            elif iv_selection == "Extreme IV (>50%)":
                return iv > 0.50
            return True
        
        # Apply filters with safe comparisons
        filtered_trades = []
        for trade in trades:
            try:
                # Safe numeric conversions for filtering
                trade_premium = float(trade.get('premium', 0))
                trade_dte = int(trade.get('dte', 0))
                trade_order_side = str(trade.get('order_side', ''))
                trade_iv = float(trade.get('iv', 0))
                
                # Apply filters
                if (apply_premium_filter(trade_premium, premium_range) and 
                    apply_dte_filter(trade_dte, dte_filter) and
                    apply_iv_filter(trade_iv, iv_filter)):
                    if include_selling or 'BUY' in trade_order_side:
                        filtered_trades.append(trade)
            except (ValueError, TypeError):
                continue  # Skip trades with invalid data
        
        st.success(f"Found {len(filtered_trades)} trades matching criteria (Premium: {premium_range}, DTE: {dte_filter}, IV: {iv_filter})")
        
        if not filtered_trades:
            st.warning("No trades match your current filters. Try adjusting the premium range or include selling activity.")
            st.info("💡 **Tip:** Try 'All Premiums (No Filter)' and 'All DTE' to see all available data.")
        else:
            # Always show a quick preview table first
            st.markdown("### 📋 Quick Preview (Top 10 by Premium)")
            preview_data = []
            for trade in sorted(filtered_trades, key=lambda x: x.get('premium', 0), reverse=True)[:10]:
                preview_data.append({
                    'Ticker': trade['ticker'],
                    'Type': trade['type'], 
                    'Strike': f"${trade['strike']:.0f}",
                    'Price': f"${trade['price']:.2f}" if trade['price'] != 'N/A' and trade['price'] > 0 else 'N/A',
                    'IV': trade['iv_percentage'],
                    'DTE': trade['dte'],
                    'Premium': f"${trade['premium']:,.0f}",
                    'Side': trade.get('order_side', 'Unknown'),
                    'Volume': trade['volume'],
                    'Scenarios': ", ".join(trade.get('scenarios', [])[:2]),
                    'Time': trade['time_ny']
                })
            
            if preview_data:
                st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
            
            # Now show the selected analysis type
            if "Comprehensive" in scan_type:
                create_enhanced_dashboard(filtered_trades)
                
                # Additional detailed tables
                st.markdown("### 📋 Detailed Trade Analysis")
                
                tabs = st.tabs(["🟢 Call Activity", "🔴 Put Activity", "💎 High Premium", "⚡ High Volume", "📊 High IV"])
                
                with tabs[0]:
                    call_trades = [t for t in filtered_trades if t['type'] == 'C']
                    if call_trades:
                        call_df = pd.DataFrame([{
                            'Ticker': t['ticker'], 
                            'Strike': f"${t['strike']:.0f}", 
                            'Price': f"${t['price']:.2f}" if t['price'] != 'N/A' and t['price'] > 0 else 'N/A',
                            'IV': t['iv_percentage'],
                            'Expiry': t['expiry'],
                            'DTE': t['dte'], 
                            'Premium': f"${t['premium']:,}", 
                            'Side': t.get('order_side', 'Unknown'),
                            'Volume': t['volume'],
                            'Scenarios': ", ".join(t['scenarios'][:2]), 
                            'Time': t['time_ny']
                        } for t in sorted(call_trades, key=lambda x: x['premium'], reverse=True)[:20]])
                        st.dataframe(call_df, use_container_width=True)
                    else:
                        st.info("No call trades found with current filters")
                
                with tabs[1]:
                    put_trades = [t for t in filtered_trades if t['type'] == 'P']
                    if put_trades:
                        put_df = pd.DataFrame([{
                            'Ticker': t['ticker'], 
                            'Strike': f"${t['strike']:.0f}", 
                            'Price': f"${t['price']:.2f}" if t['price'] != 'N/A' and t['price'] > 0 else 'N/A',
                            'IV': t['iv_percentage'],
                            'Expiry': t['expiry'],
                            'DTE': t['dte'], 
                            'Premium': f"${t['premium']:,}", 
                            'Side': t.get('order_side', 'Unknown'),
                            'Volume': t['volume'],
                            'Scenarios': ", ".join(t['scenarios'][:2]), 
                            'Time': t['time_ny']
                        } for t in sorted(put_trades, key=lambda x: x['premium'], reverse=True)[:20]])
                        st.dataframe(put_df, use_container_width=True)
                    else:
                        st.info("No put trades found with current filters")
                
                with tabs[2]:
                    high_premium = sorted(filtered_trades, key=lambda x: x['premium'], reverse=True)[:15]
                    if high_premium:
                        hp_df = pd.DataFrame([{
                            'Ticker': t['ticker'], 
                            'Type': t['type'], 
                            'Strike': f"${t['strike']:.0f}",
                            'Price': f"${t['price']:.2f}" if t['price'] != 'N/A' and t['price'] > 0 else 'N/A',
                            'IV': t['iv_percentage'],
                            'Premium': f"${t['premium']:,}", 
                            'Side': t.get('order_side', 'Unknown'),
                            'Volume': t['volume'],
                            'Strategy': ", ".join(t['scenarios'][:2])
                        } for t in high_premium])
                        st.dataframe(hp_df, use_container_width=True)
                    else:
                        st.info("No high premium trades found")
                
                with tabs[3]:
                    high_volume = sorted(filtered_trades, key=lambda x: x['volume'], reverse=True)[:15]
                    if high_volume:
                        hv_df = pd.DataFrame([{
                            'Ticker': t['ticker'], 
                            'Type': t['type'], 
                            'Price': f"${t['price']:.2f}" if t['price'] != 'N/A' and t['price'] > 0 else 'N/A',
                            'IV': t['iv_percentage'],
                            'Volume': t['volume'],
                            'Premium': f"${t['premium']:,}", 
                            'Vol/OI': f"{t['vol_oi_ratio']:.1f}",
                            'Strategy': ", ".join(t['scenarios'][:2])
                        } for t in high_volume])
                        st.dataframe(hv_df, use_container_width=True)
                    else:
                        st.info("No high volume trades found")
                
                with tabs[4]:
                    high_iv = sorted([t for t in filtered_trades if t.get('iv', 0) > 0], 
                                   key=lambda x: x.get('iv', 0), reverse=True)[:15]
                    if high_iv:
                        hiv_df = pd.DataFrame([{
                            'Ticker': t['ticker'], 
                            'Type': t['type'], 
                            'Strike': f"${t['strike']:.0f}",
                            'IV': t['iv_percentage'],
                            'DTE': t['dte'],
                            'Premium': f"${t['premium']:,}", 
                            'Side': t.get('order_side', 'Unknown'),
                            'Volume': t['volume'],
                            'Strategy': ", ".join(t['scenarios'][:2])
                        } for t in high_iv])
                        st.dataframe(hiv_df, use_container_width=True)
                    else:
                        st.info("No high IV trades found")
            
            elif "Selling" in scan_type:
                display_selling_analysis(filtered_trades)
            
            elif "IV Analysis" in scan_type:
                display_iv_analysis(filtered_trades)
            
            elif "Alerts" in scan_type:
                alerts = generate_enhanced_alerts(filtered_trades)
                if alerts:
                    st.markdown("### 🚨 Enhanced Alert System")
                    for i, alert in enumerate(alerts[:15], 1):
                        alert_type = alert.get('alert_type', 'INFO')
                        icon = "🔥" if alert_type == "CRITICAL" else "⚠️" if alert_type == "HIGH" else "💰" if alert_type == "SELL_ALERT" else "📊" if alert_type == "IV_ALERT" else "⚡" if alert_type == "IV_CRUSH_ALERT" else "ℹ️"
                        
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.markdown(f"**{icon} {i}. {alert['ticker']} ${alert['strike']:.0f}{alert['type']} "
                                          f"{alert['expiry']} ({alert['dte']}d)**")
                                st.write(f"💰 Premium: ${alert['premium']:,.0f} | Price: ${alert['price']:.2f} | IV: {alert['iv_percentage']} | Side: {alert.get('order_side', 'Unknown')} | "
                                       f"Vol: {alert['volume']} | {alert.get('moneyness', 'N/A')}")
                                st.write(f"🎯 Strategies: {', '.join(alert.get('scenarios', [])[:3])}")
                                st.write(f"📍 Alert Reasons: {', '.join(alert.get('reasons', []))}")
                            with col2:
                                st.metric("Alert Score", alert.get('alert_score', 0))
                                st.write(f"**{alert_type}**")
                            st.divider()
                else:
                    st.info("No alerts triggered with current criteria")
            
            # Export functionality - always show when there are trades
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
                    
                    # Show column info
                    st.info(f"CSV contains {len(df.columns)} columns: {', '.join(df.columns[:10])}...")
                else:
                    st.warning("No data to export")

else:
    st.markdown("""
    ## Welcome to the Enhanced Options Flow Tracker! 🚀
    
    ### New Features:
    - **🔍 Buy/Sell Detection**: Identify whether traders are buying or selling options
    - **💰 Premium Collection Analysis**: Track large option selling strategies  
    - **📊 IV Analysis**: Comprehensive implied volatility tracking and analysis
    - **⚡ Enhanced Alerts**: Multi-tier alert system with critical/high/sell/IV alerts
    - **🎯 Advanced Scenarios**: Detect gamma squeezes, dark pool activity, institutional flow
    - **📈 Comprehensive Dashboard**: Net flow analysis, put/call ratios, sentiment tracking
    - **🏛️ Institutional Patterns**: Identify large block trades and sophisticated strategies
    
    ### IV Analysis Features:
    - **High IV Identification**: Spot elevated volatility opportunities
    - **IV Crush Detection**: Identify positions at risk of volatility collapse
    - **Premium Collection via IV**: Track high IV selling strategies
    - **IV Distribution Analysis**: Understand market volatility patterns
    - **IV-Based Alerts**: Get notified of extreme volatility conditions
    
    ### Strategy Detection:
    - Call/Put Overwriting
    - Cash-Secured Puts  
    - Iron Condors & Spreads
    - Gamma Squeeze Setups
    - Portfolio Hedging
    - Event/Earnings Plays
    - Volatility Strategies
    - High IV Premium Selling
    
    **Select an analysis type and click Run Enhanced Scan to begin!**
    
    **💡 Pro Tips:**
    - Use "High IV Scan" to find volatility opportunities
    - Check "IV Analysis" for comprehensive volatility insights
    - Set IV filters to focus on specific volatility ranges
    - Monitor IV crush candidates for risk management
    """)
