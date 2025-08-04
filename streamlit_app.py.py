import streamlit as st
import httpx
from datetime import datetime, date
from collections import defaultdict
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo  # Python 3.9+

# --- CONFIGURATION ---
class Config:
    UW_TOKEN = st.secrets.get("UW_TOKEN", "e6e8601a-0746-4cec-a07d-c3eabfc13926")
    EXCLUDE_TICKERS = {'MSTR', 'CRCL', 'COIN', 'META', 'NVDA','AMD'}
    ALLOWED_TICKERS = {'QQQ', 'SPY', 'IWM'}
    MIN_PREMIUM = 100000
    LIMIT = 500
    SCENARIO_OTM_CALL_MIN_PREMIUM = 100000
    SCENARIO_ITM_CONV_MIN_PREMIUM = 50000
    SCENARIO_SWEEP_VOLUME_OI_RATIO = 2
    SCENARIO_BLOCK_TRADE_VOL = 100
    HIGH_IV_THRESHOLD = 0.30  # 30% IV threshold
    EXTREME_IV_THRESHOLD = 0.50  # 50% IV threshold
    IV_CRUSH_THRESHOLD = 0.15  # 15% IV threshold for crush detection
    HIGH_VOL_OI_RATIO = 5.0  # High volume to OI ratio threshold
    UNUSUAL_OI_THRESHOLD = 1000  # Unusual open interest threshold
    
    # New pattern recognition thresholds
    GAMMA_SQUEEZE_THRESHOLD = 0.10  # 10% price movement threshold
    IV_SPIKE_THRESHOLD = 0.20  # 20% IV increase threshold
    MULTI_LEG_TIME_WINDOW = 300  # 5 minutes in seconds
    CORRELATION_THRESHOLD = 0.7  # Correlation threshold for cross-asset analysis

config = Config()

# --- API SETUP ---
headers = {
    'Accept': 'application/json, text/plain',
    'Authorization': config.UW_TOKEN
}
url = 'https://api.unusualwhales.com/api/option-trades/flow-alerts'

# --- ENHANCED BUY/SELL DETECTION ---
def determine_trade_side_enhanced(trade_data, debug=False):
    """
    Enhanced trade side determination with debugging and confidence scoring
    Returns: (side, confidence_score, reasoning)
    """
    reasoning = []
    confidence_scores = []
    
    # Check if there's explicit side information first
    side = trade_data.get('side', '').upper()
    if side in ['BUY', 'SELL']:
        return side, 1.0, ["Explicit side data available"]
    
    # Extract price data safely with multiple field name attempts
    price_fields = ['price', 'fill_price', 'execution_price', 'trade_price']
    bid_fields = ['bid', 'bid_price', 'best_bid']
    ask_fields = ['ask', 'ask_price', 'best_ask', 'offer']
    
    price = bid = ask = 0
    
    # Try to find price data
    for field in price_fields:
        if field in trade_data and trade_data[field] not in ['N/A', '', None]:
            try:
                price = float(trade_data[field])
                break
            except (ValueError, TypeError):
                continue
    
    # Try to find bid data
    for field in bid_fields:
        if field in trade_data and trade_data[field] not in ['N/A', '', None]:
            try:
                bid = float(trade_data[field])
                break
            except (ValueError, TypeError):
                continue
    
    # Try to find ask data
    for field in ask_fields:
        if field in trade_data and trade_data[field] not in ['N/A', '', None]:
            try:
                ask = float(trade_data[field])
                break
            except (ValueError, TypeError):
                continue
    
    try:
        volume = float(trade_data.get('volume', 0))
        oi = float(trade_data.get('open_interest', 1))
    except (ValueError, TypeError):
        volume = oi = 0
    
    if debug:
        st.write(f"**Price Data**: Price={price}, Bid={bid}, Ask={ask}, Volume={volume}, OI={oi}")
    
    # Method 1: Bid/Ask Analysis (Most Reliable)
    if bid > 0 and ask > 0 and price > 0:
        mid_price = (bid + ask) / 2
        spread_pct = (ask - bid) / mid_price if mid_price > 0 else 0
        
        # More aggressive thresholds for clearer signals
        if price >= ask * 0.98:  # Within 2% of ask = strong BUY signal
            reasoning.append(f"Price {price:.2f} very close to ask {ask:.2f}")
            confidence_scores.append(0.9)
            preliminary_side = "BUY"
        elif price <= bid * 1.02:  # Within 2% of bid = strong SELL signal
            reasoning.append(f"Price {price:.2f} very close to bid {bid:.2f}")
            confidence_scores.append(0.9)
            preliminary_side = "SELL"
        elif price > mid_price * 1.02:  # Above mid = likely BUY
            reasoning.append(f"Price {price:.2f} above midpoint {mid_price:.2f}")
            confidence_scores.append(0.6)
            preliminary_side = "BUY"
        elif price < mid_price * 0.98:  # Below mid = likely SELL
            reasoning.append(f"Price {price:.2f} below midpoint {mid_price:.2f}")
            confidence_scores.append(0.6)
            preliminary_side = "SELL"
        else:
            reasoning.append(f"Price {price:.2f} near midpoint {mid_price:.2f}")
            confidence_scores.append(0.3)
            preliminary_side = "NEUTRAL"
        
        if debug:
            st.write(f"**Bid/Ask Analysis**: Mid={mid_price:.2f}, Spread={spread_pct:.1%}, Initial={preliminary_side}")
    else:
        preliminary_side = "UNKNOWN"
        reasoning.append("No valid bid/ask data")
        confidence_scores.append(0.1)
    
    # Method 2: Volume/OI Analysis
    vol_oi_ratio = volume / max(oi, 1)
    if vol_oi_ratio > 10:  # Extremely high ratio = very likely new buying
        reasoning.append(f"Extremely high Vol/OI ratio: {vol_oi_ratio:.1f}")
        if preliminary_side in ["BUY", "NEUTRAL", "UNKNOWN"]:
            confidence_scores.append(0.8)
            preliminary_side = "BUY"
        else:
            confidence_scores.append(0.5)  # Conflicting signal
    elif vol_oi_ratio > 5:  # Very high ratio = likely new buying
        reasoning.append(f"Very high Vol/OI ratio: {vol_oi_ratio:.1f}")
        if preliminary_side in ["BUY", "NEUTRAL", "UNKNOWN"]:
            confidence_scores.append(0.7)
            preliminary_side = "BUY"
        else:
            confidence_scores.append(0.4)  # Conflicting signal
    elif vol_oi_ratio > 2:  # High ratio = likely buying
        reasoning.append(f"High Vol/OI ratio: {vol_oi_ratio:.1f}")
        if preliminary_side in ["BUY", "NEUTRAL", "UNKNOWN"]:
            confidence_scores.append(0.5)
            preliminary_side = "BUY"
        else:
            confidence_scores.append(0.3)
    elif vol_oi_ratio > 0.1:
        reasoning.append(f"Normal Vol/OI ratio: {vol_oi_ratio:.1f}")
        confidence_scores.append(0.2)
    else:
        reasoning.append(f"Low Vol/OI ratio: {vol_oi_ratio:.1f}")
        confidence_scores.append(0.1)
    
    # Method 3: Description and Rule Analysis
    description = trade_data.get('description', '').lower()
    rule_name = trade_data.get('rule_name', '').lower()
    
    # Strong buying indicators
    strong_buy_keywords = ['sweep', 'aggressive', 'lifted', 'taken', 'market buy', 'block buy', 'opening buy']
    if any(keyword in description for keyword in strong_buy_keywords):
        reasoning.append(f"Strong buy keywords in description")
        confidence_scores.append(0.8)
        preliminary_side = "BUY"
    
    # Strong selling indicators  
    strong_sell_keywords = ['sold', 'offer hit', 'market sell', 'hit bid', 'block sell', 'closing sell']
    if any(keyword in description for keyword in strong_sell_keywords):
        reasoning.append(f"Strong sell keywords in description")
        confidence_scores.append(0.8)
        preliminary_side = "SELL"
    
    # Rule-based analysis
    if 'ascending' in rule_name:
        reasoning.append(f"Ascending rule pattern")
        confidence_scores.append(0.6)
        if preliminary_side in ["UNKNOWN", "NEUTRAL"]:
            preliminary_side = "BUY"
    elif 'descending' in rule_name:
        reasoning.append(f"Descending rule pattern")
        confidence_scores.append(0.6)
        if preliminary_side in ["UNKNOWN", "NEUTRAL"]:
            preliminary_side = "SELL"
    
    # Check for repeated hits pattern
    if 'repeatedhits' in rule_name:
        reasoning.append("Repeated hits pattern detected")
        confidence_scores.append(0.4)
        if vol_oi_ratio > 3:  # High volume suggests buying
            if preliminary_side in ["UNKNOWN", "NEUTRAL"]:
                preliminary_side = "BUY"
    
    # Method 4: Option Type and Moneyness Analysis
    option_type = trade_data.get('type', '')
    try:
        strike = float(trade_data.get('strike', 0))
        underlying = float(trade_data.get('underlying_price', strike))
        
        if underlying > 0 and strike > 0:
            moneyness_pct = ((strike - underlying) / underlying) * 100
            
            if option_type == 'C' and moneyness_pct > 5:  # OTM calls
                reasoning.append("OTM call - typically bought for speculation")
                if preliminary_side in ["UNKNOWN", "NEUTRAL"]:
                    confidence_scores.append(0.4)
                    preliminary_side = "BUY"
            elif option_type == 'P' and moneyness_pct < -5:  # OTM puts  
                reasoning.append("OTM put - typically bought for hedging/speculation")
                if preliminary_side in ["UNKNOWN", "NEUTRAL"]:
                    confidence_scores.append(0.4)
                    preliminary_side = "BUY"
            elif option_type == 'P' and moneyness_pct > 2:  # ITM puts
                reasoning.append("ITM put - could be protective or speculative")
                confidence_scores.append(0.3)
            elif abs(moneyness_pct) < 2:  # ATM options
                reasoning.append("At-the-money option")
                confidence_scores.append(0.2)
    except (ValueError, TypeError):
        pass
    
    # Method 5: Premium Size Analysis
    try:
        premium = float(trade_data.get('premium', 0))
        if premium > 1000000:  # Very large premium trades
            reasoning.append(f"Very large premium trade: ${premium:,.0f}")
            confidence_scores.append(0.4)
            # Very large trades are often institutional buying
            if preliminary_side in ["UNKNOWN", "NEUTRAL"]:
                preliminary_side = "BUY"
        elif premium > 500000:  # Large premium trades
            reasoning.append(f"Large premium trade: ${premium:,.0f}")
            confidence_scores.append(0.3)
            if preliminary_side in ["UNKNOWN", "NEUTRAL"]:
                preliminary_side = "BUY"
    except (ValueError, TypeError):
        pass
    
    # Method 6: Time-based Analysis
    try:
        time_str = trade_data.get('created_at', '')
        if time_str and time_str != 'N/A':
            utc_time = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            ny_time = utc_time.astimezone(ZoneInfo("America/New_York"))
            hour = ny_time.hour
            
            # Market open buying pressure
            if 9 <= hour <= 10 and vol_oi_ratio > 3:
                reasoning.append("Morning session with high activity")
                confidence_scores.append(0.3)
                if preliminary_side in ["UNKNOWN", "NEUTRAL"]:
                    preliminary_side = "BUY"
            
            # End of day positioning
            elif hour >= 15 and option_type == 'P':
                reasoning.append("EOD put activity - likely hedging")
                confidence_scores.append(0.3)
    except Exception:
        pass
    
    # Calculate final confidence score
    if confidence_scores:
        # Weight recent scores more heavily
        if len(confidence_scores) > 3:
            weights = [1.0] * (len(confidence_scores) - 2) + [1.5, 2.0]
            final_confidence = np.average(confidence_scores, weights=weights[:len(confidence_scores)])
        else:
            final_confidence = np.mean(confidence_scores)
    else:
        final_confidence = 0.1
    
    # Determine final side with qualifiers
    if preliminary_side == "BUY":
        if final_confidence >= 0.8:
            final_side = "BUY (High Confidence)"
        elif final_confidence >= 0.6:
            final_side = "BUY (Medium Confidence)"
        elif final_confidence >= 0.4:
            final_side = "BUY (Low Confidence)"
        else:
            final_side = "BUY (Very Low Confidence)"
    elif preliminary_side == "SELL":
        if final_confidence >= 0.8:
            final_side = "SELL (High Confidence)"
        elif final_confidence >= 0.6:
            final_side = "SELL (Medium Confidence)"
        elif final_confidence >= 0.4:
            final_side = "SELL (Low Confidence)"
        else:
            final_side = "SELL (Very Low Confidence)"
    else:
        if vol_oi_ratio > 5:  # Fallback: high vol/oi usually means buying
            final_side = "BUY (Volume-Based)"
            final_confidence = 0.4
        else:
            final_side = "UNKNOWN"
    
    if debug:
        st.write(f"**Final Analysis**: Side={final_side}, Confidence={final_confidence:.2f}")
        st.write(f"**Reasoning**: {'; '.join(reasoning)}")
    
    return final_side, final_confidence, reasoning

def diagnose_trade_data(trades):
    """Diagnostic function to check data quality"""
    st.markdown("## 🔍 Trade Data Diagnostics")
    
    if not trades:
        st.error("No trades to diagnose!")
        return
    
    # Sample size
    st.write(f"**Total Trades**: {len(trades)}")
    
    # Data completeness
    fields_to_check = ['price', 'bid', 'ask', 'volume', 'open_interest', 'description', 'rule_name']
    completeness = {}
    
    for field in fields_to_check:
        valid_count = sum(1 for t in trades if t.get(field) not in ['N/A', '', None, 0])
        completeness[field] = valid_count / len(trades)
    
    st.write("**Data Completeness**:")
    for field, pct in completeness.items():
        color = "🟢" if pct > 0.8 else "🟡" if pct > 0.5 else "🔴"
        st.write(f"{color} {field}: {pct:.1%}")
    
    # Sample trade inspection
    st.write("**Sample Trade Data**:")
    sample = trades[0]
    important_fields = ['ticker', 'price', 'bid', 'ask', 'volume', 'open_interest', 'description', 'rule_name', 'side']
    for key in important_fields:
        if key in sample:
            st.write(f"- {key}: {sample[key]} ({type(sample[key]).__name__})")
    
    # Quick buy/sell test on sample
    st.write("**Sample Buy/Sell Analysis**:")
    side, confidence, reasoning = determine_trade_side_enhanced(sample, debug=True)

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

def analyze_open_interest(trade_data, ticker_trades):
    """
    Analyze open interest patterns for the trade
    """
    try:
        oi = float(trade_data.get('open_interest', 0))
        volume = float(trade_data.get('volume', 0))
        strike = float(trade_data.get('strike', 0))
    except (ValueError, TypeError):
        oi = volume = strike = 0
        
    option_type = trade_data.get('type', '')
    
    analysis = {
        'oi_level': 'Normal',
        'oi_change_indicator': 'Stable',
        'liquidity_score': 'Medium',
        'oi_concentration': 'Distributed'
    }
    
    # Determine OI level
    if oi > 10000:
        analysis['oi_level'] = 'Very High'
    elif oi > 5000:
        analysis['oi_level'] = 'High'
    elif oi > 1000:
        analysis['oi_level'] = 'Medium'
    elif oi > 100:
        analysis['oi_level'] = 'Low'
    else:
        analysis['oi_level'] = 'Very Low'
    
    # Volume to OI ratio analysis
    vol_oi_ratio = volume / max(oi, 1)
    if vol_oi_ratio > 5:
        analysis['oi_change_indicator'] = 'Major Increase Expected'
    elif vol_oi_ratio > 2:
        analysis['oi_change_indicator'] = 'Increase Expected'
    elif vol_oi_ratio > 0.5:
        analysis['oi_change_indicator'] = 'Moderate Activity'
    else:
        analysis['oi_change_indicator'] = 'Low Activity'
    
    # Liquidity scoring
    if oi > 5000 and volume > 100:
        analysis['liquidity_score'] = 'Excellent'
    elif oi > 1000 and volume > 50:
        analysis['liquidity_score'] = 'Good'
    elif oi > 500 and volume > 20:
        analysis['liquidity_score'] = 'Fair'
    else:
        analysis['liquidity_score'] = 'Poor'
    
    # Check for strike concentration within ticker
    try:
        same_strike_oi = sum(1 for t in ticker_trades 
                           if abs(float(t.get('strike', 0)) - strike) < 1 
                           and t.get('type') == option_type)
    except (ValueError, TypeError):
        same_strike_oi = 0
    if same_strike_oi > 3:
        analysis['oi_concentration'] = 'High Concentration'
    elif same_strike_oi > 1:
        analysis['oi_concentration'] = 'Some Concentration'
    
    return analysis

# --- NEW PATTERN RECOGNITION FUNCTIONS ---
def detect_multi_leg_strategies(ticker_trades):
    """
    Detect multi-leg option strategies like spreads, straddles, and collars
    """
    strategies = []
    
    # Group trades by ticker and time window
    time_grouped = defaultdict(list)
    for trade in ticker_trades:
        try:
            trade_time = datetime.fromisoformat(trade.get('time_utc', '').replace('Z', '+00:00'))
            time_key = int(trade_time.timestamp() // config.MULTI_LEG_TIME_WINDOW)
            time_grouped[time_key].append(trade)
        except:
            continue
    
    for time_window, trades in time_grouped.items():
        if len(trades) < 2:
            continue
            
        # Sort by strike price
        trades.sort(key=lambda x: float(x.get('strike', 0)))
        
        # Detect vertical spreads (same expiry, different strikes)
        call_trades = [t for t in trades if t.get('type') == 'C']
        put_trades = [t for t in trades if t.get('type') == 'P']
        
        # Call spreads
        if len(call_trades) >= 2:
            for i in range(len(call_trades) - 1):
                trade1, trade2 = call_trades[i], call_trades[i + 1]
                if (trade1.get('expiry') == trade2.get('expiry') and 
                    trade1.get('enhanced_side', '') != trade2.get('enhanced_side', '')):
                    
                    if 'BUY' in trade1.get('enhanced_side', '') and 'SELL' in trade2.get('enhanced_side', ''):
                        strategies.append({
                            'strategy': 'Call Debit Spread',
                            'ticker': trade1.get('ticker'),
                            'strikes': f"{trade1.get('strike'):.0f}/{trade2.get('strike'):.0f}",
                            'expiry': trade1.get('expiry'),
                            'premium': trade1.get('premium', 0) - trade2.get('premium', 0),
                            'confidence': 'High'
                        })
                    elif 'SELL' in trade1.get('enhanced_side', '') and 'BUY' in trade2.get('enhanced_side', ''):
                        strategies.append({
                            'strategy': 'Call Credit Spread',
                            'ticker': trade1.get('ticker'),
                            'strikes': f"{trade1.get('strike'):.0f}/{trade2.get('strike'):.0f}",
                            'expiry': trade1.get('expiry'),
                            'premium': trade2.get('premium', 0) - trade1.get('premium', 0),
                            'confidence': 'High'
                        })
        
        # Put spreads
        if len(put_trades) >= 2:
            for i in range(len(put_trades) - 1):
                trade1, trade2 = put_trades[i], put_trades[i + 1]
                if (trade1.get('expiry') == trade2.get('expiry') and 
                    trade1.get('enhanced_side', '') != trade2.get('enhanced_side', '')):
                    
                    if 'BUY' in trade1.get('enhanced_side', '') and 'SELL' in trade2.get('enhanced_side', ''):
                        strategies.append({
                            'strategy': 'Put Debit Spread',
                            'ticker': trade1.get('ticker'),
                            'strikes': f"{trade1.get('strike'):.0f}/{trade2.get('strike'):.0f}",
                            'expiry': trade1.get('expiry'),
                            'premium': trade1.get('premium', 0) - trade2.get('premium', 0),
                            'confidence': 'High'
                        })
        
        # Detect straddles/strangles (same strike or different strikes, same expiry)
        if len(call_trades) >= 1 and len(put_trades) >= 1:
            for call_trade in call_trades:
                for put_trade in put_trades:
                    if (call_trade.get('expiry') == put_trade.get('expiry') and
                        call_trade.get('enhanced_side', '') == put_trade.get('enhanced_side', '')):
                        
                        if abs(float(call_trade.get('strike', 0)) - float(put_trade.get('strike', 0))) < 1:
                            # Straddle
                            strategies.append({
                                'strategy': 'Long Straddle' if 'BUY' in call_trade.get('enhanced_side', '') else 'Short Straddle',
                                'ticker': call_trade.get('ticker'),
                                'strikes': f"{call_trade.get('strike'):.0f}",
                                'expiry': call_trade.get('expiry'),
                                'premium': call_trade.get('premium', 0) + put_trade.get('premium', 0),
                                'confidence': 'High'
                            })
                        elif abs(float(call_trade.get('strike', 0)) - float(put_trade.get('strike', 0))) > 1:
                            # Strangle
                            strategies.append({
                                'strategy': 'Long Strangle' if 'BUY' in call_trade.get('enhanced_side', '') else 'Short Strangle',
                                'ticker': call_trade.get('ticker'),
                                'strikes': f"{put_trade.get('strike'):.0f}/{call_trade.get('strike'):.0f}",
                                'expiry': call_trade.get('expiry'),
                                'premium': call_trade.get('premium', 0) + put_trade.get('premium', 0),
                                'confidence': 'Medium'
                            })
    
    return strategies

def detect_gamma_squeeze_indicators(ticker_trades):
    """
    Detect potential gamma squeeze conditions
    """
    gamma_indicators = []
    
    # Group by ticker
    ticker_groups = defaultdict(list)
    for trade in ticker_trades:
        ticker_groups[trade.get('ticker', '')].append(trade)
    
    for ticker, trades in ticker_groups.items():
        if len(trades) < 3:
            continue
            
        # Calculate metrics for gamma squeeze detection
        call_trades = [t for t in trades if t.get('type') == 'C']
        total_call_volume = sum(float(t.get('volume', 0)) for t in call_trades)
        total_call_oi = sum(float(t.get('open_interest', 0)) for t in call_trades)
        
        # Look for high call volume relative to OI
        if total_call_oi > 0:
            call_vol_oi_ratio = total_call_volume / total_call_oi
            
            # Check for concentrated strikes near current price
            if len(call_trades) > 0:
                avg_underlying = np.mean([float(t.get('underlying_price', 0)) for t in call_trades if t.get('underlying_price')])
                
                # Find strikes within 5% of current price
                near_money_calls = [
                    t for t in call_trades 
                    if abs(float(t.get('strike', 0)) - avg_underlying) / avg_underlying < 0.05
                ]
                
                if len(near_money_calls) >= 2 and call_vol_oi_ratio > 3:
                    total_near_money_volume = sum(float(t.get('volume', 0)) for t in near_money_calls)
                    total_near_money_premium = sum(float(t.get('premium', 0)) for t in near_money_calls)
                    
                    # Check for buying pressure
                    buy_trades = [t for t in near_money_calls if 'BUY' in t.get('enhanced_side', '')]
                    buy_ratio = len(buy_trades) / len(near_money_calls) if near_money_calls else 0
                    
                    if buy_ratio > 0.6:  # 60% or more are buys
                        gamma_indicators.append({
                            'ticker': ticker,
                            'indicator': 'Gamma Squeeze Setup',
                            'strikes': [f"{t.get('strike'):.0f}" for t in near_money_calls[:3]],
                            'total_volume': total_near_money_volume,
                            'total_premium': total_near_money_premium,
                            'vol_oi_ratio': call_vol_oi_ratio,
                            'buy_ratio': buy_ratio,
                            'confidence': 'High' if buy_ratio > 0.75 else 'Medium'
                        })
    
    return gamma_indicators

def detect_iv_spikes(ticker_trades):
    """
    Detect unusual IV spikes that may indicate upcoming events
    """
    iv_alerts = []
    
    # Group by ticker
    ticker_groups = defaultdict(list)
    for trade in ticker_trades:
        if trade.get('iv', 0) > 0:
            ticker_groups[trade.get('ticker', '')].append(trade)
    
    for ticker, trades in ticker_groups.items():
        if len(trades) < 2:
            continue
            
        # Calculate average IV for the ticker
        iv_values = [float(t.get('iv', 0)) for t in trades if t.get('iv', 0) > 0]
        if not iv_values:
            continue
            
        avg_iv = np.mean(iv_values)
        max_iv = max(iv_values)
        
        # Look for individual trades with IV significantly above average
        for trade in trades:
            trade_iv = float(trade.get('iv', 0))
            if trade_iv > avg_iv * (1 + config.IV_SPIKE_THRESHOLD):
                iv_alerts.append({
                    'ticker': ticker,
                    'strike': trade.get('strike'),
                    'type': trade.get('type'),
                    'expiry': trade.get('expiry'),
                    'iv': trade_iv,
                    'avg_iv': avg_iv,
                    'iv_premium': (trade_iv - avg_iv) / avg_iv,
                    'premium': trade.get('premium', 0),
                    'trade_side': trade.get('enhanced_side', 'UNKNOWN'),
                    'confidence': 'High' if trade_iv > avg_iv * 1.5 else 'Medium'
                })
        
        # Check for overall elevated IV across multiple strikes
        if avg_iv > config.EXTREME_IV_THRESHOLD:
            high_iv_trades = [t for t in trades if float(t.get('iv', 0)) > config.EXTREME_IV_THRESHOLD]
            if len(high_iv_trades) >= 3:
                iv_alerts.append({
                    'ticker': ticker,
                    'alert_type': 'Broad IV Elevation',
                    'avg_iv': avg_iv,
                    'max_iv': max_iv,
                    'affected_strikes': len(high_iv_trades),
                    'total_premium': sum(float(t.get('premium', 0)) for t in high_iv_trades),
                    'confidence': 'High'
                })
    
    return iv_alerts

def analyze_cross_asset_correlation(ticker_trades):
    """
    Analyze correlations between options flow and identify related movements
    """
    correlations = []
    
    # Group by sector/industry (simplified mapping)
    sector_map = {
        'SPY': 'Market',
        'QQQ': 'Tech',
        'IWM': 'Small Cap',
        'AAPL': 'Tech',
        'MSFT': 'Tech',
        'GOOGL': 'Tech',
        'AMZN': 'Tech',
        'NVDA': 'Tech',
        'JPM': 'Finance',
        'BAC': 'Finance',
        'WFC': 'Finance',
        'XOM': 'Energy',
        'CVX': 'Energy'
    }
    
    # Group trades by sector
    sector_trades = defaultdict(list)
    for trade in ticker_trades:
        ticker = trade.get('ticker', '')
        sector = sector_map.get(ticker, 'Other')
        sector_trades[sector].append(trade)
    
    # Analyze flow patterns within sectors
    for sector, trades in sector_trades.items():
        if len(trades) < 5:
            continue
            
        # Calculate sector metrics
        total_premium = sum(float(t.get('premium', 0)) for t in trades)
        call_premium = sum(float(t.get('premium', 0)) for t in trades if t.get('type') == 'C')
        put_premium = sum(float(t.get('premium', 0)) for t in trades if t.get('type') == 'P')
        
        call_ratio = call_premium / total_premium if total_premium > 0 else 0
        
        # Look for concentrated sector activity
        unique_tickers = len(set(t.get('ticker') for t in trades))
        if unique_tickers >= 3 and total_premium > 1000000:  # $1M+ across 3+ tickers
            
            # Analyze sentiment consistency
            sentiment = "Bullish" if call_ratio > 0.6 else "Bearish" if call_ratio < 0.4 else "Neutral"
            
            correlations.append({
                'sector': sector,
                'correlation_type': 'Sector Flow Concentration',
                'tickers': list(set(t.get('ticker') for t in trades))[:5],
                'total_premium': total_premium,
                'call_ratio': call_ratio,
                'sentiment': sentiment,
                'trade_count': len(trades),
                'confidence': 'High' if unique_tickers >= 5 else 'Medium'
            })
    
    return correlations

def detect_scenarios(trade, underlying_price=None, oi_analysis=None):
    scenarios = []
    opt_type = trade['type']
    try:
        strike = float(trade['strike'])
        premium = float(trade['premium'])
        volume = float(trade.get('volume', 0))
        oi = float(trade.get('open_interest', 0))
        iv = float(trade.get('iv', 0))
    except (ValueError, TypeError):
        strike = premium = volume = oi = iv = 0
        
    rule_name = trade.get('rule_name', '')
    ticker = trade['ticker']
    trade_side = trade.get('enhanced_side', 'UNKNOWN')

    if underlying_price is None:
        underlying_price = strike
    
    try:
        underlying_price = float(underlying_price)
    except (ValueError, TypeError):
        underlying_price = strike

    moneyness = "ATM"
    if opt_type == 'C' and strike > underlying_price:
        moneyness = "OTM"
    elif opt_type == 'C' and strike < underlying_price:
        moneyness = "ITM"
    elif opt_type == 'P' and strike < underlying_price:
        moneyness = "OTM"
    elif opt_type == 'P' and strike > underlying_price:
        moneyness = "ITM"

    # Enhanced scenarios with buy/sell consideration
    if opt_type == 'C' and moneyness == 'OTM' and premium >= config.SCENARIO_OTM_CALL_MIN_PREMIUM:
        if 'BUY' in trade_side:
            scenarios.append("Large OTM Call Buying")
        else:
            scenarios.append("Large OTM Call Writing")
    
    if opt_type == 'P' and moneyness == 'OTM' and premium >= config.SCENARIO_OTM_CALL_MIN_PREMIUM:
        if 'BUY' in trade_side:
            scenarios.append("Large OTM Put Buying")
        else:
            scenarios.append("Large OTM Put Writing")
    
    if moneyness == 'ITM' and premium >= config.SCENARIO_ITM_CONV_MIN_PREMIUM:
        scenarios.append("ITM Conviction Trade")
    
    # Volume/OI scenarios
    vol_oi_ratio = volume / max(oi, 1)
    if vol_oi_ratio > config.SCENARIO_SWEEP_VOLUME_OI_RATIO:
        scenarios.append("Sweep Orders")
    
    if volume >= config.SCENARIO_BLOCK_TRADE_VOL:
        scenarios.append("Block Trade")
    
    # Open Interest based scenarios
    if oi_analysis:
        if oi_analysis['oi_level'] in ['Very High', 'High'] and vol_oi_ratio > 2:
            scenarios.append("High OI + Volume Surge")
        
        if oi_analysis['liquidity_score'] == 'Poor' and premium > 200000:
            scenarios.append("Illiquid Large Trade")
        
        if oi_analysis['oi_concentration'] == 'High Concentration':
            scenarios.append("Strike Concentration Play")
    
    # Pattern-based scenarios
    if rule_name in ['RepeatedHits', 'RepeatedHitsAscendingFill']:
        scenarios.append("Repeated Buying at Same Strike")
    elif rule_name in ['RepeatedHitsDescendingFill']:
        scenarios.append("Repeated Selling at Same Strike")
    
    # Advanced scenarios
    if opt_type == 'C' and moneyness == 'OTM' and 'SELL' in trade_side and iv > config.HIGH_IV_THRESHOLD:
        scenarios.append("High IV Call Selling")
    
    if opt_type == 'P' and moneyness == 'OTM' and 'SELL' in trade_side:
        scenarios.append("Put Selling for Income")
    
    if ticker in ['SPY', 'QQQ'] and opt_type == 'P' and moneyness in ['ITM', 'ATM']:
        scenarios.append("Portfolio Hedging")
    
    # Insider-like activity detection
    if premium > 500000 and vol_oi_ratio > 10:
        scenarios.append("Potential Insider Activity")
    
    # IV-based scenarios
    if iv > config.EXTREME_IV_THRESHOLD:
        scenarios.append("Extreme IV Play")
    elif iv > config.HIGH_IV_THRESHOLD:
        scenarios.append("High IV Premium")
    
    if iv > config.IV_CRUSH_THRESHOLD and trade.get('dte', 0) <= 7:
        scenarios.append("IV Crush Risk")
    
    # Volatility trading scenarios
    if iv > config.HIGH_IV_THRESHOLD and premium > 200000:
        if 'BUY' in trade_side:
            scenarios.append("Long Volatility Strategy")
        else:
            scenarios.append("Short Volatility Strategy")

    return scenarios if scenarios else ["Normal Flow"]

def calculate_moneyness(strike, current_price):
    if current_price == 'N/A' or current_price == 0:
        return "Unknown"
    try:
        strike = float(strike)
        price = float(current_price)
        diff_percent = ((strike - price) / price) * 100
        if abs(diff_percent) < 2:
            return "ATM"
        elif diff_percent > 0:
            return f"OTM +{diff_percent:.1f}%"
        else:
            return f"ITM {diff_percent:.1f}%"
    except (ValueError, TypeError):
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
    call_premium = sum(t['premium'] for t in trades if t['type'] == 'C' and 'BUY' in t.get('enhanced_side', ''))
    put_premium = sum(t['premium'] for t in trades if t['type'] == 'P' and 'BUY' in t.get('enhanced_side', ''))
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

def generate_enhanced_alerts(trades):
    alerts = []
    for trade in trades:
        score = 0
        reasons = []

        premium = trade.get('premium', 0)
        if premium > 1000000:
            score += 4
            reasons.append("Mega Premium (>$1M)")
        elif premium > 500000:
            score += 3
            reasons.append("Massive Premium")
        elif premium > 250000:
            score += 2
            reasons.append("Large Premium")

        vol_oi_ratio = trade.get('vol_oi_ratio', 0)
        if vol_oi_ratio > 10:
            score += 3
            reasons.append("Extreme Vol/OI Ratio")
        elif vol_oi_ratio > 5:
            score += 2
            reasons.append("High Vol/OI")

        dte = trade.get('dte', 0)
        if dte <= 7 and premium > 200000:
            score += 2
            reasons.append("Short-term + Size")

        # Enhanced moneyness scoring
        moneyness = trade.get('moneyness', '')
        if "ATM" in moneyness:
            score += 2
            reasons.append("At-the-Money")
        elif "ITM" in moneyness and premium > 300000:
            score += 2
            reasons.append("Deep ITM + Size")

        # Trade side consideration with confidence
        enhanced_side = trade.get('enhanced_side', '')
        side_confidence = trade.get('side_confidence', 0)
        
        if 'High Confidence' in enhanced_side:
            score += 2
            reasons.append("High Confidence Trade Direction")
        elif 'Medium Confidence' in enhanced_side:
            score += 1
            reasons.append("Medium Confidence Trade Direction")
        
        if 'BUY' in enhanced_side and vol_oi_ratio > 5:
            score += 2
            reasons.append("High Confidence Buying + High Vol/OI")

        # Open Interest analysis
        oi_analysis = trade.get('oi_analysis', {})
        if oi_analysis.get('liquidity_score') == 'Poor' and premium > 200000:
            score += 2
            reasons.append("Illiquid Large Trade")
        
        if oi_analysis.get('oi_change_indicator') == 'Major Increase Expected':
            score += 2
            reasons.append("Major OI Increase Expected")

        # IV-based alerts
        iv = trade.get('iv', 0)
        if iv > config.EXTREME_IV_THRESHOLD:
            score += 3
            reasons.append("Extreme IV (>50%)")
        elif iv > config.HIGH_IV_THRESHOLD:
            score += 2
            reasons.append("High IV (>30%)")
        
        if iv > config.IV_CRUSH_THRESHOLD and dte <= 7:
            score += 2
            reasons.append("IV Crush Risk")

        # Scenario-based scoring
        scenarios = trade.get('scenarios', [])
        high_impact_scenarios = ['Potential Insider Activity', 'High OI + Volume Surge', 'Strike Concentration Play']
        for scenario in scenarios:
            if scenario in high_impact_scenarios:
                score += 2
                reasons.append(f"Pattern: {scenario}")

        if score >= 5:
            trade['alert_score'] = score
            trade['reasons'] = reasons
            alerts.append(trade)

    return sorted(alerts, key=lambda x: -x.get('alert_score', 0))

# --- SHORT-TERM ETF SCANNER ---
def parse_option_chain_simple(opt_str):
    """Simplified option chain parser for ETF scanner"""
    try:
        idx = next(i for i, c in enumerate(opt_str) if c.isdigit())
        ticker = opt_str[:idx]
        date_str = opt_str[idx:idx+6]
        expiry_date = date(2000 + int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6]))
        dte = (expiry_date - date.today()).days
        option_type = opt_str[idx+6]
        strike = int(opt_str[idx+7:]) / 1000
        return ticker.upper(), expiry_date.strftime('%Y-%m-%d'), dte, option_type.upper(), strike
    except Exception:
        return None, None, None, None, None

def fetch_etf_trades():
    """Fetch ETF trades specifically for SPY/QQQ/IWM with ≤7 DTE"""
    allowed_tickers = {'QQQ', 'SPY', 'IWM'}
    max_dte = 7
    
    params = {
        'limit': config.LIMIT
    }
    
    try:
        response = httpx.get(url, headers=headers, params=params, timeout=30)
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return []
        
        data = response.json().get('data', [])
        filtered_trades = []
        
        for trade in data:
            option_chain = trade.get('option_chain', '')
            ticker, expiry, dte, opt_type, strike = parse_option_chain_simple(option_chain)

            if not ticker or ticker.upper() not in allowed_tickers:
                continue
            if dte is None or dte > max_dte:
                continue

            # Time conversion
            utc_time_str = trade.get('created_at')
            ny_time_str = "N/A"
            if utc_time_str != "N/A":
                try:
                    utc_time = datetime.fromisoformat(utc_time_str.replace("Z", "+00:00"))
                    ny_time = utc_time.astimezone(ZoneInfo("America/New_York"))
                    ny_time_str = ny_time.strftime("%I:%M %p")
                except Exception:
                    ny_time_str = "N/A"

            # Safe data extraction
            try:
                premium = float(trade.get('total_premium', 0))
                volume = float(trade.get('volume', 0))
                oi = float(trade.get('open_interest', 0))
                price = trade.get('price', 'N/A')
                if price != 'N/A':
                    price = float(price)
            except (ValueError, TypeError):
                premium = volume = oi = 0
                price = 'N/A'

            trade_data = {
                'ticker': ticker,
                'type': opt_type,
                'strike': strike,
                'expiry': expiry,
                'dte': dte,
                'side': trade.get('side', 'N/A'),
                'price': price,
                'premium': premium,
                'volume': volume,
                'oi': oi,
                'vol_oi_ratio': volume / max(oi, 1),
                'time_ny': ny_time_str,
                'option': option_chain,
                'underlying_price': trade.get('underlying_price', strike),
                'rule_name': trade.get('rule_name', ''),
                'description': trade.get('description', ''),
                'moneyness': calculate_moneyness(strike, trade.get('underlying_price', strike)),
                'bid': trade.get('bid', 0),
                'ask': trade.get('ask', 0),
                'iv': trade.get('iv', 0)
            }
            
            # Add enhanced trade side detection
            enhanced_side, confidence, reasoning = determine_trade_side_enhanced(trade)
            trade_data['enhanced_side'] = enhanced_side
            trade_data['side_confidence'] = confidence
            trade_data['side_reasoning'] = reasoning
            
            filtered_trades.append(trade_data)
        
        return filtered_trades

    except Exception as e:
        st.error(f"Error fetching ETF trades: {e}")
        return []

# --- FETCH FUNCTION ---
def fetch_general_flow():
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
        data = response.json().get('data', [])
        result = []
        ticker_data = defaultdict(list)

        for trade in data:
            option_chain = trade.get('option_chain', '')
            ticker, expiry, dte, opt_type, strike = parse_option_chain(option_chain)

            if not ticker or ticker in config.EXCLUDE_TICKERS:
                continue

            premium = float(trade.get('total_premium', 0))
            if premium < config.MIN_PREMIUM:
                continue

            utc_time_str = trade.get('created_at')
            ny_time_str = "N/A"
            if utc_time_str != "N/A":
                try:
                    utc_time = datetime.fromisoformat(utc_time_str.replace("Z", "+00:00"))
                    ny_time = utc_time.astimezone(ZoneInfo("America/New_York"))
                    ny_time_str = ny_time.strftime("%I:%M %p")
                except Exception:
                    ny_time_str = "N/A"

            # Extract IV data
            iv = 0
            iv_fields = ['iv', 'implied_volatility', 'volatility', 'impliedVolatility', 'vol', 'IV']
            for field in iv_fields:
                if field in trade and trade[field] not in ['N/A', '', None, 0]:
                    try:
                        iv = float(trade[field])
                        if iv > 0:
                            break
                    except (ValueError, TypeError):
                        continue

            trade_data = {
                'ticker': ticker,
                'option': option_chain,
                'type': opt_type,
                'strike': strike,
                'expiry': expiry,
                'dte': dte,
                'price': trade.get('price', 'N/A'),
                'premium': premium,
                'volume': trade.get('volume', 0),
                'open_interest': trade.get('open_interest', 0),
                'time_utc': utc_time_str,
                'time_ny': ny_time_str,
                'rule_name': trade.get('rule_name', ''),
                'description': trade.get('description', ''),
                'underlying_price': trade.get('underlying_price', strike),
                'moneyness': calculate_moneyness(strike, trade.get('underlying_price', strike)),
                'vol_oi_ratio': float(trade.get('volume', 0)) / max(float(trade.get('open_interest', 1)), 1),
                'iv': iv,
                'iv_percentage': f"{iv:.1%}" if iv > 0 else "N/A",
                'bid': float(trade.get('bid', 0)) if trade.get('bid') not in ['N/A', '', None] else 0,
                'ask': float(trade.get('ask', 0)) if trade.get('ask') not in ['N/A', '', None] else 0
            }
            
            # Add enhanced trade side detection
            enhanced_side, confidence, reasoning = determine_trade_side_enhanced(trade)
            trade_data['enhanced_side'] = enhanced_side
            trade_data['side_confidence'] = confidence
            trade_data['side_reasoning'] = reasoning

            ticker_data[ticker].append(trade_data)

        # Process each ticker's trades
        for ticker, trade_list in ticker_data.items():
            atm_calls = [t['strike'] for t in trade_list if t['type'] == 'C']
            avg_underlying_price = sum(atm_calls) / len(atm_calls) if atm_calls else None

            for trade in trade_list:
                underlying_price = avg_underlying_price if avg_underlying_price is not None else trade['strike']
                
                # Analyze open interest
                oi_analysis = analyze_open_interest(trade, trade_list)
                trade['oi_analysis'] = oi_analysis
                
                # Detect scenarios with enhanced analysis
                scenarios = detect_scenarios(trade, underlying_price, oi_analysis)
                trade['scenarios'] = scenarios
                result.append(trade)

        return result

    except Exception as e:
        st.error(f"Error fetching general flow: {e}")
        return []

# --- FILTER FUNCTIONS ---
def apply_premium_filter(trades, premium_range):
    if premium_range == "All Premiums (No Filter)":
        return trades
    
    filtered_trades = []
    for trade in trades:
        premium = trade.get('premium', 0)
        
        if premium_range == "Under $100K" and premium < 100000:
            filtered_trades.append(trade)
        elif premium_range == "Under $250K" and premium < 250000:
            filtered_trades.append(trade)
        elif premium_range == "$100K - $250K" and 100000 <= premium < 250000:
            filtered_trades.append(trade)
        elif premium_range == "$250K - $500K" and 250000 <= premium < 500000:
            filtered_trades.append(trade)
        elif premium_range == "Above $250K" and premium >= 250000:
            filtered_trades.append(trade)
        elif premium_range == "Above $500K" and premium >= 500000:
            filtered_trades.append(trade)
        elif premium_range == "Above $1M" and premium >= 1000000:
            filtered_trades.append(trade)
    
    return filtered_trades

def apply_dte_filter(trades, dte_filter):
    if dte_filter == "All DTE":
        return trades
    
    filtered_trades = []
    for trade in trades:
        dte = trade.get('dte', 0)
        
        if dte_filter == "0DTE Only" and dte == 0:
            filtered_trades.append(trade)
        elif dte_filter == "Weekly (≤7d)" and dte <= 7:
            filtered_trades.append(trade)
        elif dte_filter == "Monthly (≤30d)" and dte <= 30:
            filtered_trades.append(trade)
        elif dte_filter == "Quarterly (≤90d)" and dte <= 90:
            filtered_trades.append(trade)
        elif dte_filter == "LEAPS (>90d)" and dte > 90:
            filtered_trades.append(trade)
    
    return filtered_trades

def apply_trade_side_filter(trades, side_filter):
    if side_filter == "All Trades":
        return trades
    
    filtered_trades = []
    for trade in trades:
        enhanced_side = trade.get('enhanced_side', 'UNKNOWN')
        
        if side_filter == "Buy Only" and 'BUY' in enhanced_side:
            filtered_trades.append(trade)
        elif side_filter == "Sell Only" and 'SELL' in enhanced_side:
            filtered_trades.append(trade)
        elif side_filter == "High Confidence Only" and 'High Confidence' in enhanced_side:
            filtered_trades.append(trade)
        elif side_filter == "Medium+ Confidence" and ('High Confidence' in enhanced_side or 'Medium Confidence' in enhanced_side):
            filtered_trades.append(trade)
    
    return filtered_trades

# --- DISPLAY FUNCTIONS ---
def display_enhanced_summary(trades):
    st.markdown("### 📊 Enhanced Market Summary")
    
    if not trades:
        st.warning("No trades to analyze")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_ratio, sentiment_label = calculate_sentiment_score(trades)
        st.metric("Market Sentiment", sentiment_label, f"{sentiment_ratio:.1%} calls")
    
    with col2:
        total_premium = sum(t.get('premium', 0) for t in trades)
        st.metric("Total Premium", f"${total_premium:,.0f}")
    
    with col3:
        buy_trades = len([t for t in trades if 'BUY' in t.get('enhanced_side', '')])
        sell_trades = len([t for t in trades if 'SELL' in t.get('enhanced_side', '')])
        st.metric("Buy vs Sell", f"{buy_trades}/{sell_trades}")
    
    with col4:
        high_conf_trades = len([t for t in trades if 'High Confidence' in t.get('enhanced_side', '')])
        st.metric("High Confidence", high_conf_trades)
    
    # Enhanced buy/sell confidence analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_confidence = np.mean([t.get('side_confidence', 0) for t in trades]) if trades else 0
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    with col2:
        medium_plus_conf = len([t for t in trades if t.get('side_confidence', 0) >= 0.6])
        st.metric("Medium+ Confidence", medium_plus_conf)
    
    with col3:
        unknown_trades = len([t for t in trades if 'UNKNOWN' in t.get('enhanced_side', '')])
        st.metric("Unknown Direction", unknown_trades)

def display_pattern_recognition_analysis(trades):
    """Display advanced pattern recognition results"""
    st.markdown("### 🔍 Advanced Pattern Recognition")
    
    if not trades:
        st.info("No trades available for pattern analysis")
        return
    
    # Group trades by ticker for pattern analysis
    ticker_groups = defaultdict(list)
    for trade in trades:
        ticker_groups[trade.get('ticker', '')].append(trade)
    
    # Multi-leg strategies
    st.markdown("#### 🎯 Multi-Leg Strategy Detection")
    all_strategies = []
    for ticker, ticker_trades in ticker_groups.items():
        strategies = detect_multi_leg_strategies(ticker_trades)
        all_strategies.extend(strategies)
    
    if all_strategies:
        strategy_data = []
        for strategy in all_strategies[:10]:  # Show top 10 strategies
            strategy_data.append({
                'Strategy': strategy['strategy'],
                'Ticker': strategy['ticker'],
                'Strikes': strategy['strikes'],
                'Expiry': strategy['expiry'],
                'Net Premium': f"${strategy['premium']:,.0f}",
                'Confidence': strategy['confidence']
            })
        
        df = pd.DataFrame(strategy_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No multi-leg strategies detected")
    
    # Gamma squeeze indicators
    st.markdown("#### ⚡ Gamma Squeeze Indicators")
    gamma_indicators = []
    for ticker, ticker_trades in ticker_groups.items():
        indicators = detect_gamma_squeeze_indicators(ticker_trades)
        gamma_indicators.extend(indicators)
    
    if gamma_indicators:
        gamma_data = []
        for indicator in gamma_indicators[:5]:  # Show top 5 gamma indicators
            gamma_data.append({
                'Ticker': indicator['ticker'],
                'Indicator': indicator['indicator'],
                'Key Strikes': ', '.join(indicator['strikes']),
                'Volume': f"{indicator['total_volume']:,.0f}",
                'Premium': f"${indicator['total_premium']:,.0f}",
                'Vol/OI Ratio': f"{indicator['vol_oi_ratio']:.1f}",
                'Buy Ratio': f"{indicator['buy_ratio']:.1%}",
                'Confidence': indicator['confidence']
            })
        
        df = pd.DataFrame(gamma_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No gamma squeeze indicators detected")
    
    # IV spike analysis
    st.markdown("#### 📈 Unusual IV Spikes")
    iv_alerts = []
    for ticker, ticker_trades in ticker_groups.items():
        alerts = detect_iv_spikes(ticker_trades)
        iv_alerts.extend(alerts)
    
    if iv_alerts:
        iv_data = []
        for alert in iv_alerts[:10]:  # Show top 10 IV alerts
            if 'alert_type' in alert:
                # Broad IV elevation
                iv_data.append({
                    'Ticker': alert['ticker'],
                    'Alert Type': alert['alert_type'],
                    'Avg IV': f"{alert['avg_iv']:.1%}",
                    'Max IV': f"{alert['max_iv']:.1%}",
                    'Affected Strikes': alert['affected_strikes'],
                    'Total Premium': f"${alert['total_premium']:,.0f}",
                    'Confidence': alert['confidence']
                })
            else:
                # Individual spike
                iv_data.append({
                    'Ticker': alert['ticker'],
                    'Strike': f"${alert['strike']:.0f}",
                    'Type': alert['type'],
                    'Expiry': alert['expiry'],
                    'IV': f"{alert['iv']:.1%}",
                    'IV Premium': f"{alert['iv_premium']:.1%}",
                    'Trade Premium': f"${alert['premium']:,.0f}",
                    'Side': alert['trade_side'],
                    'Confidence': alert['confidence']
                })
        
        df = pd.DataFrame(iv_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No unusual IV spikes detected")
    
    # Cross-asset correlations
    st.markdown("#### 🔗 Cross-Asset Correlations")
    correlations = analyze_cross_asset_correlation(trades)
    
    if correlations:
        corr_data = []
        for corr in correlations[:8]:  # Show top 8 correlations
            if corr['correlation_type'] == 'Sector Flow Concentration':
                corr_data.append({
                    'Type': corr['correlation_type'],
                    'Sector': corr['sector'],
                    'Tickers': ', '.join(corr['tickers']),
                    'Total Premium': f"${corr['total_premium']:,.0f}",
                    'Call Ratio': f"{corr['call_ratio']:.1%}",
                    'Sentiment': corr['sentiment'],
                    'Trade Count': corr['trade_count'],
                    'Confidence': corr['confidence']
                })
        
        df = pd.DataFrame(corr_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No significant cross-asset correlations detected")

def display_enhanced_buy_sell_analysis(trades):
    """
    Display enhanced buy/sell analysis with debugging information
    """
    st.markdown("### 🔍 Enhanced Buy/Sell Analysis")
    
    if not trades:
        st.warning("No trades to analyze")
        return
    
    # Calculate enhanced statistics
    buy_trades = [t for t in trades if 'BUY' in t.get('enhanced_side', '')]
    sell_trades = [t for t in trades if 'SELL' in t.get('enhanced_side', '')]
    unknown_trades = [t for t in trades if 'UNKNOWN' in t.get('enhanced_side', '')]
    high_conf_trades = [t for t in trades if t.get('side_confidence', 0) >= 0.7]
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Buy Trades", len(buy_trades))
        buy_premium = sum(t.get('premium', 0) for t in buy_trades)
        st.write(f"💰 ${buy_premium:,.0f}")
    
    with col2:
        st.metric("Sell Trades", len(sell_trades))
        sell_premium = sum(t.get('premium', 0) for t in sell_trades)
        st.write(f"💰 ${sell_premium:,.0f}")
    
    with col3:
        st.metric("High Confidence", len(high_conf_trades))
        avg_confidence = np.mean([t.get('side_confidence', 0) for t in trades]) if trades else 0
        st.write(f"📊 {avg_confidence:.1%} avg")
    
    with col4:
        st.metric("Unknown/Low Conf", len(unknown_trades))
        low_conf_trades = [t for t in trades if t.get('side_confidence', 0) < 0.4]
        st.write(f"⚠️ {len(low_conf_trades)} low conf")
    
    # Display confidence distribution
    st.markdown("#### 📊 Confidence Distribution")
    confidence_ranges = {
        'Very High (80%+)': len([t for t in trades if t.get('side_confidence', 0) >= 0.8]),
        'High (60-79%)': len([t for t in trades if 0.6 <= t.get('side_confidence', 0) < 0.8]),
        'Medium (40-59%)': len([t for t in trades if 0.4 <= t.get('side_confidence', 0) < 0.6]),
        'Low (20-39%)': len([t for t in trades if 0.2 <= t.get('side_confidence', 0) < 0.4]),
        'Very Low (<20%)': len([t for t in trades if t.get('side_confidence', 0) < 0.2])
    }
    
    col1, col2 = st.columns(2)
    with col1:
        for range_name, count in confidence_ranges.items():
            pct = count / len(trades) * 100 if trades else 0
            st.write(f"**{range_name}**: {count} trades ({pct:.1f}%)")
    
    with col2:
        # Key insights
        st.markdown("**💡 Key Insights:**")
        total_trades = len(trades)
        buy_ratio = len(buy_trades) / total_trades if total_trades > 0 else 0
        high_conf_ratio = len(high_conf_trades) / total_trades if total_trades > 0 else 0
        
        if buy_ratio > 0.7:
            st.write("• 🟢 Strong buying pressure detected")
        elif buy_ratio < 0.3:
            st.write("• 🔴 Strong selling pressure detected")
        else:
            st.write("• ⚪ Mixed buy/sell activity")
        
        if high_conf_ratio > 0.5:
            st.write("• ✅ High quality trade direction data")
        elif high_conf_ratio < 0.2:
            st.write("• ⚠️ Limited trade direction clarity")
        
        if avg_confidence < 0.4:
            st.write("• 🔧 Consider data source improvements")
    
    # Debug mode toggle
    debug_mode = st.checkbox("🔧 Enable Debug Mode (shows detailed reasoning)")
    
    # Detailed trade table
    st.markdown("#### 📊 Enhanced Trade Analysis")
    
    # Create enhanced table
    table_data = []
    for trade in trades[:50]:  # Show top 50 trades
        enhanced_side = trade.get('enhanced_side', 'UNKNOWN')
        confidence = trade.get('side_confidence', 0)
        reasoning = trade.get('side_reasoning', [])
        
        # Color coding for confidence
        if confidence >= 0.7:
            confidence_color = "🟢"
        elif confidence >= 0.4:
            confidence_color = "🟡"
        else:
            confidence_color = "🔴"
        
        # Side color coding
        if 'BUY' in enhanced_side:
            side_emoji = "🟢"
        elif 'SELL' in enhanced_side:
            side_emoji = "🔴"
        else:
            side_emoji = "⚪"
        
        table_data.append({
            'Ticker': trade.get('ticker', ''),
            'Type': trade.get('type', ''),
            'Strike': f"${trade.get('strike', 0):.0f}",
            'Side': f"{side_emoji} {enhanced_side}",
            'Confidence': f"{confidence_color} {confidence:.1%}",
            'Premium': f"${trade.get('premium', 0):,.0f}",
            'Vol/OI': f"{trade.get('vol_oi_ratio', 0):.1f}",
            'Price': f"${trade.get('price', 0):.2f}" if trade.get('price', 0) != 'N/A' else 'N/A',
            'Bid': f"${trade.get('bid', 0):.2f}" if trade.get('bid', 0) > 0 else 'N/A',
            'Ask': f"${trade.get('ask', 0):.2f}" if trade.get('ask', 0) > 0 else 'N/A',
            'Top Reason': reasoning[0] if reasoning else 'No clear signal',
            'Time': trade.get('time_ny', 'N/A')
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True)
    
    # Sample trade analysis in debug mode
    if debug_mode and trades:
        st.markdown("#### 🔍 Sample Trade Analysis")
        sample_trade = trades[0]  # Analyze first trade as example
        
        st.write("**Sample Trade Data:**")
        debug_fields = ['ticker', 'type', 'strike', 'price', 'bid', 'ask', 'volume', 'open_interest', 'description', 'rule_name']
        sample_data = {k: v for k, v in sample_trade.items() if k in debug_fields}
        st.json(sample_data)
        
        side, confidence, reasoning = determine_trade_side_enhanced(sample_trade, debug=True)
        
        st.write(f"**Result**: {side} (Confidence: {confidence:.1%})")
        st.write(f"**Reasoning**: {'; '.join(reasoning)}")
    
    # Show trades with poor confidence for debugging
    low_confidence_trades = [t for t in trades if t.get('side_confidence', 0) < 0.4]
    if low_confidence_trades:
        st.markdown("#### ⚠️ Low Confidence Trades Analysis")
        st.write(f"Found {len(low_confidence_trades)} trades with confidence < 40%")
        
        # Analyze reasons for low confidence
        no_bid_ask = len([t for t in low_confidence_trades if t.get('bid', 0) == 0 or t.get('ask', 0) == 0])
        no_price = len([t for t in low_confidence_trades if t.get('price', 'N/A') == 'N/A'])
        low_vol_oi = len([t for t in low_confidence_trades if t.get('vol_oi_ratio', 0) < 0.5])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Missing Bid/Ask", no_bid_ask)
        with col2:
            st.metric("Missing Price", no_price)
        with col3:
            st.metric("Low Vol/OI", low_vol_oi)
        
        if st.button("Show Sample Low Confidence Trades"):
            for i, trade in enumerate(low_confidence_trades[:5]):
                with st.expander(f"Trade {i+1}: {trade.get('ticker')} {trade.get('strike'):.0f}{trade.get('type')}"):
                    st.write(f"**Confidence**: {trade.get('side_confidence', 0):.1%}")
                    st.write(f"**Reasoning**: {'; '.join(trade.get('side_reasoning', []))}")
                    st.write(f"**Available Data**: Price={trade.get('price')}, Bid={trade.get('bid')}, Ask={trade.get('ask')}")
                    st.write(f"**Volume/OI**: {trade.get('volume')}/{trade.get('open_interest')} (Ratio: {trade.get('vol_oi_ratio', 0):.1f})")

def display_main_trades_table(trades, title="📋 Main Trades Analysis"):
    st.markdown(f"### {title}")
    
    if not trades:
        st.info("No trades found")
        return
    
    # Separate calls and puts
    calls = [t for t in trades if t['type'] == 'C']
    puts = [t for t in trades if t['type'] == 'P']
    
    def create_trade_table(trade_list, trade_type_emoji, trade_type_name):
        if not trade_list:
            st.info(f"No {trade_type_name.lower()} found")
            return
        
        # Sort by premium descending
        sorted_trades = sorted(trade_list, key=lambda x: x.get('premium', 0), reverse=True)
        
        table_data = []
        for trade in sorted_trades[:25]:  # Show top 25 per section
            oi_analysis = trade.get('oi_analysis', {})
            enhanced_side = trade.get('enhanced_side', 'UNKNOWN')
            confidence = trade.get('side_confidence', 0)
            
            # Side display with confidence indicator
            if 'BUY' in enhanced_side:
                side_display = f"🟢 {enhanced_side}"
            elif 'SELL' in enhanced_side:
                side_display = f"🔴 {enhanced_side}"
            else:
                side_display = f"⚪ {enhanced_side}"
            
            # Confidence indicator
            if confidence >= 0.7:
                conf_indicator = "🟢"
            elif confidence >= 0.4:
                conf_indicator = "🟡"
            else:
                conf_indicator = "🔴"
            
            table_data.append({
                'Ticker': trade['ticker'],
                'Side': side_display,
                'Conf': f"{conf_indicator} {confidence:.0%}",
                'Strike': f"${trade['strike']:.0f}",
                'Expiry': trade['expiry'],
                'DTE': trade['dte'],
                'Price': f"${trade['price']}" if trade['price'] != 'N/A' else 'N/A',
                'Premium': f"${trade['premium']:,.0f}",
                'Volume': f"{trade['volume']:,}",
                'Open Interest': f"{trade['open_interest']:,}",
                'Vol/OI': f"{trade['vol_oi_ratio']:.1f}",
                'OI Level': oi_analysis.get('oi_level', 'N/A'),
                'Liquidity': oi_analysis.get('liquidity_score', 'N/A'),
                'IV': trade['iv_percentage'],
                'Moneyness': trade['moneyness'],
                'Primary Scenario': trade.get('scenarios', ['Normal Flow'])[0],
                'Time': trade['time_ny']
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    
    # Display in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🟢 CALLS")
        create_trade_table(calls, "🟢", "Calls")
    
    with col2:
        st.markdown("#### 🔴 PUTS")
        create_trade_table(puts, "🔴", "Puts")
    
    # Add Short-Term ETF section after calls/puts
    st.divider()
    display_short_term_etf_section(trades)

def display_short_term_etf_section(all_trades):
    """Display short-term ETF section as part of main analysis"""
    st.markdown("### ⚡ Short-Term ETF Focus (SPY/QQQ/IWM ≤ 7 DTE)")
    
    # Filter for short-term ETF trades
    allowed_tickers = {'QQQ', 'SPY', 'IWM'}
    max_dte = 7
    
    etf_trades = [
        t for t in all_trades 
        if t['ticker'] in allowed_tickers and t.get('dte', 0) <= max_dte
    ]
    
    if not etf_trades:
        st.info("No short-term ETF trades found in current dataset")
        return
    
    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_premium = sum(t.get('premium', 0) for t in etf_trades)
        st.metric("ETF Premium", f"${total_premium:,.0f}")
    
    with col2:
        zero_dte = len([t for t in etf_trades if t.get('dte', 0) == 0])
        st.metric("0DTE Trades", zero_dte)
    
    with col3:
        buy_trades = len([t for t in etf_trades if 'BUY' in t.get('enhanced_side', '')])
        sell_trades = len([t for t in etf_trades if 'SELL' in t.get('enhanced_side', '')])
        st.metric("Buy/Sell", f"{buy_trades}/{sell_trades}")
    
    with col4:
        avg_confidence = np.mean([t.get('side_confidence', 0) for t in etf_trades]) if etf_trades else 0
        st.metric("Avg Confidence", f"{avg_confidence:.0%}")
    
    # Create ETF table
    def create_etf_summary_table(trades):
        if not trades:
            return
        
        # Sort by premium descending
        sorted_trades = sorted(trades, key=lambda x: x.get('premium', 0), reverse=True)
        
        table_data = []
        for trade in sorted_trades[:15]:  # Top 15 ETF trades
            enhanced_side = trade.get('enhanced_side', 'UNKNOWN')
            confidence = trade.get('side_confidence', 0)
            
            # Side display with emoji
            if 'BUY' in enhanced_side:
                side_display = f"🟢 {enhanced_side}"
            elif 'SELL' in enhanced_side:
                side_display = f"🔴 {enhanced_side}"
            else:
                side_display = f"⚪ {enhanced_side}"
            
            table_data.append({
                'Ticker': trade['ticker'],
                'Type': trade['type'],
                'Side': side_display,
                'Conf': f"{confidence:.0%}",
                'Strike': f"${trade['strike']:.0f}",
                'DTE': trade.get('dte', 0),
                'Premium': f"${trade.get('premium', 0):,.0f}",
                'Volume': f"{trade.get('volume', 0):,}",
                'OI': f"{trade.get('open_interest', 0):,}",
                'Vol/OI': f"{trade.get('vol_oi_ratio', 0):.1f}",
                'Moneyness': trade.get('moneyness', 'N/A'),
                'Primary Scenario': trade.get('scenarios', ['Normal Flow'])[0],
                'Time': trade.get('time_ny', 'N/A')
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    
    create_etf_summary_table(etf_trades)
    
    # Most active strikes
    strike_activity = {}
    for trade in etf_trades:
        key = f"{trade['ticker']} ${trade['strike']:.0f}{trade['type']}"
        if key not in strike_activity:
            strike_activity[key] = {'premium': 0, 'volume': 0, 'count': 0}
        strike_activity[key]['premium'] += trade.get('premium', 0)
        strike_activity[key]['volume'] += trade.get('volume', 0)
        strike_activity[key]['count'] += 1
    
    if strike_activity:
        top_strikes = sorted(strike_activity.items(), 
                           key=lambda x: x[1]['premium'], reverse=True)[:5]
        
        st.markdown("#### 🎯 Most Active ETF Strikes:")
        for i, (strike_key, data) in enumerate(top_strikes, 1):
            st.write(f"**{i}. {strike_key}** - ${data['premium']:,.0f} premium, "
                    f"{data['volume']:,.0f} volume ({data['count']} trades)")

def display_etf_scanner(trades):
    """Display the dedicated ETF scanner section"""
    st.markdown("### ⚡ ETF Flow Scanner (SPY/QQQ/IWM ≤ 7 DTE)")
    
    if not trades:
        st.warning("No ETF trades found")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_premium = sum(t['premium'] for t in trades)
        st.metric("Total Premium", f"${total_premium:,.0f}")
    
    with col2:
        zero_dte = len([t for t in trades if t['dte'] == 0])
        st.metric("0DTE Trades", zero_dte)
    
    with col3:
        buy_trades = len([t for t in trades if 'BUY' in t.get('enhanced_side', '')])
        sell_trades = len([t for t in trades if 'SELL' in t.get('enhanced_side', '')])
        st.metric("Buy/Sell", f"{buy_trades}/{sell_trades}")
    
    with col4:
        avg_confidence = np.mean([t.get('side_confidence', 0) for t in trades]) if trades else 0
        st.metric("Avg Confidence", f"{avg_confidence:.0%}")
    
    # Separate by ETF
    spy_trades = [t for t in trades if t['ticker'] == 'SPY']
    qqq_trades = [t for t in trades if t['ticker'] == 'QQQ']
    iwm_trades = [t for t in trades if t['ticker'] == 'IWM']
    
    def create_etf_table(ticker_trades, ticker_name):
        if not ticker_trades:
            st.info(f"No {ticker_name} trades found")
            return
        
        # Sort by premium descending
        sorted_trades = sorted(ticker_trades, key=lambda x: x['premium'], reverse=True)
        
        table_data = []
        for trade in sorted_trades[:20]:  # Top 20 per ETF
            enhanced_side = trade.get('enhanced_side', 'UNKNOWN')
            confidence = trade.get('side_confidence', 0)
            
            # Side display with confidence
            if 'BUY' in enhanced_side:
                side_display = f"🟢 {enhanced_side}"
            elif 'SELL' in enhanced_side:
                side_display = f"🔴 {enhanced_side}"
            else:
                side_display = f"⚪ {enhanced_side}"
            
            table_data.append({
                'Type': trade['type'],
                'Side': side_display,
                'Conf': f"{confidence:.0%}",
                'Strike': f"${trade['strike']:.0f}",
                'DTE': trade['dte'],
                'Price': f"${trade['price']:.2f}" if trade['price'] != 'N/A' else 'N/A',
                'Premium': f"${trade['premium']:,.0f}",
                'Volume': f"{trade['volume']:,.0f}",
                'OI': f"{trade['oi']:,.0f}",
                'Vol/OI': f"{trade['vol_oi_ratio']:.1f}",
                'Moneyness': trade['moneyness'],
                'Time': trade['time_ny'],
                'Rule': trade.get('rule_name', 'N/A')
            })
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    
    # Display each ETF in tabs
    tab1, tab2, tab3 = st.tabs(["🕷️ SPY", "🔷 QQQ", "🔸 IWM"])
    
    with tab1:
        st.markdown("#### SPY Short-Term Flow")
        spy_premium = sum(t['premium'] for t in spy_trades)
        spy_count = len(spy_trades)
        spy_buy_count = len([t for t in spy_trades if 'BUY' in t.get('enhanced_side', '')])
        st.write(f"**{spy_count} trades | ${spy_premium:,.0f} premium | {spy_buy_count} buys**")
        create_etf_table(spy_trades, "SPY")
    
    with tab2:
        st.markdown("#### QQQ Short-Term Flow")
        qqq_premium = sum(t['premium'] for t in qqq_trades)
        qqq_count = len(qqq_trades)
        qqq_buy_count = len([t for t in qqq_trades if 'BUY' in t.get('enhanced_side', '')])
        st.write(f"**{qqq_count} trades | ${qqq_premium:,.0f} premium | {qqq_buy_count} buys**")
        create_etf_table(qqq_trades, "QQQ")
    
    with tab3:
        st.markdown("#### IWM Short-Term Flow")
        iwm_premium = sum(t['premium'] for t in iwm_trades)
        iwm_count = len(iwm_trades)
        iwm_buy_count = len([t for t in iwm_trades if 'BUY' in t.get('enhanced_side', '')])
        st.write(f"**{iwm_count} trades | ${iwm_premium:,.0f} premium | {iwm_buy_count} buys**")
        create_etf_table(iwm_trades, "IWM")
    
    # Key insights
    st.markdown("#### 🔍 Key ETF Insights")
    
    # Most active strikes
    strike_activity = {}
    for trade in trades:
        key = f"{trade['ticker']} ${trade['strike']:.0f}{trade['type']}"
        if key not in strike_activity:
            strike_activity[key] = {'count': 0, 'total_premium': 0, 'total_volume': 0, 'buy_count': 0}
        strike_activity[key]['count'] += 1
        strike_activity[key]['total_premium'] += trade['premium']
        strike_activity[key]['total_volume'] += trade['volume']
        if 'BUY' in trade.get('enhanced_side', ''):
            strike_activity[key]['buy_count'] += 1
    
    # Sort by total premium
    top_strikes = sorted(strike_activity.items(), 
                        key=lambda x: x[1]['total_premium'], reverse=True)[:8]
    
    if top_strikes:
        st.markdown("**🎯 Most Active ETF Strikes by Premium:**")
        col1, col2 = st.columns(2)
        
        for i, (strike_key, data) in enumerate(top_strikes):
            col = col1 if i % 2 == 0 else col2
            buy_ratio = data['buy_count'] / data['count'] if data['count'] > 0 else 0
            sentiment_emoji = "🟢" if buy_ratio > 0.6 else "🔴" if buy_ratio < 0.4 else "⚪"
            
            with col:
                st.write(f"**{strike_key}** {sentiment_emoji}")
                st.write(f"💰 ${data['total_premium']:,.0f} | 📊 {data['total_volume']:,.0f} vol")
                st.write(f"🔄 {data['count']} trades | {buy_ratio:.0%} buys")
    
    # 0DTE focus
    zero_dte_trades = [t for t in trades if t['dte'] == 0]
    if zero_dte_trades:
        st.markdown("#### ⚡ 0DTE Spotlight")
        zero_dte_premium = sum(t['premium'] for t in zero_dte_trades)
        zero_dte_buys = len([t for t in zero_dte_trades if 'BUY' in t.get('enhanced_side', '')])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("0DTE Total Premium", f"${zero_dte_premium:,.0f}")
        with col2:
            st.metric("0DTE Buy Trades", zero_dte_buys)
        
        # Top 0DTE trades
        top_0dte = sorted(zero_dte_trades, key=lambda x: x['premium'], reverse=True)[:5]
        st.markdown("**Top 0DTE Trades:**")
        for i, trade in enumerate(top_0dte, 1):
            enhanced_side = trade.get('enhanced_side', 'UNKNOWN')
            confidence = trade.get('side_confidence', 0)
            
            if 'BUY' in enhanced_side:
                side_indicator = "🟢"
            elif 'SELL' in enhanced_side:
                side_indicator = "🔴"
            else:
                side_indicator = "⚪"
                
            conf_indicator = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.4 else "🔴"
            
            st.write(f"{i}. {side_indicator} {trade['ticker']} {trade['strike']:.0f}{trade['type']} - "
                    f"${trade['premium']:,.0f} ({enhanced_side}) {conf_indicator}")

def display_open_interest_analysis(trades):
    st.markdown("### 📈 Open Interest Deep Dive")
    
    if not trades:
        st.info("No data available")
        return
    
    # OI Level Distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### OI Level Summary")
        oi_levels = {}
        for trade in trades:
            level = trade.get('oi_analysis', {}).get('oi_level', 'Unknown')
            oi_levels[level] = oi_levels.get(level, 0) + 1
        
        for level, count in sorted(oi_levels.items()):
            st.write(f"**{level}**: {count} trades")
    
    with col2:
        st.markdown("#### Liquidity Analysis")
        liquidity_scores = {}
        for trade in trades:
            score = trade.get('oi_analysis', {}).get('liquidity_score', 'Unknown')
            liquidity_scores[score] = liquidity_scores.get(score, 0) + 1
        
        for score, count in sorted(liquidity_scores.items()):
            st.write(f"**{score}**: {count} trades")
    
    # High OI Concentration Trades
    st.markdown("#### 🎯 High OI Concentration Plays")
    concentration_trades = [
        t for t in trades 
        if t.get('oi_analysis', {}).get('oi_concentration') == 'High Concentration'
    ]
    
    if concentration_trades:
        conc_data = []
        for trade in sorted(concentration_trades, key=lambda x: x.get('premium', 0), reverse=True)[:10]:
            enhanced_side = trade.get('enhanced_side', 'UNKNOWN')
            confidence = trade.get('side_confidence', 0)
            
            side_display = f"🟢 {enhanced_side}" if 'BUY' in enhanced_side else f"🔴 {enhanced_side}" if 'SELL' in enhanced_side else f"⚪ {enhanced_side}"
            
            conc_data.append({
                'Ticker': trade['ticker'],
                'Strike': f"${trade['strike']:.0f}",
                'Type': trade['type'],
                'Side': side_display,
                'Conf': f"{confidence:.0%}",
                'Premium': f"${trade['premium']:,.0f}",
                'OI': f"{trade['open_interest']:,}",
                'Volume': f"{trade['volume']:,}",
                'Primary Scenario': trade.get('scenarios', ['Normal Flow'])[0]
            })
        
        st.dataframe(pd.DataFrame(conc_data), use_container_width=True)
    else:
        st.info("No high concentration plays found")

def display_enhanced_alerts(trades):
    alerts = generate_enhanced_alerts(trades)
    if not alerts:
        st.info("No high-priority alerts found")
        return
    
    st.markdown("### 🚨 Enhanced Priority Alerts")
    
    # Alert summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Alerts", len(alerts))
    with col2:
        avg_score = np.mean([a.get('alert_score', 0) for a in alerts])
        st.metric("Avg Alert Score", f"{avg_score:.1f}")
    with col3:
        high_conf_alerts = len([a for a in alerts if 'High Confidence' in a.get('enhanced_side', '')])
        st.metric("High Conf Alerts", high_conf_alerts)
    
    for i, alert in enumerate(alerts[:15], 1):
        with st.container():
            col1, col2 = st.columns([4, 1])
            with col1:
                enhanced_side = alert.get('enhanced_side', 'UNKNOWN')
                confidence = alert.get('side_confidence', 0)
                
                if 'BUY' in enhanced_side:
                    side_emoji = "🟢"
                elif 'SELL' in enhanced_side:
                    side_emoji = "🔴"
                else:
                    side_emoji = "⚪"
                
                conf_emoji = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.4 else "🔴"
                
                st.markdown(f"**{i}. {side_emoji} {alert['ticker']} {alert['strike']:.0f}{alert['type']} "
                            f"{alert['expiry']} ({alert['dte']}d) - {enhanced_side} {conf_emoji}**")
                
                oi_analysis = alert.get('oi_analysis', {})
                st.write(f"💰 Premium: ${alert['premium']:,.0f} | Vol: {alert['volume']:,} | "
                         f"OI: {alert['open_interest']:,} | Vol/OI: {alert['vol_oi_ratio']:.1f}")
                st.write(f"📊 OI Level: {oi_analysis.get('oi_level', 'N/A')} | "
                         f"Liquidity: {oi_analysis.get('liquidity_score', 'N/A')} | "
                         f"IV: {alert['iv_percentage']} | Confidence: {confidence:.0%}")
                st.write(f"🎯 Scenarios: {', '.join(alert.get('scenarios', [])[:3])}")
                st.write(f"📍 Alert Reasons: {', '.join(alert.get('reasons', []))}")
            with col2:
                st.metric("Alert Score", alert.get('alert_score', 0))
            st.divider()

# --- CSV EXPORT ---
def save_to_csv(trades, filename_prefix):
    if not trades:
        st.warning("No data to save")
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    csv_data = []
    for trade in trades:
        row = trade.copy()
        if isinstance(row.get('reasons'), list):
            row['reasons'] = ', '.join(row['reasons'])
        if isinstance(row.get('scenarios'), list):
            row['scenarios'] = ', '.join(row['scenarios'])
        if isinstance(row.get('side_reasoning'), list):
            row['side_reasoning'] = ', '.join(row['side_reasoning'])
        if isinstance(row.get('oi_analysis'), dict):
            oi_analysis = row['oi_analysis']
            row['oi_level'] = oi_analysis.get('oi_level', '')
            row['liquidity_score'] = oi_analysis.get('liquidity_score', '')
            row['oi_change_indicator'] = oi_analysis.get('oi_change_indicator', '')
            del row['oi_analysis']
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
st.set_page_config(page_title="Enhanced Options Flow Tracker", page_icon="📊", layout="wide")
st.title("📊 Enhanced Options Flow Tracker")
st.markdown("### Real-time unusual options activity with Enhanced Buy/Sell Detection and Advanced Pattern Recognition")

with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    scan_type = st.selectbox(
        "Select Analysis Type:",
        [
            "🔍 Main Flow Analysis",
            "📈 Open Interest Deep Dive", 
            "🔄 Enhanced Buy/Sell Analysis",
            "🚨 Enhanced Alert System",
            "⚡ ETF Flow Scanner",
            "🎯 Pattern Recognition"
        ]
    )
    
    # Premium Range Filter
    st.markdown("### 💰 Premium Range Filter")
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
        index=0
    )
    
    # DTE Filter
    st.markdown("### 📅 Time to Expiry Filter")
    dte_filter = st.selectbox(
        "Select DTE Range:",
        [
            "All DTE",
            "0DTE Only",
            "Weekly (≤7d)",
            "Monthly (≤30d)",
            "Quarterly (≤90d)",
            "LEAPS (>90d)"
        ],
        index=0
    )
    
    # Enhanced Trade Side Filter
    st.markdown("### 🔄 Trade Side Filter")
    side_filter = st.selectbox(
        "Filter by Trade Side:",
        [
            "All Trades",
            "Buy Only",
            "Sell Only", 
            "High Confidence Only",
            "Medium+ Confidence"
        ],
        index=0
    )
    
    # Debug Mode
    st.markdown("### 🔧 Debug Options")
    debug_mode = st.checkbox("Enable Diagnostics", help="Show data quality diagnostics")
    
    # Quick Filter Buttons
    st.markdown("### ⚡ Quick Filters")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔥 Mega Trades", use_container_width=True):
            premium_range = "Above $1M"
            st.rerun()
    with col2:
        if st.button("⚡ 0DTE Plays", use_container_width=True):
            dte_filter = "0DTE Only"
            st.rerun()
    
    # Additional quick filters
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎯 High Conf", use_container_width=True):
            side_filter = "High Confidence Only"
            st.rerun()
    with col2:
        if st.button("🟢 Buys Only", use_container_width=True):
            side_filter = "Buy Only"
            st.rerun()
    
    run_scan = st.button("🔄 Run Enhanced Scan", type="primary", use_container_width=True)

if run_scan:
    with st.spinner(f"Running {scan_type}..."):
        if "ETF Flow Scanner" in scan_type:
            # ETF scanner uses its own data fetch
            trades = fetch_etf_trades()
            # Apply filters to ETF trades
            original_count = len(trades)
            trades = apply_premium_filter(trades, premium_range)
            trades = apply_dte_filter(trades, dte_filter)
            trades = apply_trade_side_filter(trades, side_filter)
            
            # Show filter results
            if len(trades) != original_count:
                st.info(f"**Filter Results:** {original_count} → {len(trades)} ETF trades after applying filters")
            
            # Debug diagnostics
            if debug_mode and trades:
                diagnose_trade_data(trades)
            
            if not trades:
                st.warning("⚠️ No ETF trades match your current filters. Try adjusting the filters.")
            else:
                display_etf_scanner(trades)
                with st.expander("💾 Export Data", expanded=False):
                    save_to_csv(trades, "enhanced_etf_flow_scanner")
        else:
            # Regular analysis types use general flow data
            trades = fetch_general_flow()
            
            # Apply filters
            original_count = len(trades)
            trades = apply_premium_filter(trades, premium_range)
            trades = apply_dte_filter(trades, dte_filter)
            trades = apply_trade_side_filter(trades, side_filter)
            
            # Show filter results
            if len(trades) != original_count:
                st.info(f"**Filter Results:** {original_count} → {len(trades)} trades after applying filters")
            
            # Debug diagnostics
            if debug_mode and trades:
                diagnose_trade_data(trades)
            
            if not trades:
                st.warning("⚠️ No trades match your current filters. Try adjusting the filters.")
            else:
                # Display enhanced summary for all scan types
                display_enhanced_summary(trades)
                
                if "Main Flow" in scan_type:
                    display_main_trades_table(trades)
                    with st.expander("💾 Export Data", expanded=False):
                        save_to_csv(trades, "enhanced_main_flow")

                elif "Open Interest" in scan_type:
                    display_open_interest_analysis(trades)
                    display_main_trades_table(trades, "📋 OI-Focused Trade Analysis")
                    with st.expander("💾 Export Data", expanded=False):
                        save_to_csv(trades, "enhanced_oi_analysis")

                elif "Buy/Sell" in scan_type:
                    display_enhanced_buy_sell_analysis(trades)
                    with st.expander("💾 Export Data", expanded=False):
                        save_to_csv(trades, "enhanced_buy_sell_flow")

                elif "Alert" in scan_type:
                    display_enhanced_alerts(trades)
                    with st.expander("💾 Export Data", expanded=False):
                        save_to_csv(trades, "enhanced_priority_alerts")
                
                elif "Pattern Recognition" in scan_type:
                    display_pattern_recognition_analysis(trades)
                    display_main_trades_table(trades, "📋 Pattern-Based Trade Analysis")
                    with st.expander("💾 Export Data", expanded=False):
                        save_to_csv(trades, "enhanced_pattern_analysis")

else:
    st.markdown("""
    ## Welcome to the Enhanced Options Flow Tracker! 👋
    
    ### 🆕 Major Updates in This Version:
    
    #### 🔄 **Revolutionary Buy/Sell Detection System**
    - **Multi-Method Analysis**: Combines 6+ detection methods for maximum accuracy
    - **Confidence Scoring**: Every trade gets a confidence score (0-100%)
    - **Enhanced Reasoning**: See exactly why each trade was classified as buy/sell
    - **Fallback Systems**: Multiple backup methods when primary data is missing
    - **Real-time Debugging**: Built-in diagnostics to troubleshoot detection issues
    
    #### 🎯 **Advanced Detection Methods:**
    1. **Bid/Ask Price Analysis** - Most reliable when available
    2. **Volume/OI Ratio Analysis** - Identifies new position building
    3. **Description Keyword Analysis** - Parses trade descriptions for buying/selling indicators
    4. **Rule Pattern Analysis** - Uses ascending/descending fill patterns
    5. **Option Moneyness Analysis** - OTM calls typically bought, etc.
    6. **Time-based Analysis** - Market open patterns, EOD positioning
    
    #### 📊 **Enhanced Trade Information:**
    - **Side Display**: 🟢 BUY / 🔴 SELL / ⚪ UNKNOWN with confidence levels
    - **Confidence Indicators**: 🟢 High (70%+) / 🟡 Medium (40-69%) / 🔴 Low (<40%)
    - **Detailed Reasoning**: See the logic behind each classification
    - **Quality Metrics**: Track data completeness and detection success rates
    
    #### 🔧 **Built-in Diagnostics:**
    - **Data Quality Check**: Automatically analyze bid/ask availability, price data quality
    - **Debug Mode**: Step-by-step analysis of individual trades
    - **Low Confidence Analysis**: Identify and troubleshoot problem trades
    - **API Health Monitoring**: Track response quality and completeness
    
    ### 📋 **Enhanced Analysis Types:**
    
    #### 🔍 **Main Flow Analysis**
    - All trades with enhanced buy/sell detection
    - Confidence scoring and reasoning for each trade
    - Short-term ETF focus section included
    
    #### 🔄 **Enhanced Buy/Sell Analysis** ⭐ NEW!
    - **Detailed confidence distribution** analysis
    - **Side-by-side buy vs sell** comparison with premiums
    - **Debug mode** for troubleshooting detection issues
    - **Low confidence trade analysis** to improve data quality
    - **Key insights** about market sentiment and data quality
    
    #### 🚨 **Enhanced Alert System**
    - Incorporates buy/sell confidence into alert scoring
    - Higher scores for high-confidence directional trades
    - Enhanced reasoning includes confidence metrics
    
    #### ⚡ **ETF Flow Scanner**
    - All ETF trades now show enhanced buy/sell detection
    - Confidence metrics for short-term plays
    - Buy/sell ratios for each ETF
    
    #### 🎯 **Pattern Recognition**
    - Multi-leg strategies now use enhanced trade sides
    - Gamma squeeze detection considers buy/sell confidence
    - Cross-asset correlation with directional analysis
    
    ### 🛠️ **Troubleshooting Features:**
    
    #### 🔧 **Debug Mode** (Enable in sidebar)
    - **Data Quality Diagnostics**: Check completeness of price, bid/ask, volume data
    - **Sample Trade Analysis**: Step-by-step breakdown of detection logic
    - **Field Availability Check**: See what data fields are actually available
    - **API Response Inspection**: Raw data examination for troubleshooting
    
    #### ⚠️ **Automatic Issue Detection**
    - **Missing Bid/Ask Data**: Identifies when price discovery is limited
    - **Low Confidence Trades**: Highlights trades needing better data
    - **Data Format Issues**: Detects and handles inconsistent data types
    - **Rate Limiting**: Handles API limitations gracefully
    
    ### 🎛️ **Enhanced Filters:**
    - **High Confidence Only**: Show only trades with 70%+ confidence
    - **Medium+ Confidence**: Trades with 40%+ confidence
    - **Buy/Sell Specific**: Filter by trade direction
    - **Debug Diagnostics**: Toggle detailed analysis on/off
    
    ### 💡 **Pro Tips for Best Results:**
    
    1. **Enable Debug Mode** if you're seeing too many "UNKNOWN" trades
    2. **Use "High Confidence Only"** filter for the most reliable directional data
    3. **Check the diagnostics** to see data quality metrics
    4. **Look for 🟢 High Confidence** indicators in the trade tables
    5. **Review low confidence trades** to understand data limitations
    
    ### 🚀 **What's Fixed:**
    - **Better field name detection** - handles different API response formats
    - **Improved error handling** - graceful degradation when data is missing
    - **Enhanced Volume/OI analysis** - more sophisticated thresholds
    - **Contextual rule interpretation** - better understanding of trade patterns
    - **Multiple fallback methods** - never gives up on classification
    
    **Ready to see the enhanced buy/sell detection in action? Select your analysis type and click 'Run Enhanced Scan'!**
    """)

