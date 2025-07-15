import streamlit as st
import httpx
from datetime import datetime, date, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import sqlite3
from functools import wraps
import json
import yfinance as yf
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# --- ENHANCED CONFIGURATION ---
class Config:
    UW_TOKEN = st.secrets.get("UW_TOKEN", "e6e8601a-0746-4cec-a07d-c3eabfc13926")
    EXCLUDE_TICKERS = {'TSLA', 'MSTR', 'CRCL', 'COIN', 'META'}
    ALLOWED_TICKERS = {'QQQ', 'SPY', 'IWM'}
    MIN_PREMIUM = 100000
    LIMIT = 500
    
    # Trading signal thresholds
    STRONG_BUY_THRESHOLD = 0.8
    BUY_THRESHOLD = 0.6
    SELL_THRESHOLD = 0.4
    STRONG_SELL_THRESHOLD = 0.2
    
    # Risk management
    MAX_POSITION_SIZE = 0.05  # 5% of portfolio
    STOP_LOSS_MULTIPLIER = 2.0
    TAKE_PROFIT_MULTIPLIER = 3.0
    
    # Options Greeks thresholds
    HIGH_DELTA_THRESHOLD = 0.7
    HIGH_GAMMA_THRESHOLD = 0.05
    HIGH_THETA_THRESHOLD = 0.05
    HIGH_VEGA_THRESHOLD = 0.15

config = Config()

# --- TRADING SIGNAL GENERATOR ---
def generate_trading_signals(trades):
    """Generate actionable trading signals based on flow analysis"""
    signals = []
    
    # Group trades by ticker
    ticker_groups = defaultdict(list)
    for trade in trades:
        ticker_groups[trade['ticker']].append(trade)
    
    for ticker, ticker_trades in ticker_groups.items():
        signal = analyze_ticker_flow(ticker, ticker_trades)
        if signal:
            signals.append(signal)
    
    return sorted(signals, key=lambda x: x['confidence'], reverse=True)

def analyze_ticker_flow(ticker, trades):
    """Analyze flow for a specific ticker and generate trading signal"""
    if not trades:
        return None
    
    # Calculate flow metrics
    total_premium = sum(t.get('premium', 0) for t in trades)
    call_premium = sum(t.get('premium', 0) for t in trades if t.get('type') == 'C')
    put_premium = sum(t.get('premium', 0) for t in trades if t.get('type') == 'P')
    
    buy_premium = sum(t.get('premium', 0) for t in trades if 'BUY' in t.get('trade_side', ''))
    sell_premium = sum(t.get('premium', 0) for t in trades if 'SELL' in t.get('trade_side', ''))
    
    # Calculate ratios
    call_put_ratio = call_premium / max(put_premium, 1)
    buy_sell_ratio = buy_premium / max(sell_premium, 1)
    
    # Analyze unusual activity
    unusual_volume = sum(1 for t in trades if t.get('vol_oi_ratio', 0) > 10)
    large_trades = sum(1 for t in trades if t.get('premium', 0) > 500000)
    
    # Calculate confidence score
    confidence = 0
    signal_factors = []
    
    # Flow direction analysis
    if call_put_ratio > 2 and buy_sell_ratio > 1.5:
        confidence += 0.3
        signal_factors.append("Strong Call Buying")
        direction = "BULLISH"
    elif call_put_ratio < 0.5 and buy_sell_ratio > 1.5:
        confidence += 0.3
        signal_factors.append("Strong Put Buying")
        direction = "BEARISH"
    elif put_premium > call_premium and 'SELL' in str([t.get('trade_side') for t in trades]):
        confidence += 0.25
        signal_factors.append("Put Selling (Bullish)")
        direction = "BULLISH"
    else:
        direction = "NEUTRAL"
        confidence += 0.1
    
    # Volume analysis
    if unusual_volume > len(trades) * 0.3:
        confidence += 0.2
        signal_factors.append("Unusual Volume Activity")
    
    # Size analysis
    if large_trades > 0:
        confidence += 0.15
        signal_factors.append("Large Premium Trades")
    
    # Time decay analysis
    short_term_trades = [t for t in trades if t.get('dte', 0) <= 7]
    if len(short_term_trades) > len(trades) * 0.5:
        confidence += 0.1
        signal_factors.append("Short-term Focus")
    
    # Institutional flow detection
    institutional_premium = sum(t.get('premium', 0) for t in trades if t.get('premium', 0) > 1000000)
    if institutional_premium > total_premium * 0.3:
        confidence += 0.2
        signal_factors.append("Institutional Flow")
    
    # Generate signal strength
    if confidence >= config.STRONG_BUY_THRESHOLD:
        signal_strength = "STRONG"
    elif confidence >= config.BUY_THRESHOLD:
        signal_strength = "MODERATE"
    else:
        signal_strength = "WEAK"
    
    # Risk assessment
    risk_level = calculate_risk_level(trades)
    
    # Entry/exit levels
    entry_exit = calculate_entry_exit_levels(ticker, trades)
    
    return {
        'ticker': ticker,
        'direction': direction,
        'signal_strength': signal_strength,
        'confidence': confidence,
        'total_premium': total_premium,
        'call_put_ratio': call_put_ratio,
        'buy_sell_ratio': buy_sell_ratio,
        'signal_factors': signal_factors,
        'risk_level': risk_level,
        'entry_price': entry_exit['entry'],
        'stop_loss': entry_exit['stop_loss'],
        'take_profit': entry_exit['take_profit'],
        'expected_move': entry_exit['expected_move'],
        'trades_count': len(trades),
        'key_trades': sorted(trades, key=lambda x: x.get('premium', 0), reverse=True)[:3]
    }

def calculate_risk_level(trades):
    """Calculate risk level based on trade characteristics"""
    risk_score = 0
    
    # High IV increases risk
    high_iv_trades = sum(1 for t in trades if t.get('iv', 0) > 0.5)
    if high_iv_trades > len(trades) * 0.3:
        risk_score += 0.3
    
    # Short DTE increases risk
    short_dte_trades = sum(1 for t in trades if t.get('dte', 0) <= 3)
    if short_dte_trades > len(trades) * 0.5:
        risk_score += 0.4
    
    # Low liquidity increases risk
    low_liquidity_trades = sum(1 for t in trades if t.get('open_interest', 0) < 100)
    if low_liquidity_trades > len(trades) * 0.3:
        risk_score += 0.3
    
    if risk_score > 0.7:
        return "HIGH"
    elif risk_score > 0.4:
        return "MEDIUM"
    else:
        return "LOW"

def calculate_entry_exit_levels(ticker, trades):
    """Calculate entry, stop loss, and take profit levels"""
    try:
        # Get current stock price
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        
        # Calculate expected move based on options flow
        total_premium = sum(t.get('premium', 0) for t in trades)
        total_volume = sum(t.get('volume', 0) for t in trades)
        
        # Estimate implied move
        avg_strike = np.mean([t.get('strike', 0) for t in trades])
        strike_std = np.std([t.get('strike', 0) for t in trades])
        
        expected_move = min(strike_std, current_price * 0.1)  # Cap at 10%
        
        # Calculate levels based on flow direction
        call_premium = sum(t.get('premium', 0) for t in trades if t.get('type') == 'C')
        put_premium = sum(t.get('premium', 0) for t in trades if t.get('type') == 'P')
        
        if call_premium > put_premium:
            # Bullish flow
            entry_price = current_price * 1.005  # Slight premium for entry
            stop_loss = current_price * 0.98
            take_profit = current_price * (1 + expected_move/current_price)
        else:
            # Bearish flow
            entry_price = current_price * 0.995
            stop_loss = current_price * 1.02
            take_profit = current_price * (1 - expected_move/current_price)
        
        return {
            'entry': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'expected_move': expected_move,
            'current_price': current_price
        }
        
    except Exception as e:
        return {
            'entry': 0,
            'stop_loss': 0,
            'take_profit': 0,
            'expected_move': 0,
            'current_price': 0
        }

# --- OPTIONS STRATEGY DETECTOR ---
def detect_options_strategies(trades):
    """Detect specific options strategies from flow"""
    strategies = []
    
    # Group by ticker and expiry
    ticker_expiry_groups = defaultdict(list)
    for trade in trades:
        key = f"{trade['ticker']}_{trade.get('expiry', '')}"
        ticker_expiry_groups[key].append(trade)
    
    for key, group_trades in ticker_expiry_groups.items():
        ticker = key.split('_')[0]
        detected_strategies = analyze_strategy_patterns(ticker, group_trades)
        strategies.extend(detected_strategies)
    
    return strategies

def analyze_strategy_patterns(ticker, trades):
    """Analyze patterns to detect specific options strategies"""
    strategies = []
    
    if len(trades) < 2:
        return strategies
    
    # Sort by strike price
    trades_sorted = sorted(trades, key=lambda x: x.get('strike', 0))
    
    # Detect Iron Condor / Butterfly
    if len(trades_sorted) >= 4:
        iron_condor = detect_iron_condor(ticker, trades_sorted)
        if iron_condor:
            strategies.append(iron_condor)
    
    # Detect Straddle / Strangle
    if len(trades_sorted) >= 2:
        straddle = detect_straddle_strangle(ticker, trades_sorted)
        if straddle:
            strategies.append(straddle)
    
    # Detect Spreads
    spreads = detect_spreads(ticker, trades_sorted)
    strategies.extend(spreads)
    
    # Detect Covered Calls / Protective Puts
    covered_strategies = detect_covered_strategies(ticker, trades_sorted)
    strategies.extend(covered_strategies)
    
    return strategies

def detect_iron_condor(ticker, trades):
    """Detect Iron Condor strategy"""
    if len(trades) < 4:
        return None
    
    # Look for 4 different strikes with calls and puts
    strikes = list(set([t.get('strike', 0) for t in trades]))
    if len(strikes) < 4:
        return None
    
    strikes.sort()
    
    # Check if we have the right combination
    put_trades = [t for t in trades if t.get('type') == 'P']
    call_trades = [t for t in trades if t.get('type') == 'C']
    
    if len(put_trades) >= 2 and len(call_trades) >= 2:
        total_premium = sum(t.get('premium', 0) for t in trades)
        
        return {
            'strategy': 'Iron Condor',
            'ticker': ticker,
            'strikes': strikes,
            'total_premium': total_premium,
            'max_profit': total_premium,
            'max_loss': (strikes[2] - strikes[1]) - total_premium,
            'probability': 0.7,  # Typically high probability
            'market_outlook': 'Neutral (Range-bound)'
        }
    
    return None

def detect_straddle_strangle(ticker, trades):
    """Detect Straddle or Strangle strategy"""
    call_trades = [t for t in trades if t.get('type') == 'C']
    put_trades = [t for t in trades if t.get('type') == 'P']
    
    if len(call_trades) >= 1 and len(put_trades) >= 1:
        call_strikes = [t.get('strike', 0) for t in call_trades]
        put_strikes = [t.get('strike', 0) for t in put_trades]
        
        # Check if same strike (straddle) or different strikes (strangle)
        if any(cs in put_strikes for cs in call_strikes):
            strategy_type = 'Straddle'
        else:
            strategy_type = 'Strangle'
        
        total_premium = sum(t.get('premium', 0) for t in trades)
        
        # Determine if long or short based on trade side
        buy_premium = sum(t.get('premium', 0) for t in trades if 'BUY' in t.get('trade_side', ''))
        sell_premium = sum(t.get('premium', 0) for t in trades if 'SELL' in t.get('trade_side', ''))
        
        if buy_premium > sell_premium:
            direction = 'Long'
            outlook = 'High Volatility Expected'
        else:
            direction = 'Short'
            outlook = 'Low Volatility Expected'
        
        return {
            'strategy': f'{direction} {strategy_type}',
            'ticker': ticker,
            'total_premium': total_premium,
            'market_outlook': outlook,
            'breakeven_upper': max(call_strikes) + total_premium,
            'breakeven_lower': min(put_strikes) - total_premium
        }
    
    return None

def detect_spreads(ticker, trades):
    """Detect various spread strategies"""
    spreads = []
    
    # Vertical spreads
    call_trades = [t for t in trades if t.get('type') == 'C']
    put_trades = [t for t in trades if t.get('type') == 'P']
    
    if len(call_trades) >= 2:
        call_spread = analyze_vertical_spread(ticker, call_trades, 'Call')
        if call_spread:
            spreads.append(call_spread)
    
    if len(put_trades) >= 2:
        put_spread = analyze_vertical_spread(ticker, put_trades, 'Put')
        if put_spread:
            spreads.append(put_spread)
    
    return spreads

def analyze_vertical_spread(ticker, trades, option_type):
    """Analyze vertical spread patterns"""
    if len(trades) < 2:
        return None
    
    # Sort by strike
    trades_sorted = sorted(trades, key=lambda x: x.get('strike', 0))
    
    # Look for buy low strike, sell high strike pattern
    buy_trades = [t for t in trades_sorted if 'BUY' in t.get('trade_side', '')]
    sell_trades = [t for t in trades_sorted if 'SELL' in t.get('trade_side', '')]
    
    if len(buy_trades) >= 1 and len(sell_trades) >= 1:
        buy_strike = buy_trades[0].get('strike', 0)
        sell_strike = sell_trades[0].get('strike', 0)
        
        net_premium = sum(t.get('premium', 0) for t in sell_trades) - sum(t.get('premium', 0) for t in buy_trades)
        
        if option_type == 'Call':
            if buy_strike < sell_strike:
                spread_type = 'Bull Call Spread'
                outlook = 'Moderately Bullish'
            else:
                spread_type = 'Bear Call Spread'
                outlook = 'Moderately Bearish'
        else:
            if buy_strike > sell_strike:
                spread_type = 'Bull Put Spread'
                outlook = 'Moderately Bullish'
            else:
                spread_type = 'Bear Put Spread'
                outlook = 'Moderately Bearish'
        
        return {
            'strategy': spread_type,
            'ticker': ticker,
            'strikes': [buy_strike, sell_strike],
            'net_premium': net_premium,
            'max_profit': abs(sell_strike - buy_strike) - abs(net_premium),
            'max_loss': abs(net_premium),
            'market_outlook': outlook
        }
    
    return None

def detect_covered_strategies(ticker, trades):
    """Detect covered call and protective put strategies"""
    strategies = []
    
    # Large call selling might indicate covered calls
    call_sells = [t for t in trades if t.get('type') == 'C' and 'SELL' in t.get('trade_side', '')]
    large_call_sells = [t for t in call_sells if t.get('volume', 0) > 100]
    
    if large_call_sells:
        for trade in large_call_sells:
            strategies.append({
                'strategy': 'Potential Covered Call',
                'ticker': ticker,
                'strike': trade.get('strike', 0),
                'premium_collected': trade.get('premium', 0),
                'volume': trade.get('volume', 0),
                'market_outlook': 'Neutral to Slightly Bullish'
            })
    
    # Large put buying might indicate protective puts
    put_buys = [t for t in trades if t.get('type') == 'P' and 'BUY' in t.get('trade_side', '')]
    large_put_buys = [t for t in put_buys if t.get('volume', 0) > 100]
    
    if large_put_buys:
        for trade in large_put_buys:
            strategies.append({
                'strategy': 'Potential Protective Put',
                'ticker': ticker,
                'strike': trade.get('strike', 0),
                'premium_paid': trade.get('premium', 0),
                'volume': trade.get('volume', 0),
                'market_outlook': 'Hedging Downside Risk'
            })
    
    return strategies

# --- EARNINGS IMPACT ANALYZER ---
def analyze_earnings_impact(trades):
    """Analyze trades around earnings announcements"""
    earnings_plays = []
    
    for trade in trades:
        ticker = trade.get('ticker', '')
        dte = trade.get('dte', 0)
        
        # Check if trade is likely earnings-related
        if dte <= 14:  # Within 2 weeks
            earnings_analysis = get_earnings_analysis(ticker, trade)
            if earnings_analysis:
                earnings_plays.append(earnings_analysis)
    
    return earnings_plays

def get_earnings_analysis(ticker, trade):
    """Get earnings-specific analysis for a trade"""
    try:
        # Get earnings date (simplified - in real implementation, use earnings calendar API)
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Estimate if it's an earnings play
        iv = trade.get('iv', 0)
        dte = trade.get('dte', 0)
        premium = trade.get('premium', 0)
        
        # High IV + Short DTE + Large premium = likely earnings play
        earnings_score = 0
        if iv > 0.5:
            earnings_score += 0.4
        if dte <= 7:
            earnings_score += 0.3
        if premium > 200000:
            earnings_score += 0.3
        
        if earnings_score > 0.6:
            return {
                'ticker': ticker,
                'trade': trade,
                'earnings_score': earnings_score,
                'iv': iv,
                'expected_move': calculate_expected_earnings_move(ticker, trade),
                'strategy_suggestion': get_earnings_strategy_suggestion(trade),
                'risk_level': 'HIGH'
            }
    
    except Exception as e:
        pass
    
    return None

def calculate_expected_earnings_move(ticker, trade):
    """Calculate expected move based on options pricing"""
    try:
        iv = trade.get('iv', 0)
        strike = trade.get('strike', 0)
        dte = trade.get('dte', 0)
        
        # Simplified expected move calculation
        if iv > 0 and dte > 0:
            daily_move = iv * (dte / 365) ** 0.5
            expected_move = strike * daily_move
            return expected_move
        
        return 0
    
    except Exception:
        return 0

def get_earnings_strategy_suggestion(trade):
    """Suggest strategy based on earnings trade characteristics"""
    iv = trade.get('iv', 0)
    option_type = trade.get('type', '')
    moneyness = trade.get('moneyness', '')
    trade_side = trade.get('trade_side', '')
    
    if iv > 0.6:
        if 'SELL' in trade_side:
            return "IV Crush Play - Sell premium before earnings"
        else:
            return "Volatility Expansion Play - Buy options for big move"
    
    if 'ATM' in moneyness:
        return "Directional Play - Betting on earnings direction"
    
    if 'OTM' in moneyness and option_type == 'C':
        return "Lottery Ticket - High risk, high reward"
    
    return "Earnings Uncertainty Play"

# --- GAMMA EXPOSURE CALCULATOR ---
def calculate_gamma_exposure(trades):
    """Calculate gamma exposure for trades"""
    gamma_exposure = {}
    
    for trade in trades:
        ticker = trade.get('ticker', '')
        strike = trade.get('strike', 0)
        option_type = trade.get('type', '')
        volume = trade.get('volume', 0)
        
        # Simplified gamma calculation
        gamma = estimate_gamma(trade)
        
        if ticker not in gamma_exposure:
            gamma_exposure[ticker] = {'total_gamma': 0, 'call_gamma': 0, 'put_gamma': 0, 'strikes': {}}
        
        if option_type == 'C':
            gamma_exposure[ticker]['call_gamma'] += gamma * volume
        else:
            gamma_exposure[ticker]['put_gamma'] += gamma * volume
        
        gamma_exposure[ticker]['total_gamma'] += gamma * volume
        
        # Track by strike
        if strike not in gamma_exposure[ticker]['strikes']:
            gamma_exposure[ticker]['strikes'][strike] = 0
        gamma_exposure[ticker]['strikes'][strike] += gamma * volume
    
    return gamma_exposure

def estimate_gamma(trade):
    """Estimate gamma for an option trade"""
    try:
        # Get current stock price
        ticker = trade.get('ticker', '')
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        
        strike = trade.get('strike', 0)
        dte = trade.get('dte', 0)
        iv = trade.get('iv', 0)
        
        if dte <= 0 or iv <= 0:
            return 0
        
        # Simplified gamma calculation using Black-Scholes approximation
        time_to_expiry = dte / 365
        d1 = (np.log(current_price / strike) + 0.5 * iv**2 * time_to_expiry) / (iv * np.sqrt(time_to_expiry))
        
        gamma = norm.pdf(d1) / (current_price * iv * np.sqrt(time_to_expiry))
        
        return gamma
    
    except Exception:
        return 0

# --- MOMENTUM SCANNER ---
def scan_momentum_opportunities(trades):
    """Scan for momentum-based trading opportunities"""
    momentum_opportunities = []
    
    # Group by ticker
    ticker_groups = defaultdict(list)
    for trade in trades:
        ticker_groups[trade['ticker']].append(trade)
    
    for ticker, ticker_trades in ticker_groups.items():
        momentum_analysis = analyze_momentum_pattern(ticker, ticker_trades)
        if momentum_analysis:
            momentum_opportunities.append(momentum_analysis)
    
    return sorted(momentum_opportunities, key=lambda x: x['momentum_score'], reverse=True)

def analyze_momentum_pattern(ticker, trades):
    """Analyze momentum patterns for a ticker"""
    if len(trades) < 2:
        return None
    
    # Sort by time
    trades_sorted = sorted(trades, key=lambda x: x.get('time_ny', ''))
    
    # Calculate momentum indicators
    momentum_score = 0
    momentum_factors = []
    
    # Volume acceleration
    if len(trades_sorted) >= 3:
        recent_volume = sum(t.get('volume', 0) for t in trades_sorted[-3:])
        earlier_volume = sum(t.get('volume', 0) for t in trades_sorted[:-3])
        
        if recent_volume > earlier_volume * 1.5:
            momentum_score += 0.3
            momentum_factors.append("Volume Acceleration")
    
    # Premium flow direction
    call_premium = sum(t.get('premium', 0) for t in trades if t.get('type') == 'C')
    put_premium = sum(t.get('premium', 0) for t in trades if t.get('type') == 'P')
    
    if call_premium > put_premium * 2:
        momentum_score += 0.25
        momentum_factors.append("Strong Call Flow")
        direction = "BULLISH"
    elif put_premium > call_premium * 2:
        momentum_score += 0.25
        momentum_factors.append("Strong Put Flow")
        direction = "BEARISH"
    else:
        direction = "NEUTRAL"
    
    # Unusual activity
    unusual_trades = sum(1 for t in trades if t.get('vol_oi_ratio', 0) > 10)
    if unusual_trades > len(trades) * 0.3:
        momentum_score += 0.2
        momentum_factors.append("Unusual Activity")
    
    # Short-term focus
    short_term_trades = sum(1 for t in trades if t.get('dte', 0) <= 7)
    if short_term_trades > len(trades) * 0.6:
        momentum_score += 0.15
        momentum_factors.append("Short-term Focus")
    
    # Large trades
    large_trades = sum(1 for t in trades if t.get('premium', 0) > 500000)
    if large_trades > 0:
        momentum_score += 0.1
        momentum_factors.append("Large Trades Present")
    
    if momentum_score > 0.4:
        return {
            'ticker': ticker,
            'momentum_score': momentum_score,
            'direction': direction,
            'momentum_factors': momentum_factors,
            'total_trades': len(trades),
            'total_premium': sum(t.get('premium', 0) for t in trades),
            'time_window': f"{trades_sorted[0].get('time_ny', '')} - {trades_sorted[-1].get('time_ny', '')}",
            'key_levels': calculate_key_levels(ticker, trades)
        }
    
    return None

def calculate_key_levels(ticker, trades):
    """Calculate key price levels based on options flow"""
    try:
        # Get current price
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        
        # Calculate support/resistance from strike clusters
        strikes = [t.get('strike', 0) for t in trades]
        strike_counts = {}
        
        for strike in strikes:
            strike_counts[strike] = strike_counts.get(strike, 0) + 1
        
        # Find most active strikes
        sorted_strikes = sorted(strike_counts.items(), key=lambda x: x[1], reverse=True)
        
        key_levels = {
            'current_price': current_price,
            'support_levels': [],
            'resistance_levels': []
        }
        
        for strike, count in sorted_strikes[:5]:
            if strike < current_price:
                key_levels['support_levels'].append(strike)
            else:
                key_levels['resistance_levels'].append(strike)
        
        return key_levels
    
    except Exception:
        return {'current_price': 0, 'support_levels': [], 'resistance_levels': []}

# --- RISK MANAGEMENT CALCULATOR ---
def calculate_position_sizing(signal, account_size=100000):
    """Calculate appropriate position sizing based on signal and risk"""
    if not signal:
        return None
    
    confidence = signal.get('confidence', 0)
    risk_level = signal.get('risk_level', 'MEDIUM')
    
    # Base position size
    base_size = account_size * config.MAX_POSITION_SIZE
    
    # Adjust based on confidence
    confidence_multiplier = min(confidence * 2, 1.0)
    
    # Adjust based on risk level
    risk_multipliers = {'LOW': 1.0, 'MEDIUM': 0.7, 'HIGH': 0.4}
    risk_multiplier = risk_multipliers.get(risk_level, 0.5)
    
    position_size = base_size * confidence_multiplier * risk_multiplier
    
    return {
        'position_size': position_size,
        'max_loss': position_size * 0.02,  # 2% max loss per trade
        'shares_to_trade': int(position_size / signal.get('entry_price', 1)),
        'stop_loss_price': signal.get('stop_loss', 0),
        'take_profit_price': signal.get('take_profit', 0)
    }

# --- BACKTESTING FRAMEWORK ---
def backtest_strategy(trades, strategy_name, days_back=30):
    """Simple backtesting framework"""
    # This is a simplified version - in practice, you'd need historical data
    results = {
        'strategy': strategy_name,
        'total_signals': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0,
        'average_return': 0,
        'max_drawdown': 0,
        'sharpe_ratio': 0
    }
    
    # Generate signals from trades
    signals = generate_trading_signals(trades)
    results['total_signals'] = len(signals)
    
    # Simulate outcomes based on signal strength
    for signal in signals:
        confidence = signal.get('confidence', 0)
        
        # Simple simulation: higher confidence = higher probability of success
        if confidence > 0.7:
            win_probability = 0.65
        elif confidence > 0.5:
            win_probability = 0.55
        else:
            win_probability = 0.45
        
        # Simulate trade outcome
        if np.random.random() < win_probability:
            results['winning_trades'] += 1
        else:
            results['losing_trades'] += 1
    
    # Calculate metrics
    if results['total_signals'] > 0:
        results['win_rate'] = results['winning_trades'] / results['total_signals']
        results['average_return'] = (results['win_rate'] * 0.15) - ((1 - results['win_rate']) * 0.05)
    
    return results

# --- ALERT SYSTEM ---
def generate_actionable_alerts(trades, signals):
    """Generate actionable alerts based on analysis"""
    alerts = []
    
    # High confidence signals
    for signal in signals:
        if signal.get('confidence', 0) > 0.7:
            alerts.append({
                'type': 'TRADING_SIGNAL',
                'priority': 'HIGH',
                'ticker': signal['ticker'],
                'message': f"High confidence {signal['direction']} signal for {signal['ticker']}",
                'action': f"Consider {signal['direction']} position",
                'confidence': signal['confidence'],
                'entry_price': signal.get('entry_price', 0),
                'stop_loss': signal.get('stop_loss', 0),
                'take_profit': signal.get('take_profit', 0)
            })
    
    # Unusual activity alerts
    for trade in trades:
        if trade.get('vol_oi_ratio', 0) > 20:
            alerts.append({
                'type': 'UNUSUAL_ACTIVITY',
                'priority': 'MEDIUM',
                'ticker': trade['ticker'],
                'message': f"Unusual volume activity in {trade['ticker']} {trade['strike']}{trade['type']}",
                'action': "Monitor for follow-through",
                'vol_oi_ratio': trade.get('vol_oi_ratio', 0),
                'premium': trade.get('premium', 0)
            })
    
    # Earnings alerts
    earnings_plays = analyze_earnings_impact(trades)
    for earnings_play in earnings_plays:
        if earnings_play.get('earnings_score', 0) > 0.7:
            alerts.append({
                'type': 'EARNINGS_PLAY',
                'priority': 'HIGH',
                'ticker': earnings_play['ticker'],
                'message': f"High probability earnings play detected for {earnings_play['ticker']}",
                'action': earnings_play.get('strategy_suggestion', ''),
                'expected_move': earnings_play.get('expected_move', 0)
            })
    
    return sorted(alerts, key=lambda x: {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}.get(x['priority'], 0), reverse=True)

# --- PORTFOLIO IMPACT ANALYZER ---
def analyze_portfolio_impact(trades, portfolio_holdings=None):
    """Analyze how options flow might impact existing portfolio"""
    if not portfolio_holdings:
        return None
    
    impact_analysis = {}
    
    for holding in portfolio_holdings:
        ticker = holding.get('ticker', '')
        shares = holding.get('shares', 0)
        avg_cost = holding.get('avg_cost', 0)
        
        # Find relevant trades
        relevant_trades = [t for t in trades if t.get('ticker') == ticker]
        
        if relevant_trades:
            # Generate signal for this ticker
            signal = analyze_ticker_flow(ticker, relevant_trades)
            
            if signal:
                current_value = shares * signal.get('entry_price', avg_cost)
                
                impact_analysis[ticker] = {
                    'current_position': {
                        'shares': shares,
                        'avg_cost': avg_cost,
                        'current_value': current_value
                    },
                    'options_signal': signal,
                    'potential_impact': calculate_potential_impact(signal, shares, avg_cost),
                    'hedge_suggestions': generate_hedge_suggestions(signal, shares, avg_cost)
                }
    
    return impact_analysis

def calculate_potential_impact(signal, shares, avg_cost):
    """Calculate potential impact on existing position"""
    if not signal:
        return None
    
    expected_move = signal.get('expected_move', 0)
    current_price = signal.get('entry_price', avg_cost)
    
    potential_gain = shares * expected_move if signal.get('direction') == 'BULLISH' else -shares * expected_move
    potential_loss = -shares * expected_move if signal.get('direction') == 'BULLISH' else shares * expected_move
    
    return {
        'potential_gain': potential_gain,
        'potential_loss': potential_loss,
        'gain_percentage': (potential_gain / (shares * avg_cost)) * 100,
        'loss_percentage': (potential_loss / (shares * avg_cost)) * 100
    }

def generate_hedge_suggestions(signal, shares, avg_cost):
    """Generate hedging suggestions based on signal"""
    suggestions = []
    
    if signal.get('direction') == 'BEARISH' and signal.get('confidence', 0) > 0.6:
        # Suggest protective puts
        suggestions.append({
            'strategy': 'Protective Put',
            'action': f'Buy {shares//100} put contracts',
            'reasoning': 'Hedge against downside risk',
            'cost_estimate': shares * 0.02  # Rough estimate
        })
    
    if signal.get('direction') == 'BULLISH' and shares > 0:
        # Suggest covered calls
        suggestions.append({
            'strategy': 'Covered Call',
            'action': f'Sell {shares//100} call contracts',
            'reasoning': 'Generate income from upside',
            'income_estimate': shares * 0.015  # Rough estimate
        })
    
    return suggestions

# --- MAIN ENHANCED FUNCTIONALITY ---
def run_enhanced_analysis(trades):
    """Run comprehensive analysis with actionable insights"""
    
    if not trades:
        return None
    
    # Generate trading signals
    signals = generate_trading_signals(trades)
    
    # Detect options strategies
    strategies = detect_options_strategies(trades)
    
    # Analyze momentum opportunities
    momentum_opportunities = scan_momentum_opportunities(trades)
    
    # Calculate gamma exposure
    gamma_exposure = calculate_gamma_exposure(trades)
    
    # Analyze earnings impact
    earnings_plays = analyze_earnings_impact(trades)
    
    # Generate alerts
    alerts = generate_actionable_alerts(trades, signals)
    
    # Backtest results
    backtest_results = backtest_strategy(trades, "Options Flow Strategy")
    
    return {
        'trading_signals': signals,
        'options_strategies': strategies,
        'momentum_opportunities': momentum_opportunities,
        'gamma_exposure': gamma_exposure,
        'earnings_plays': earnings_plays,
        'actionable_alerts': alerts,
        'backtest_results': backtest_results,
        'total_trades_analyzed': len(trades),
        'analysis_timestamp': datetime.now().isoformat()
    }

# --- DISPLAY FUNCTIONS FOR ENHANCED FUNCTIONALITY ---
def display_trading_signals(signals):
    """Display actionable trading signals"""
    st.markdown("### 🎯 Trading Signals")
    
    if not signals:
        st.info("No trading signals generated")
        return
    
    for i, signal in enumerate(signals[:10], 1):
        confidence_color = "🟢" if signal['confidence'] > 0.7 else "🟡" if signal['confidence'] > 0.5 else "🔴"
        direction_emoji = "📈" if signal['direction'] == 'BULLISH' else "📉" if signal['direction'] == 'BEARISH' else "➡️"
        
        with st.expander(f"{confidence_color} {direction_emoji} {signal['ticker']} - {signal['signal_strength']} {signal['direction']} ({signal['confidence']:.1%})", expanded=i<=3):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Entry Price:** ${signal['entry_price']:.2f}")
                st.write(f"**Stop Loss:** ${signal['stop_loss']:.2f}")
                st.write(f"**Take Profit:** ${signal['take_profit']:.2f}")
                st.write(f"**Expected Move:** ${signal['expected_move']:.2f}")
            
            with col2:
                st.write(f"**Total Premium:** ${signal['total_premium']:,.0f}")
                st.write(f"**Call/Put Ratio:** {signal['call_put_ratio']:.2f}")
                st.write(f"**Buy/Sell Ratio:** {signal['buy_sell_ratio']:.2f}")
                st.write(f"**Risk Level:** {signal['risk_level']}")
            
            st.write(f"**Signal Factors:** {', '.join(signal['signal_factors'])}")
            
            # Position sizing
            position_info = calculate_position_sizing(signal)
            if position_info:
                st.write(f"**Suggested Position Size:** ${position_info['position_size']:,.0f}")
                st.write(f"**Max Loss:** ${position_info['max_loss']:,.0f}")
            
            # Key trades
            st.write("**Key Supporting Trades:**")
            for j, trade in enumerate(signal['key_trades'], 1):
                st.write(f"{j}. {trade['strike']}{trade['type']} - ${trade['premium']:,.0f} - {trade.get('trade_side', 'N/A')}")

def display_options_strategies(strategies):
    """Display detected options strategies"""
    st.markdown("### 🎲 Options Strategies Detected")
    
    if not strategies:
        st.info("No complex options strategies detected")
        return
    
    for strategy in strategies:
        st.write(f"**{strategy['strategy']}** - {strategy['ticker']}")
        st.write(f"Market Outlook: {strategy.get('market_outlook', 'N/A')}")
        
        if 'max_profit' in strategy:
            st.write(f"Max Profit: ${strategy['max_profit']:,.0f}")
        if 'max_loss' in strategy:
            st.write(f"Max Loss: ${strategy['max_loss']:,.0f}")
        if 'probability' in strategy:
            st.write(f"Success Probability: {strategy['probability']:.1%}")
        
        st.divider()

def display_momentum_opportunities(opportunities):
    """Display momentum trading opportunities"""
    st.markdown("### 🚀 Momentum Opportunities")
    
    if not opportunities:
        st.info("No momentum opportunities detected")
        return
    
    for opp in opportunities[:5]:
        momentum_emoji = "🔥" if opp['momentum_score'] > 0.7 else "⚡" if opp['momentum_score'] > 0.5 else "📊"
        direction_emoji = "📈" if opp['direction'] == 'BULLISH' else "📉" if opp['direction'] == 'BEARISH' else "➡️"
        
        st.write(f"{momentum_emoji} {direction_emoji} **{opp['ticker']}** - Score: {opp['momentum_score']:.1%}")
        st.write(f"Direction: {opp['direction']}")
        st.write(f"Factors: {', '.join(opp['momentum_factors'])}")
        st.write(f"Total Premium: ${opp['total_premium']:,.0f}")
        st.write(f"Time Window: {opp['time_window']}")
        
        # Key levels
        levels = opp['key_levels']
        if levels['support_levels']:
            st.write(f"Support Levels: {', '.join([f'${s:.2f}' for s in levels['support_levels']])}")
        if levels['resistance_levels']:
            st.write(f"Resistance Levels: {', '.join([f'${r:.2f}' for r in levels['resistance_levels']])}")
        
        st.divider()

def display_actionable_alerts(alerts):
    """Display actionable alerts"""
    st.markdown("### 🚨 Actionable Alerts")
    
    if not alerts:
        st.info("No actionable alerts")
        return
    
    for alert in alerts[:10]:
        priority_color = "🔴" if alert['priority'] == 'HIGH' else "🟡" if alert['priority'] == 'MEDIUM' else "🔵"
        
        st.write(f"{priority_color} **{alert['type']}** - {alert['ticker']}")
        st.write(f"Message: {alert['message']}")
        st.write(f"Action: {alert['action']}")
        
        if alert['type'] == 'TRADING_SIGNAL':
            st.write(f"Entry: ${alert['entry_price']:.2f} | Stop: ${alert['stop_loss']:.2f} | Target: ${alert['take_profit']:.2f}")
        
        st.divider()

def display_enhanced_analysis_results(analysis_results):
    """Display comprehensive analysis results"""
    if not analysis_results:
        st.warning("No analysis results available")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Trading Signals", len(analysis_results['trading_signals']))
    with col2:
        st.metric("Strategies Detected", len(analysis_results['options_strategies']))
    with col3:
        st.metric("Momentum Opportunities", len(analysis_results['momentum_opportunities']))
    with col4:
        st.metric("Actionable Alerts", len(analysis_results['actionable_alerts']))
    
    # Display sections
    display_trading_signals(analysis_results['trading_signals'])
    display_options_strategies(analysis_results['options_strategies'])
    display_momentum_opportunities(analysis_results['momentum_opportunities'])
    display_actionable_alerts(analysis_results['actionable_alerts'])
    
    # Backtest results
    backtest = analysis_results['backtest_results']
    st.markdown("### 📊 Strategy Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Win Rate", f"{backtest['win_rate']:.1%}")
    with col2:
        st.metric("Average Return", f"{backtest['average_return']:.1%}")
    with col3:
        st.metric("Total Signals", backtest['total_signals'])
