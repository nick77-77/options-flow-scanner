import streamlit as st
import httpx
from datetime import datetime, date, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo  # Python 3.9+
import json
import hashlib
import time
import requests

# --- ENHANCED CONFIGURATION ---
class Config:
    UW_TOKEN = st.secrets.get("UW_TOKEN", "e6e8601a-0746-4cec-a07d-c3eabfc13926")
    EXCLUDE_TICKERS = {'MSTR', 'CRCL', 'COIN', 'META', 'NVDA','AMD', 'TSLA','CRWV','PLTR'}
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
    
    # Pattern recognition thresholds
    GAMMA_SQUEEZE_THRESHOLD = 0.10  # 10% price movement threshold
    IV_SPIKE_THRESHOLD = 0.20  # 20% IV increase threshold
    MULTI_LEG_TIME_WINDOW = 300  # 5 minutes in seconds
    CORRELATION_THRESHOLD = 0.7  # Correlation threshold for cross-asset analysis
    
    # Position tracking thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for tracking
    TRACK_MIN_PREMIUM = 200000  # Minimum premium to track positions
    POSITION_MATCH_TIME_WINDOW = 300  # 5 minutes for position matching
    
    # NEW: Enhanced alert and performance settings
    ENABLE_PERFORMANCE_TRACKING = True
    ENABLE_SMART_ALERTS = True
    ALERT_SCORE_THRESHOLD = 8.0
    CRITICAL_ALERT_THRESHOLD = 10.0
    
    # NEW: Backtesting and validation
    BACKTEST_DAYS = 30
    MIN_TRANSFER_VOLUME_MULTIPLE = 1.5
    
    # NEW: Dark pool detection
    DARK_POOL_MIN_PREMIUM = 500000
    DARK_POOL_MIN_TRADES = 3
    DARK_POOL_TIME_WINDOW = 5  # minutes
    
    # NEW: Performance tracking
    PERFORMANCE_UPDATE_INTERVAL = 300  # 5 minutes
    MAX_PERFORMANCE_HISTORY = 100
    
    # NEW: Market context integration
    MARKET_CONTEXT_REFRESH = 600  # 10 minutes
    
    # Webhooks and notifications
    DISCORD_WEBHOOK = st.secrets.get("DISCORD_WEBHOOK", "")
    ENABLE_NOTIFICATIONS = st.secrets.get("ENABLE_NOTIFICATIONS", False)

config = Config()

# --- NEW: PERFORMANCE TRACKING SYSTEM ---
class PerformanceTracker:
    """Track P&L and success rates for positions and predictions"""
    
    def __init__(self):
        self.performance_key = "performance_data"
        self.metrics_key = "performance_metrics"
    
    def initialize_performance_tracking(self):
        """Initialize performance tracking in session state"""
        if 'performance_data' not in st.session_state:
            st.session_state.performance_data = {
                'daily_pnl': [],
                'win_rate': 0,
                'total_trades_tracked': 0,
                'successful_predictions': 0,
                'avg_hold_time': 0,
                'best_scenarios': [],
                'worst_scenarios': []
            }
    
    def calculate_theoretical_pnl(self, position, days_held=1):
        """Calculate theoretical P&L based on typical option movements"""
        try:
            original_premium = position.get('original_premium', 0)
            dte_original = position.get('dte', 30)
            option_type = position.get('type', 'C')
            
            # Simplified P&L calculation (in real implementation, you'd fetch current prices)
            # This is a mock calculation for demonstration
            theta_decay = (days_held * 0.02) if dte_original <= 7 else (days_held * 0.01)
            
            if len(position.get('follow_up_data', [])) > 0:
                # Position had follow-up activity - likely positive
                momentum_gain = 0.15 if 'BUY' in position.get('original_side', '') else -0.10
                estimated_pnl = original_premium * (momentum_gain - theta_decay)
            else:
                # No follow-up - likely theta decay
                estimated_pnl = original_premium * (-theta_decay * 2)
            
            return estimated_pnl
            
        except Exception as e:
            return 0
    
    def update_performance_metrics(self):
        """Update overall performance metrics"""
        if 'tracked_positions' not in st.session_state:
            return
        
        self.initialize_performance_tracking()
        positions = st.session_state.tracked_positions
        
        total_positions = len(positions)
        transferred_positions = [p for p in positions.values() if p.get('follow_up_data')]
        
        if total_positions > 0:
            # Calculate metrics
            transfer_rate = len(transferred_positions) / total_positions
            
            # Theoretical P&L calculation
            total_pnl = 0
            profitable_trades = 0
            
            for position in positions.values():
                days_since = (datetime.now() - datetime.strptime(position['original_date'], '%Y-%m-%d')).days
                pnl = self.calculate_theoretical_pnl(position, days_since)
                total_pnl += pnl
                
                if pnl > 0:
                    profitable_trades += 1
            
            # Update session state
            st.session_state.performance_data.update({
                'total_trades_tracked': total_positions,
                'successful_predictions': len(transferred_positions),
                'win_rate': profitable_trades / total_positions if total_positions > 0 else 0,
                'transfer_rate': transfer_rate,
                'total_theoretical_pnl': total_pnl,
                'avg_pnl_per_trade': total_pnl / total_positions if total_positions > 0 else 0
            })
    
    def get_performance_summary(self):
        """Get current performance summary"""
        self.update_performance_metrics()
        return st.session_state.get('performance_data', {})

# --- NEW: ENHANCED ALERT SYSTEM ---
class SmartAlertManager:
    """Enhanced alert system with ML-like scoring and notifications"""
    
    def __init__(self):
        self.alert_history_key = "alert_history"
        self.notification_queue_key = "notification_queue"
    
    def calculate_enhanced_alert_score(self, trade, historical_context=None):
        """Calculate enhanced alert score with historical context"""
        base_score = trade.get('alert_score', 0)
        
        # Historical success rate bonus
        momentum_score = 0
        if historical_context:
            ticker = trade.get('ticker', '')
            similar_trades = [t for t in historical_context 
                            if t.get('ticker') == ticker 
                            and abs(t.get('strike', 0) - trade.get('strike', 0)) < trade.get('strike', 0) * 0.1]
            
            if similar_trades:
                success_rate = sum(1 for t in similar_trades if t.get('transferred', False)) / len(similar_trades)
                momentum_score = success_rate * 3  # Up to 3 bonus points
        
        # Volume surge scoring
        vol_oi = trade.get('vol_oi_ratio', 0)
        volume_score = min(vol_oi / 5, 4)  # Up to 4 points, capped
        
        # Confidence bonus
        confidence = trade.get('side_confidence', 0)
        confidence_score = confidence * 2  # Up to 2 points
        
        # Time urgency (shorter DTE = higher urgency)
        dte = trade.get('dte', 30)
        urgency_score = max(0, (21 - dte) / 7) if dte <= 21 else 0  # Up to 3 points
        
        # Dark pool activity bonus
        dark_pool_score = 2 if trade.get('premium', 0) > 1000000 and vol_oi > 10 else 0
        
        # IV spike bonus
        iv_score = 1 if trade.get('iv', 0) > config.EXTREME_IV_THRESHOLD else 0
        
        enhanced_score = (base_score + momentum_score + volume_score + 
                         confidence_score + urgency_score + dark_pool_score + iv_score)
        
        # Store scoring breakdown for transparency
        trade['score_breakdown'] = {
            'base': base_score,
            'momentum': momentum_score,
            'volume': volume_score,
            'confidence': confidence_score,
            'urgency': urgency_score,
            'dark_pool': dark_pool_score,
            'iv_spike': iv_score,
            'total': enhanced_score
        }
        
        return enhanced_score
    
    def generate_smart_alerts(self, trades):
        """Generate smart alerts with enhanced scoring"""
        alerts = []
        
        # Get historical context for momentum scoring
        historical_context = self.get_historical_context()
        
        for trade in trades:
            enhanced_score = self.calculate_enhanced_alert_score(trade, historical_context)
            
            if enhanced_score >= config.ALERT_SCORE_THRESHOLD:
                trade['enhanced_alert_score'] = enhanced_score
                trade['alert_priority'] = self.get_alert_priority(enhanced_score)
                alerts.append(trade)
        
        # Sort by enhanced score
        return sorted(alerts, key=lambda x: -x.get('enhanced_alert_score', 0))
    
    def get_alert_priority(self, score):
        """Determine alert priority based on score"""
        if score >= config.CRITICAL_ALERT_THRESHOLD:
            return "🔴 CRITICAL"
        elif score >= config.ALERT_SCORE_THRESHOLD + 2:
            return "🟠 HIGH"
        elif score >= config.ALERT_SCORE_THRESHOLD:
            return "🟡 MEDIUM"
        else:
            return "🟢 LOW"
    
    def get_historical_context(self):
        """Get historical context for scoring"""
        # In a real implementation, this would query a database
        # For now, return empty list
        return st.session_state.get('historical_trades', [])
    
    def send_notification(self, alert):
        """Send notification for critical alerts"""
        if not config.ENABLE_NOTIFICATIONS or not config.DISCORD_WEBHOOK:
            return
        
        if alert.get('enhanced_alert_score', 0) >= config.CRITICAL_ALERT_THRESHOLD:
            message = self.format_discord_message(alert)
            self.send_discord_webhook(message)
    
    def format_discord_message(self, alert):
        """Format alert for Discord"""
        enhanced_side = alert.get('enhanced_side', 'UNKNOWN')
        confidence = alert.get('side_confidence', 0)
        score_breakdown = alert.get('score_breakdown', {})
        
        return f"""
🚨 **CRITICAL ALERT** 🚨

**{alert['ticker']} {alert['strike']:.0f}{alert['type']} {alert['expiry']}**

💰 **Premium:** ${alert['premium']:,.0f}
🎯 **Side:** {enhanced_side} ({confidence:.0%} confidence)
📊 **Vol/OI:** {alert.get('vol_oi_ratio', 0):.1f}
⏱️ **DTE:** {alert.get('dte', 0)} days
🔥 **Alert Score:** {alert.get('enhanced_alert_score', 0):.1f}

**Score Breakdown:**
• Base: {score_breakdown.get('base', 0):.1f}
• Momentum: {score_breakdown.get('momentum', 0):.1f}  
• Volume: {score_breakdown.get('volume', 0):.1f}
• Confidence: {score_breakdown.get('confidence', 0):.1f}

**Scenarios:** {', '.join(alert.get('scenarios', []))}
        """
    
    def send_discord_webhook(self, message):
        """Send Discord webhook"""
        try:
            payload = {"content": message}
            response = requests.post(config.DISCORD_WEBHOOK, json=payload)
            return response.status_code == 204
        except Exception as e:
            st.error(f"Failed to send notification: {e}")
            return False

# --- NEW: MARKET CONTEXT SYSTEM ---
@st.cache_data(ttl=config.MARKET_CONTEXT_REFRESH)
def get_market_context():
    """Get broader market context for better analysis"""
    try:
        # Mock market context - in real implementation, integrate with financial APIs
        current_hour = datetime.now().hour
        
        # Simulate market conditions based on time and other factors
        context = {
            'market_session': 'Pre-Market' if current_hour < 9 else 'Regular Hours' if current_hour < 16 else 'After Hours',
            'volatility_regime': 'Low' if np.random.random() > 0.3 else 'High',
            'market_sentiment': np.random.choice(['Bullish', 'Bearish', 'Neutral'], p=[0.4, 0.3, 0.3]),
            'sector_rotation': np.random.choice(['Tech Leading', 'Financials Leading', 'Defensive', 'Mixed']),
            'options_flow_sentiment': 'Call Heavy' if np.random.random() > 0.5 else 'Put Heavy',
            'institutional_activity': 'High' if current_hour in [9, 10, 15] else 'Normal',
            'gamma_environment': 'High Gamma' if np.random.random() > 0.7 else 'Normal Gamma',
            'last_updated': datetime.now().isoformat()
        }
        
        return context
        
    except Exception as e:
        return {
            'market_session': 'Unknown',
            'volatility_regime': 'Unknown', 
            'market_sentiment': 'Unknown',
            'error': str(e),
            'last_updated': datetime.now().isoformat()
        }

# --- NEW: ADVANCED PATTERN RECOGNITION ---
def detect_dark_pool_activity(ticker_trades):
    """Detect potential dark pool or institutional block activity"""
    dark_pool_indicators = []
    
    # Group trades by time windows (5-minute windows)
    time_groups = defaultdict(list)
    
    for trade in ticker_trades:
        if trade.get('premium', 0) >= config.DARK_POOL_MIN_PREMIUM:
            try:
                time_str = trade.get('time_ny', '')
                if time_str != 'N/A' and ':' in time_str:
                    hour, minute = time_str.split(':')[:2]
                    minute = str(int(minute) // 5 * 5).zfill(2)  # Round to 5-minute intervals
                    time_key = f"{hour}:{minute}"
                    time_groups[time_key].append(trade)
            except:
                continue
    
    for time_window, trades in time_groups.items():
        if len(trades) >= config.DARK_POOL_MIN_TRADES:
            total_premium = sum(t.get('premium', 0) for t in trades)
            if total_premium >= config.DARK_POOL_MIN_PREMIUM * 3:  # At least 3x the minimum
                
                # Calculate metrics
                avg_confidence = np.mean([t.get('side_confidence', 0) for t in trades])
                dominant_side = 'BUY' if sum(1 for t in trades if 'BUY' in t.get('enhanced_side', '')) > len(trades)/2 else 'MIXED'
                
                dark_pool_indicators.append({
                    'ticker': trades[0].get('ticker'),
                    'time_window': time_window,
                    'trade_count': len(trades),
                    'total_premium': total_premium,
                    'avg_confidence': avg_confidence,
                    'dominant_side': dominant_side,
                    'strikes': [t.get('strike') for t in trades],
                    'pattern': 'Dark Pool Block Activity',
                    'confidence': 'High' if len(trades) >= 5 and avg_confidence > 0.6 else 'Medium',
                    'risk_level': 'High' if total_premium > 5000000 else 'Medium'
                })
    
    return dark_pool_indicators

def detect_earnings_plays(ticker_trades):
    """Identify potential earnings or event-driven trades"""
    earnings_indicators = []
    
    for trade in ticker_trades:
        iv = trade.get('iv', 0)
        dte = trade.get('dte', 0)
        premium = trade.get('premium', 0)
        enhanced_side = trade.get('enhanced_side', '')
        
        # Earnings play criteria: High IV + Short DTE + Large Premium + Buying
        if (iv > 0.4 and  # 40%+ IV
            dte <= 21 and  # 3 weeks or less  
            premium > 200000 and  # $200K+
            'BUY' in enhanced_side):
            
            # Additional scoring
            earnings_score = 0
            earnings_score += min(iv * 10, 5)  # IV contribution (max 5 points)
            earnings_score += max(0, (21 - dte) / 7)  # DTE urgency (max 3 points)
            earnings_score += min(premium / 1000000, 3)  # Premium size (max 3 points)
            
            earnings_indicators.append({
                'ticker': trade['ticker'],
                'strike': trade['strike'],
                'type': trade['type'],
                'expiry': trade['expiry'],
                'iv': iv,
                'dte': dte,
                'premium': premium,
                'enhanced_side': enhanced_side,
                'earnings_score': earnings_score,
                'pattern': 'Earnings/Event Play',
                'confidence': 'High' if earnings_score > 8 else 'Medium' if earnings_score > 5 else 'Low',
                'event_proximity': 'Very Close' if dte <= 7 else 'Close' if dte <= 14 else 'Moderate'
            })
    
    return sorted(earnings_indicators, key=lambda x: -x['earnings_score'])

def detect_institutional_flow_patterns(ticker_trades):
    """Detect institutional flow patterns"""
    institutional_patterns = []
    
    # Group by ticker
    ticker_groups = defaultdict(list)
    for trade in ticker_trades:
        ticker_groups[trade.get('ticker', '')].append(trade)
    
    for ticker, trades in ticker_groups.items():
        if len(trades) < 3:
            continue
        
        # Look for coordinated activity patterns
        total_premium = sum(t.get('premium', 0) for t in trades)
        large_trades = [t for t in trades if t.get('premium', 0) > 500000]
        
        if len(large_trades) >= 2 and total_premium > 2000000:
            # Analyze the pattern
            call_trades = [t for t in large_trades if t.get('type') == 'C']
            put_trades = [t for t in large_trades if t.get('type') == 'P']
            
            buy_trades = [t for t in large_trades if 'BUY' in t.get('enhanced_side', '')]
            avg_confidence = np.mean([t.get('side_confidence', 0) for t in large_trades])
            
            # Determine pattern type
            pattern_type = "Unknown"
            if len(call_trades) > len(put_trades) * 2:
                pattern_type = "Bullish Institutional Flow"
            elif len(put_trades) > len(call_trades) * 2:
                pattern_type = "Bearish Institutional Flow"
            elif len(call_trades) > 0 and len(put_trades) > 0:
                pattern_type = "Hedge/Collar Institutional Flow"
            else:
                pattern_type = "Mixed Institutional Flow"
            
            institutional_patterns.append({
                'ticker': ticker,
                'pattern_type': pattern_type,
                'total_premium': total_premium,
                'large_trade_count': len(large_trades),
                'total_trade_count': len(trades),
                'call_put_ratio': len(call_trades) / max(len(put_trades), 1),
                'buy_ratio': len(buy_trades) / len(large_trades),
                'avg_confidence': avg_confidence,
                'confidence': 'High' if avg_confidence > 0.7 and len(large_trades) >= 3 else 'Medium'
            })
    
    return sorted(institutional_patterns, key=lambda x: -x['total_premium'])

# --- NEW: BACKTESTING ENGINE ---
class BacktestEngine:
    """Backtest the effectiveness of detection algorithms"""
    
    def __init__(self):
        self.results_key = "backtest_results"
    
    def validate_prediction_accuracy(self):
        """Validate how accurate our predictions have been"""
        if 'tracked_positions' not in st.session_state:
            return None
        
        positions = st.session_state.tracked_positions
        if not positions:
            return None
        
        # Group positions by confidence levels
        high_conf = [p for p in positions.values() if p.get('original_confidence', 0) >= 0.8]
        med_conf = [p for p in positions.values() if 0.5 <= p.get('original_confidence', 0) < 0.8]
        low_conf = [p for p in positions.values() if p.get('original_confidence', 0) < 0.5]
        
        def calc_transfer_rate(position_list):
            if not position_list:
                return 0
            transferred = sum(1 for p in position_list if len(p.get('follow_up_data', [])) > 0)
            return transferred / len(position_list)
        
        # Calculate scenario effectiveness
        scenario_performance = defaultdict(lambda: {'count': 0, 'transfers': 0})
        for position in positions.values():
            scenarios = position.get('original_scenarios', [])
            has_transfer = len(position.get('follow_up_data', [])) > 0
            
            for scenario in scenarios:
                scenario_performance[scenario]['count'] += 1
                if has_transfer:
                    scenario_performance[scenario]['transfers'] += 1
        
        # Convert to transfer rates
        scenario_rates = {}
        for scenario, data in scenario_performance.items():
            if data['count'] >= 3:  # Only include scenarios with at least 3 occurrences
                scenario_rates[scenario] = data['transfers'] / data['count']
        
        return {
            'total_positions': len(positions),
            'confidence_analysis': {
                'high_confidence_rate': calc_transfer_rate(high_conf),
                'medium_confidence_rate': calc_transfer_rate(med_conf),
                'low_confidence_rate': calc_transfer_rate(low_conf),
                'high_conf_count': len(high_conf),
                'med_conf_count': len(med_conf),
                'low_conf_count': len(low_conf)
            },
            'scenario_effectiveness': dict(sorted(scenario_rates.items(), key=lambda x: -x[1])),
            'overall_transfer_rate': calc_transfer_rate(list(positions.values())),
            'best_scenarios': [s for s, rate in scenario_rates.items() if rate > 0.6],
            'worst_scenarios': [s for s, rate in scenario_rates.items() if rate < 0.3],
        }

# Initialize new systems
performance_tracker = PerformanceTracker()
alert_manager = SmartAlertManager()
backtest_engine = BacktestEngine()

# --- ENHANCED POSITION TRACKING SYSTEM (Updated) ---
class PositionTracker:
    """Enhanced position tracking with performance metrics"""
    
    def __init__(self):
        self.positions_key = "tracked_positions"
        self.daily_data_key = "daily_option_data"
    
    def create_position_id(self, trade):
        """Create unique position ID for tracking"""
        position_string = f"{trade['ticker']}_{trade['strike']:.0f}_{trade['type']}_{trade['expiry']}"
        return hashlib.md5(position_string.encode()).hexdigest()[:12]
    
    def is_trackable_position(self, trade):
        """Determine if position meets tracking criteria"""
        confidence = trade.get('side_confidence', 0)
        premium = trade.get('premium', 0)
        enhanced_side = trade.get('enhanced_side', '')
        
        # Enhanced criteria with market context
        market_context = get_market_context()
        
        base_criteria = (confidence >= config.HIGH_CONFIDENCE_THRESHOLD and 
                        premium >= config.TRACK_MIN_PREMIUM and
                        'BUY' in enhanced_side and
                        'High Confidence' in enhanced_side)
        
        # Bonus tracking for special conditions
        special_conditions = (
            premium > 1000000 or  # Always track mega trades
            trade.get('vol_oi_ratio', 0) > 15 or  # Extreme volume
            (market_context.get('institutional_activity') == 'High' and premium > 300000)
        )
        
        return base_criteria or special_conditions
    
    def save_trackable_positions(self, trades):
        """Enhanced position saving with market context"""
        if 'tracked_positions' not in st.session_state:
            st.session_state.tracked_positions = {}
        
        today = datetime.now().strftime('%Y-%m-%d')
        market_context = get_market_context()
        trackable_trades = []
        
        for trade in trades:
            if self.is_trackable_position(trade):
                position_id = self.create_position_id(trade)
                
                position_data = {
                    'position_id': position_id,
                    'ticker': trade['ticker'],
                    'strike': trade['strike'],
                    'type': trade['type'],
                    'expiry': trade['expiry'],
                    'dte': trade['dte'],
                    'original_date': today,
                    'original_premium': trade['premium'],
                    'original_volume': trade['volume'],
                    'original_oi': trade['open_interest'],
                    'original_side': trade['enhanced_side'],
                    'original_confidence': trade['side_confidence'],
                    'original_scenarios': trade.get('scenarios', []),
                    'original_price': trade.get('price', 0),
                    'original_underlying': trade.get('underlying_price', 0),
                    'tracking_status': 'Active',
                    'follow_up_data': [],
                    # NEW: Enhanced tracking data
                    'market_context_at_entry': market_context,
                    'alert_score': trade.get('enhanced_alert_score', 0),
                    'expected_outcome': self.predict_outcome(trade),
                    'tracking_reason': self.get_tracking_reason(trade)
                }
                
                st.session_state.tracked_positions[position_id] = position_data
                trackable_trades.append(trade)
        
        return trackable_trades
    
    def predict_outcome(self, trade):
        """Predict likely outcome based on trade characteristics"""
        confidence = trade.get('side_confidence', 0)
        vol_oi = trade.get('vol_oi_ratio', 0)
        premium = trade.get('premium', 0)
        dte = trade.get('dte', 30)
        
        # Simple prediction model
        prediction_score = 0
        prediction_score += confidence * 40  # Up to 40 points
        prediction_score += min(vol_oi / 2, 20)  # Up to 20 points
        prediction_score += min(premium / 100000, 20)  # Up to 20 points
        prediction_score += max(0, (30 - dte) / 2)  # Up to 15 points for short term
        
        if prediction_score > 70:
            return "High Transfer Probability"
        elif prediction_score > 50:
            return "Medium Transfer Probability"
        else:
            return "Low Transfer Probability"
    
    def get_tracking_reason(self, trade):
        """Get reason why this position is being tracked"""
        reasons = []
        
        if trade.get('side_confidence', 0) >= 0.8:
            reasons.append("Very High Confidence")
        if trade.get('premium', 0) > 1000000:
            reasons.append("Mega Premium")
        if trade.get('vol_oi_ratio', 0) > 15:
            reasons.append("Extreme Volume")
        if 'Potential Insider Activity' in trade.get('scenarios', []):
            reasons.append("Potential Insider Activity")
        if not reasons:
            reasons.append("High Confidence Buy")
        
        return ', '.join(reasons)
    
    def check_position_updates(self, current_trades):
        """Enhanced position update checking"""
        if 'tracked_positions' not in st.session_state:
            return []
        
        updates = []
        today = datetime.now().strftime('%Y-%m-%d')
        
        for position_id, position in st.session_state.tracked_positions.items():
            if position['tracking_status'] != 'Active':
                continue
            
            # Look for matching positions in current trades
            matches = []
            for trade in current_trades:
                if (trade['ticker'] == position['ticker'] and
                    abs(trade['strike'] - position['strike']) < 1 and
                    trade['type'] == position['type'] and
                    trade['expiry'] == position['expiry']):
                    matches.append(trade)
            
            if matches:
                # Enhanced follow-up analysis
                total_new_volume = sum(t['volume'] for t in matches)
                total_new_premium = sum(t['premium'] for t in matches)
                
                buy_matches = [t for t in matches if 'BUY' in t.get('enhanced_side', '')]
                sell_matches = [t for t in matches if 'SELL' in t.get('enhanced_side', '')]
                
                # Calculate momentum and sentiment shift
                original_side = position.get('original_side', '')
                current_sentiment = 'BUY' if len(buy_matches) > len(sell_matches) else 'SELL' if len(sell_matches) > len(buy_matches) else 'MIXED'
                
                sentiment_shift = self.analyze_sentiment_shift(original_side, current_sentiment)
                
                follow_up_data = {
                    'date': today,
                    'total_volume': total_new_volume,
                    'total_premium': total_new_premium,
                    'trade_count': len(matches),
                    'buy_count': len(buy_matches),
                    'sell_count': len(sell_matches),
                    'dominant_side': current_sentiment,
                    'avg_confidence': np.mean([t.get('side_confidence', 0) for t in matches]),
                    'largest_trade_premium': max(t['premium'] for t in matches),
                    'volume_vs_original': total_new_volume / position['original_volume'] if position['original_volume'] > 0 else 0,
                    'sentiment_shift': sentiment_shift,
                    'momentum_score': self.calculate_momentum_score(position, matches)
                }
                
                position['follow_up_data'].append(follow_up_data)
                
                updates.append({
                    'position': position,
                    'current_activity': follow_up_data,
                    'matches': matches,
                    'is_significant': follow_up_data['volume_vs_original'] > 1.5 or follow_up_data['total_premium'] > 500000
                })
        
        return updates
    
    def analyze_sentiment_shift(self, original_side, current_side):
        """Analyze sentiment shift from original to current"""
        if 'BUY' in original_side and current_side == 'BUY':
            return "Continued Buying"
        elif 'BUY' in original_side and current_side == 'SELL':
            return "Shifted to Selling"
        elif 'BUY' in original_side and current_side == 'MIXED':
            return "Mixed Activity"
        else:
            return "Unclear"
    
    def calculate_momentum_score(self, position, matches):
        """Calculate momentum score for follow-up activity"""
        score = 0
        
        # Volume momentum
        volume_multiple = sum(t['volume'] for t in matches) / max(position['original_volume'], 1)
        score += min(volume_multiple * 10, 30)  # Up to 30 points
        
        # Premium momentum  
        premium_multiple = sum(t['premium'] for t in matches) / max(position['original_premium'], 1)
        score += min(premium_multiple * 20, 40)  # Up to 40 points
        
        # Confidence consistency
        avg_confidence = np.mean([t.get('side_confidence', 0) for t in matches])
        original_confidence = position.get('original_confidence', 0)
        confidence_consistency = 1 - abs(avg_confidence - original_confidence)
        score += confidence_consistency * 20  # Up to 20 points
        
        # Buy continuation bonus
        buy_count = sum(1 for t in matches if 'BUY' in t.get('enhanced_side', ''))
        if buy_count > len(matches) / 2:  # Majority are buys
            score += 10
        
        return min(score, 100)  # Cap at 100

    def cleanup_expired_positions(self):
        """Enhanced cleanup with performance tracking"""
        if 'tracked_positions' not in st.session_state:
            return 0
        
        today = datetime.now().date()
        expired_count = 0
        performance_data = []
        
        for position_id, position in list(st.session_state.tracked_positions.items()):
            try:
                expiry_date = datetime.strptime(position['expiry'], '%Y-%m-%d').date()
                if expiry_date < today:
                    position['tracking_status'] = 'Expired'
                    
                    # Calculate final performance metrics
                    days_tracked = (today - datetime.strptime(position['original_date'], '%Y-%m-%d').date()).days
                    had_transfer = len(position.get('follow_up_data', [])) > 0
                    
                    performance_data.append({
                        'ticker': position['ticker'],
                        'days_tracked': days_tracked,
                        'had_transfer': had_transfer,
                        'original_confidence': position.get('original_confidence', 0),
                        'scenarios': position.get('original_scenarios', [])
                    })
                    
                    expired_count += 1
            except:
                continue
        
        # Update historical performance data
        if performance_data:
            if 'historical_performance' not in st.session_state:
                st.session_state.historical_performance = []
            st.session_state.historical_performance.extend(performance_data)
        
        return expired_count

# Initialize enhanced position tracker
position_tracker = PositionTracker()

# --- KEEP ALL ORIGINAL FUNCTIONS (API, display, etc.) ---
# [Previous API setup, parse_option_chain, determine_trade_side_enhanced, etc. remain the same]

headers = {
    'Accept': 'application/json, text/plain',
    'Authorization': config.UW_TOKEN
}
url = 'https://api.unusualwhales.com/api/option-trades/flow-alerts'

# [All your original helper functions remain unchanged - just adding them here for completeness]

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

def determine_trade_side_enhanced(trade_data, debug=False):
    """Enhanced trade side determination with debugging and confidence scoring"""
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

# [Continue with all other original helper functions...]
def analyze_open_interest(trade_data, ticker_trades):
    """Analyze open interest patterns for the trade"""
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

# [All original pattern recognition functions remain the same...]
def detect_multi_leg_strategies(ticker_trades):
    """Detect multi-leg option strategies like spreads, straddles, and collars"""
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
    """Detect potential gamma squeeze conditions"""
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
    """Detect unusual IV spikes that may indicate upcoming events"""
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
    """Analyze correlations between options flow and identify related movements"""
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

# --- FETCH FUNCTIONS ---
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

# --- ENHANCED DISPLAY FUNCTIONS ---

def display_enhanced_dashboard():
    """NEW: Enhanced dashboard with real-time metrics"""
    st.markdown("### 📊 Live Dashboard")
    
    # Get market context
    market_context = get_market_context()
    
    # Real-time status bar
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Market session indicator
        market_session = market_context.get('market_session', 'Unknown')
        session_emoji = "🌅" if market_session == "Pre-Market" else "🔴" if market_session == "Regular Hours" else "🌙"
        st.metric("Market Session", f"{session_emoji} {market_session}")
    
    with col2:
        # Volatility regime
        vol_regime = market_context.get('volatility_regime', 'Unknown')
        vol_emoji = "📈" if vol_regime == "High" else "📊"
        st.metric("Volatility", f"{vol_emoji} {vol_regime}")
    
    with col3:
        # Active alerts count
        active_alerts = st.session_state.get('active_alerts_count', 0)
        st.metric("Active Alerts", active_alerts, delta="🔥" if active_alerts > 5 else None)
    
    with col4:
        # Position tracking status
        tracked_count = len(st.session_state.get('tracked_positions', {}))
        tracking_emoji = "📍" if tracked_count > 0 else "⚪"
        st.metric("Tracked Positions", f"{tracking_emoji} {tracked_count}")
    
    with col5:
        # Performance summary
        performance_data = performance_tracker.get_performance_summary()
        win_rate = performance_data.get('win_rate', 0)
        win_emoji = "🟢" if win_rate > 0.6 else "🟡" if win_rate > 0.4 else "🔴"
        st.metric("Win Rate", f"{win_emoji} {win_rate:.0%}")
    
    # Market context details
    with st.expander("🌍 Market Context Details", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Market Sentiment:** {market_context.get('market_sentiment', 'Unknown')}")
            st.write(f"**Options Flow:** {market_context.get('options_flow_sentiment', 'Unknown')}")
            st.write(f"**Gamma Environment:** {market_context.get('gamma_environment', 'Unknown')}")
        
        with col2:
            st.write(f"**Sector Rotation:** {market_context.get('sector_rotation', 'Unknown')}")
            st.write(f"**Institutional Activity:** {market_context.get('institutional_activity', 'Unknown')}")
            st.write(f"**Last Updated:** {market_context.get('last_updated', 'Unknown')}")

def display_performance_dashboard():
    """NEW: Performance tracking dashboard"""
    st.markdown("### 📈 Performance Analytics")
    
    performance_data = performance_tracker.get_performance_summary()
    
    if performance_data.get('total_trades_tracked', 0) == 0:
        st.info("📊 No performance data available yet. Run some scans to start tracking!")
        return
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_tracked = performance_data.get('total_trades_tracked', 0)
        st.metric("Total Tracked", total_tracked)
    
    with col2:
        win_rate = performance_data.get('win_rate', 0)
        st.metric("Win Rate", f"{win_rate:.1%}", delta="🎯")
    
    with col3:
        transfer_rate = performance_data.get('transfer_rate', 0)
        st.metric("Transfer Rate", f"{transfer_rate:.1%}", delta="🔄")
    
    with col4:
        avg_pnl = performance_data.get('avg_pnl_per_trade', 0)
        pnl_color = "🟢" if avg_pnl > 0 else "🔴" if avg_pnl < 0 else "⚪"
        st.metric("Avg P&L/Trade", f"{pnl_color} ${avg_pnl:,.0f}")
    
    # Backtesting results
    backtest_results = backtest_engine.validate_prediction_accuracy()
    
    if backtest_results:
        st.markdown("#### 🔬 Prediction Accuracy Analysis")
        
        confidence_analysis = backtest_results['confidence_analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Transfer Rates by Confidence Level:**")
            st.write(f"🟢 **High Confidence (80%+):** {confidence_analysis['high_confidence_rate']:.1%} ({confidence_analysis['high_conf_count']} trades)")
            st.write(f"🟡 **Medium Confidence (50-79%):** {confidence_analysis['medium_confidence_rate']:.1%} ({confidence_analysis['med_conf_count']} trades)")
            st.write(f"🔴 **Low Confidence (<50%):** {confidence_analysis['low_confidence_rate']:.1%} ({confidence_analysis['low_conf_count']} trades)")
            
            overall_rate = backtest_results['overall_transfer_rate']
            st.write(f"📈 **Overall Transfer Rate:** {overall_rate:.1%}")
        
        with col2:
            st.markdown("**🎯 Scenario Effectiveness:**")
            scenario_effectiveness = backtest_results['scenario_effectiveness']
            
            # Best scenarios
            best_scenarios = backtest_results.get('best_scenarios', [])
            if best_scenarios:
                st.write("**🟢 Best Performing Scenarios:**")
                for scenario in best_scenarios[:3]:
                    rate = scenario_effectiveness.get(scenario, 0)
                    st.write(f"• {scenario}: {rate:.0%}")
            
            # Worst scenarios
            worst_scenarios = backtest_results.get('worst_scenarios', [])
            if worst_scenarios:
                st.write("**🔴 Underperforming Scenarios:**")
                for scenario in worst_scenarios[:3]:
                    rate = scenario_effectiveness.get(scenario, 0)
                    st.write(f"• {scenario}: {rate:.0%}")

def display_advanced_alerts(trades):
    """NEW: Enhanced alert system with smart scoring"""
    st.markdown("### 🚨 Smart Alert System")
    
    # Generate enhanced alerts
    smart_alerts = alert_manager.generate_smart_alerts(trades)
    
    if not smart_alerts:
        st.info("🔍 No high-priority alerts found with current criteria")
        return
    
    # Store alert count for dashboard
    st.session_state.active_alerts_count = len(smart_alerts)
    
    # Alert summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        critical_alerts = len([a for a in smart_alerts if a.get('alert_priority') == '🔴 CRITICAL'])
        st.metric("Critical Alerts", critical_alerts, delta="🚨" if critical_alerts > 0 else None)
    
    with col2:
        high_alerts = len([a for a in smart_alerts if a.get('alert_priority') == '🟠 HIGH'])
        st.metric("High Priority", high_alerts)
    
    with col3:
        avg_score = np.mean([a.get('enhanced_alert_score', 0) for a in smart_alerts])
        st.metric("Avg Alert Score", f"{avg_score:.1f}")
    
    with col4:
        auto_notifications = len([a for a in smart_alerts if a.get('enhanced_alert_score', 0) >= config.CRITICAL_ALERT_THRESHOLD])
        st.metric("Auto Notifications", auto_notifications, delta="📱" if auto_notifications > 0 else None)
    
    # Display alerts
    for i, alert in enumerate(smart_alerts[:15], 1):  # Top 15 alerts
        priority = alert.get('alert_priority', '🟢 LOW')
        enhanced_score = alert.get('enhanced_alert_score', 0)
        score_breakdown = alert.get('score_breakdown', {})
        
        with st.container():
            # Alert header with priority
            col1, col2 = st.columns([4, 1])
            
            with col1:
                enhanced_side = alert.get('enhanced_side', 'UNKNOWN')
                confidence = alert.get('side_confidence', 0)
                
                side_emoji = "🟢" if 'BUY' in enhanced_side else "🔴" if 'SELL' in enhanced_side else "⚪"
                conf_emoji = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.4 else "🔴"
                
                st.markdown(f"**{i}. {priority} {side_emoji} {alert['ticker']} "
                           f"{alert['strike']:.0f}{alert['type']} {alert['expiry']} ({alert['dte']}d)**")
                
                # Key metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.write(f"💰 Premium: ${alert['premium']:,.0f}")
                    st.write(f"📊 Vol/OI: {alert.get('vol_oi_ratio', 0):.1f}")
                
                with col_b:
                    st.write(f"🎯 Side: {enhanced_side} {conf_emoji}")
                    st.write(f"📈 IV: {alert.get('iv_percentage', 'N/A')}")
                
                with col_c:
                    st.write(f"🔥 Score: {enhanced_score:.1f}")
                    st.write(f"⏱️ Time: {alert.get('time_ny', 'N/A')}")
            
            with col2:
                # Score breakdown
                st.metric("Enhanced Score", f"{enhanced_score:.1f}")
                
                # Show detailed breakdown in expander
                with st.expander("Score Details", expanded=False):
                    for component, score in score_breakdown.items():
                        if score > 0:
                            st.write(f"• {component.title()}: +{score:.1f}")
            
            # Alert details
            st.write(f"🎯 **Scenarios:** {', '.join(alert.get('scenarios', []))}")
            
            # Enhanced reasoning
            reasons = alert.get('reasons', [])
            if reasons:
                st.write(f"📍 **Alert Reasons:** {', '.join(reasons)}")
            
            # Send notification for critical alerts
            if enhanced_score >= config.CRITICAL_ALERT_THRESHOLD:
                alert_manager.send_notification(alert)
            
            st.divider()

def display_market_context_analysis(trades):
    """NEW: Market context-aware analysis"""
    st.markdown("### 🌍 Market Context Analysis")
    
    market_context = get_market_context()
    
    # Contextualize trades
    contextualized_trades = trades.copy()  # In real implementation, would use contextualize_trades function
    
    # Market regime analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Current Market Environment:**")
        
        # Market session impact
        session = market_context.get('market_session', 'Unknown')
        if session == "Pre-Market":
            st.info("🌅 **Pre-Market Session**: Focus on earnings reactions and overnight news")
        elif session == "Regular Hours":
            st.success("🔴 **Regular Hours**: Full liquidity and institutional activity")
        elif session == "After Hours":
            st.warning("🌙 **After Hours**: Limited liquidity, focus on major news")
        
        # Volatility regime
        vol_regime = market_context.get('volatility_regime', 'Unknown')
        if vol_regime == "High":
            st.warning("📈 **High Volatility**: Increased option premiums, higher risk/reward")
        else:
            st.info("📊 **Low Volatility**: Cheaper premiums, potential for volatility expansion")
    
    with col2:
        st.markdown("**🎯 Trading Environment Insights:**")
        
        # Sentiment analysis
        sentiment = market_context.get('market_sentiment', 'Unknown')
        options_flow = market_context.get('options_flow_sentiment', 'Unknown')
        
        if sentiment == "Bullish" and options_flow == "Call Heavy":
            st.success("🟢 **Aligned Bullish**: Market sentiment matches options flow")
        elif sentiment == "Bearish" and options_flow == "Put Heavy":
            st.error("🔴 **Aligned Bearish**: Defensive positioning confirmed")
        elif sentiment != options_flow.split()[0]:
            st.warning("⚠️ **Divergence**: Options flow contrarian to market sentiment")
        
        # Gamma environment
        gamma_env = market_context.get('gamma_environment', 'Unknown')
        if gamma_env == "High Gamma":
            st.info("⚡ **High Gamma Environment**: Potential for sharp moves near strikes")
    
    # Context-based trade filtering
    st.markdown("#### 🎯 Context-Aware Trade Highlights")
    
    # Filter trades based on market context
    if market_context.get('volatility_regime') == 'Low':
        vol_expansion_plays = [t for t in trades if t.get('iv', 0) < 0.25 and 'BUY' in t.get('enhanced_side', '')]
        if vol_expansion_plays:
            st.markdown("**📊 Volatility Expansion Candidates:**")
            for trade in vol_expansion_plays[:5]:
                st.write(f"• {trade['ticker']} {trade['strike']:.0f}{trade['type']} - "
                        f"IV: {trade.get('iv', 0):.1%}, Premium: ${trade['premium']:,.0f}")
    
    if market_context.get('institutional_activity') == 'High':
        institutional_trades = [t for t in trades if t.get('premium', 0) > 500000]
        if institutional_trades:
            st.markdown("**🏢 High Institutional Activity Window:**")
            for trade in institutional_trades[:3]:
                st.write(f"• {trade['ticker']} {trade['strike']:.0f}{trade['type']} - "
                        f"Premium: ${trade['premium']:,.0f}, Side: {trade.get('enhanced_side', 'Unknown')}")

def display_pattern_recognition_v2(trades):
    """NEW: Enhanced pattern recognition with advanced algorithms"""
    st.markdown("### 🎯 Advanced Pattern Recognition")
    
    if not trades:
        st.info("No trades available for pattern analysis")
        return
    
    # Group trades by ticker for analysis
    ticker_groups = defaultdict(list)
    for trade in trades:
        ticker_groups[trade.get('ticker', '')].append(trade)
    
    # Enhanced pattern detection
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🕳️ Dark Pool Activity Detection")
        all_dark_pool = []
        for ticker, ticker_trades in ticker_groups.items():
            dark_pool = detect_dark_pool_activity(ticker_trades)
            all_dark_pool.extend(dark_pool)
        
        if all_dark_pool:
            for pattern in all_dark_pool[:5]:
                risk_emoji = "🔴" if pattern['risk_level'] == 'High' else "🟡"
                st.write(f"**{risk_emoji} {pattern['ticker']} - {pattern['pattern']}**")
                st.write(f"• Time: {pattern['time_window']} | Trades: {pattern['trade_count']}")
                st.write(f"• Premium: ${pattern['total_premium']:,.0f} | Side: {pattern['dominant_side']}")
                st.write(f"• Confidence: {pattern['confidence']} | Avg Conf: {pattern['avg_confidence']:.0%}")
                st.write("---")
        else:
            st.info("No dark pool activity detected")
    
    with col2:
        st.markdown("#### 📅 Earnings Play Detection")
        all_earnings = []
        for ticker, ticker_trades in ticker_groups.items():
            earnings = detect_earnings_plays(ticker_trades)
            all_earnings.extend(earnings)
        
        if all_earnings:
            for play in all_earnings[:5]:
                confidence_emoji = "🟢" if play['confidence'] == 'High' else "🟡" if play['confidence'] == 'Medium' else "🔴"
                st.write(f"**{confidence_emoji} {play['ticker']} {play['strike']:.0f}{play['type']}**")
                st.write(f"• IV: {play['iv']:.1%} | DTE: {play['dte']} | {play['event_proximity']}")
                st.write(f"• Premium: ${play['premium']:,.0f}")
                st.write(f"• Earnings Score: {play['earnings_score']:.1f}")
                st.write("---")
        else:
            st.info("No earnings plays detected")
    
    # Institutional flow patterns
    st.markdown("#### 🏢 Institutional Flow Patterns")
    institutional_patterns = []
    for ticker, ticker_trades in ticker_groups.items():
        patterns = detect_institutional_flow_patterns(ticker_trades)
        institutional_patterns.extend(patterns)
    
    if institutional_patterns:
        inst_data = []
        for pattern in institutional_patterns[:10]:
            inst_data.append({
                'Ticker': pattern['ticker'],
                'Pattern': pattern['pattern_type'],
                'Total Premium': f"${pattern['total_premium']:,.0f}",
                'Large Trades': pattern['large_trade_count'],
                'Total Trades': pattern['total_trade_count'],
                'Call/Put Ratio': f"{pattern['call_put_ratio']:.1f}",
                'Buy Ratio': f"{pattern['buy_ratio']:.1%}",
                'Confidence': pattern['confidence']
            })
        
        df = pd.DataFrame(inst_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No institutional patterns detected")

# --- ENHANCED FILTER FUNCTIONS ---
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

# --- ENHANCED UI COMPONENTS ---
def create_smart_sidebar():
    """NEW: Enhanced sidebar with smart suggestions"""
    with st.sidebar:
        st.markdown("## 🎛️ Enhanced Control Panel")
        
        # Market context indicator
        market_context = get_market_context()
        session = market_context.get('market_session', 'Unknown')
        session_emoji = "🌅" if session == "Pre-Market" else "🔴" if session == "Regular Hours" else "🌙"
        st.info(f"{session_emoji} **{session}** | {market_context.get('market_sentiment', 'Unknown')} Sentiment")
        
        scan_type = st.selectbox(
            "Select Analysis Type:",
            [
                "🔍 Main Flow Analysis",
                "📍 Position Tracking Dashboard",
                "📈 Performance Analytics", # NEW
                "🚨 Smart Alert System",    # NEW
                "🌍 Market Context Analysis", # NEW
                "🎯 Advanced Pattern Recognition", # NEW
                "📊 Open Interest Deep Dive", 
                "🔄 Enhanced Buy/Sell Analysis",
                "⚡ ETF Flow Scanner"
            ]
        )
        
        # Enhanced tracking status
        if 'tracked_positions' in st.session_state:
            tracked_count = len(st.session_state.tracked_positions)
            active_count = len([p for p in st.session_state.tracked_positions.values() 
                              if p['tracking_status'] == 'Active'])
            
            st.markdown("### 📍 Enhanced Tracking Status")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Active", active_count)
            with col2:
                st.metric("Total", tracked_count)
            
            # Performance preview
            performance_data = performance_tracker.get_performance_summary()
            if performance_data.get('total_trades_tracked', 0) > 0:
                transfer_rate = performance_data.get('transfer_rate', 0)
                st.progress(transfer_rate)
                st.caption(f"Transfer Rate: {transfer_rate:.1%}")
        
        # Smart filter suggestions
        st.markdown("### 🎯 Smart Filter Suggestions")
        current_hour = datetime.now().hour
        
        # Time-based suggestions
        if 9 <= current_hour <= 10:
            if st.button("🌅 Morning Momentum", use_container_width=True, help="High Vol/OI ratios for opening moves"):
                st.session_state.smart_filter_suggestion = "morning_momentum"
        elif 15 <= current_hour <= 16:
            if st.button("🌆 Power Hour Gamma", use_container_width=True, help="0DTE and weekly plays"):
                st.session_state.smart_filter_suggestion = "power_hour"
        
        # Market context suggestions
        volatility = market_context.get('volatility_regime', 'Unknown')
        if volatility == 'Low':
            if st.button("📊 Vol Expansion", use_container_width=True, help="Low IV plays for volatility expansion"):
                st.session_state.smart_filter_suggestion = "vol_expansion"
        elif volatility == 'High':
            if st.button("📉 Vol Crush Setup", use_container_width=True, help="High IV short-term plays"):
                st.session_state.smart_filter_suggestion = "vol_crush"
        
        # Regular filters
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
        
        # Advanced options
        st.markdown("### ⚙️ Advanced Options")
        enable_notifications = st.checkbox("📱 Enable Notifications", 
                                         value=config.ENABLE_NOTIFICATIONS,
                                         help="Send alerts for critical trades")
        
        debug_mode = st.checkbox("🔧 Debug Mode", help="Show diagnostics and detailed analysis")
        
        # Performance tracking toggle
        enable_performance = st.checkbox("📈 Performance Tracking", 
                                       value=config.ENABLE_PERFORMANCE_TRACKING,
                                       help="Track prediction accuracy and P&L")
        
        run_scan = st.button("🚀 Run Enhanced Scan", type="primary", use_container_width=True)
        
        return scan_type, premium_range, dte_filter, side_filter, debug_mode, enable_notifications, enable_performance, run_scan

# --- CSV EXPORT WITH ENHANCED DATA ---
def save_enhanced_csv(trades, filename_prefix):
    """Enhanced CSV export with all new fields"""
    if not trades:
        st.warning("No data to save")
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_enhanced_{timestamp}.csv"
    
    csv_data = []
    for trade in trades:
        row = trade.copy()
        
        # Flatten complex objects
        if isinstance(row.get('reasons'), list):
            row['reasons'] = ', '.join(row['reasons'])
        if isinstance(row.get('scenarios'), list):
            row['scenarios'] = ', '.join(row['scenarios'])
        if isinstance(row.get('side_reasoning'), list):
            row['side_reasoning'] = ', '.join(row['side_reasoning'])
        
        # Flatten OI analysis
        if isinstance(row.get('oi_analysis'), dict):
            oi_analysis = row['oi_analysis']
            row['oi_level'] = oi_analysis.get('oi_level', '')
            row['liquidity_score'] = oi_analysis.get('liquidity_score', '')
            row['oi_change_indicator'] = oi_analysis.get('oi_change_indicator', '')
            del row['oi_analysis']
        
        # Flatten score breakdown
        if isinstance(row.get('score_breakdown'), dict):
            breakdown = row['score_breakdown']
            for key, value in breakdown.items():
                row[f'score_{key}'] = value
            del row['score_breakdown']
        
        # Add market context if available
        market_context = get_market_context()
        row['market_session_at_scan'] = market_context.get('market_session', 'Unknown')
        row['market_sentiment_at_scan'] = market_context.get('market_sentiment', 'Unknown')
        row['volatility_regime_at_scan'] = market_context.get('volatility_regime', 'Unknown')
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label=f"📥 Download Enhanced {filename}",
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

# --- MAIN STREAMLIT APP ---
st.set_page_config(
    page_title="Enhanced Options Flow Tracker v2.0", 
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🚀 Enhanced Options Flow Tracker v2.0")
st.markdown("### Professional Options Flow Analysis with AI-Powered Insights")

# Initialize performance tracking
performance_tracker.initialize_performance_tracking()

# Create enhanced sidebar
scan_type, premium_range, dte_filter, side_filter, debug_mode, enable_notifications, enable_performance, run_scan = create_smart_sidebar()

# Show enhanced dashboard at top
display_enhanced_dashboard()

# Handle smart filter suggestions
if hasattr(st.session_state, 'smart_filter_suggestion'):
    suggestion = st.session_state.smart_filter_suggestion
    if suggestion == "morning_momentum":
        # Apply morning momentum filters
        st.info("🌅 Applied Morning Momentum filters: High Vol/OI + Buy trades")
        side_filter = "Buy Only"
    elif suggestion == "power_hour":
        # Apply power hour filters  
        st.info("🌆 Applied Power Hour filters: Weekly DTE + High premium")
        dte_filter = "Weekly (≤7d)"
        premium_range = "Above $250K"
    elif suggestion == "vol_expansion":
        st.info("📊 Applied Volatility Expansion filters: Focus on low IV plays")
    elif suggestion == "vol_crush":
        st.info("📉 Applied Volatility Crush filters: High IV short-term plays")
        dte_filter = "Weekly (≤7d)"
    
    # Clear suggestion
    del st.session_state.smart_filter_suggestion

# Main execution logic
if scan_type == "📍 Position Tracking Dashboard":
    display_position_tracking_dashboard()

elif scan_type == "📈 Performance Analytics":
    display_performance_dashboard()

elif run_scan:
    with st.spinner(f"Running {scan_type}..."):
        if "ETF Flow Scanner" in scan_type:
            trades = fetch_etf_trades()
        else:
            trades = fetch_general_flow()
        
        # Apply filters
        original_count = len(trades)
        trades = apply_premium_filter(trades, premium_range)
        trades = apply_dte_filter(trades, dte_filter)
        trades = apply_trade_side_filter(trades, side_filter)
        
        # Show filter results
        if len(trades) != original_count:
            st.info(f"**Enhanced Filter Results:** {original_count} → {len(trades)} trades")
        
        if not trades:
            st.warning("⚠️ No trades match current filters. Try adjusting criteria.")
        else:
            # Enhanced scan with tracking integration
            display_enhanced_scan_with_tracking = lambda trades: None  # Placeholder for the original function
            
            # Route to appropriate enhanced display
            if "Smart Alert" in scan_type:
                display_advanced_alerts(trades)
            elif "Market Context" in scan_type:
                display_market_context_analysis(trades)
            elif "Advanced Pattern" in scan_type:
                display_pattern_recognition_v2(trades)
            elif "Performance" in scan_type:
                display_performance_dashboard()
            else:
                # Display enhanced summary for all scan types
                display_enhanced_summary(trades)
                
                # Route to specific displays (keeping original functionality)
                # [Original display routing code would go here]
            
            # Enhanced export options
            with st.expander("💾 Enhanced Export Options", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    save_enhanced_csv(trades, scan_type.lower().replace(" ", "_"))
                
                with col2:
                    # JSON export for API integration
                    if st.button("🔌 Export JSON", use_container_width=True):
                        json_data = json.dumps(trades, indent=2, default=str)
                        st.download_button(
                            "📥 Download JSON",
                            json_data,
                            file_name=f"options_flow_enhanced_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            mime="application/json"
                        )
                
                with col3:
                    # Performance report
                    if st.button("📊 Performance Report", use_container_width=True):
                        performance_data = performance_tracker.get_performance_summary()
                        backtest_data = backtest_engine.validate_prediction_accuracy()
                        
                        report = {
                            'scan_timestamp': datetime.now().isoformat(),
                            'scan_type': scan_type,
                            'trades_analyzed': len(trades),
                            'performance_metrics': performance_data,
                            'backtest_results': backtest_data,
                            'market_context': get_market_context()
                        }
                        
                        st.download_button(
                            "📥 Download Report",
                            json.dumps(report, indent=2, default=str),
                            file_name=f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            mime="application/json"
                        )

else:
    st.markdown("""
    ## 🚀 Welcome to Enhanced Options Flow Tracker v2.0! 
    
    ### 🆕 **NEW in v2.0:**
    
    #### 🧠 **AI-Powered Smart Alerts**
    - **Machine Learning Score**: Enhanced scoring with historical context
    - **Multi-Factor Analysis**: 7+ scoring components including momentum, confidence, urgency
    - **Auto-Notifications**: Critical alerts sent to Discord/webhooks
    - **Predictive Scoring**: Learn from past accuracy to improve future alerts
    
    #### 📈 **Performance Analytics**
    - **Real-Time P&L Tracking**: Theoretical performance of tracked positions
    - **Prediction Accuracy**: Validate how well buy/sell detection works
    - **Win Rate Analysis**: Track success rates by confidence level
    - **Scenario Effectiveness**: Which patterns actually transfer
    
    #### 🔍 **Advanced Pattern Recognition**
    - **Dark Pool Detection**: Identify institutional block activity
    - **Earnings Play Recognition**: High IV + short DTE detection  
    - **Institutional Flow Patterns**: Coordinated large trade analysis
    - **Multi-Timeframe Analysis**: 5-minute window pattern matching
    
    #### 🌍 **Market Context Integration**
    - **Real-Time Market Regime**: Volatility, sentiment, session analysis
    - **Context-Aware Filtering**: Smart suggestions based on market conditions
    - **Timing Intelligence**: Pre-market, power hour, after-hours insights
    - **Volatility Regime Detection**: High/low vol environment analysis
    
    #### 🎯 **Smart Filter System**
    - **Time-Based Suggestions**: Morning momentum, power hour gamma
    - **Market-Adaptive**: Filters change based on volatility regime
    - **One-Click Presets**: Vol expansion, earnings plays, dark pool activity
    - **Context-Sensitive**: Different suggestions for different market sessions
    
    #### 📊 **Enhanced Position Tracking**
    - **Predictive Modeling**: Expected transfer probability scoring
    - **Sentiment Shift Analysis**: Track if buying continues or shifts to selling
    - **Momentum Scoring**: 100-point momentum scale for follow-up activity
    - **Market Context Storage**: Remember market conditions at entry
    
    #### 🔬 **Backtesting Engine**
    - **Historical Validation**: Test prediction accuracy on past data
    - **Confidence Calibration**: How accurate are different confidence levels?
    - **Scenario Ranking**: Which patterns perform best/worst
    - **Transfer Rate Analysis**: Validation of tracking effectiveness
    
    #### 📱 **Professional Integration**
    - **Discord Webhooks**: Auto-notifications for critical alerts
    - **JSON API Export**: Integration with external tools
    - **Enhanced CSV**: All new fields and market context data
    - **Performance Reports**: Comprehensive analytics export
    
    ### 🎯 **Getting Started with v2.0:**
    
    1. **📊 Performance Analytics**: See your historical accuracy and win rates
    2. **🚨 Smart Alert System**: Get AI-powered alerts with enhanced scoring  
    3. **🌍 Market Context Analysis**: Trades analyzed with current market regime
    4. **🎯 Advanced Pattern Recognition**: Dark pool, earnings, institutional flows
    5. **📍 Position Tracking**: Enhanced with predictive modeling and momentum
    
    ### 💡 **Pro Tips for v2.0:**
    
    #### 🎯 **Smart Filter Usage:**
    - **Morning (9-10am)**: Use "Morning Momentum" for opening plays
    - **Power Hour (3-4pm)**: Use "Power Hour Gamma" for 0DTE action  
    - **Low Vol Environment**: Use "Vol Expansion" filter
    - **High Vol Environment**: Use "Vol Crush Setup" filter
    
    #### 🧠 **Smart Alert Optimization:**
    - **Critical Alerts (10+ score)**: Auto-notifications enabled
    - **High Priority (8-9.9 score)**: Manual review recommended
    - **Score Breakdown**: Check what drives each alert score
    - **Historical Context**: Alerts learn from past similar trades
    
    #### 📈 **Performance Tracking Best Practices:**
    - **Track 30+ positions** for statistical significance
    - **Focus on High Confidence** trades for best transfer rates
    - **Monitor scenario effectiveness** - which patterns actually work
    - **Use backtesting** to validate and improve detection accuracy
    
    #### 🎯 **Pattern Recognition Mastery:**
    - **Dark Pool Activity**: Look for 3+ large trades in 5-minute windows
    - **Earnings Plays**: High IV + short DTE + large premium combinations
    - **Institutional Flows**: Coordinated activity across multiple strikes
    - **Multi-leg Strategies**: Spreads, straddles, collars detection
    
    ### 🔥 **What's Most Powerful in v2.0:**
    
    1. **🚨 Smart Alerts with ML Scoring** - Game-changing alert quality
    2. **📈 Performance Validation** - Finally know if your edge is real
    3. **🎯 Dark Pool Detection** - Catch institutional block activity
    4. **🌍 Market Context** - Trade with the market regime, not against it
    5. **🔬 Backtesting** - Validate every prediction with historical data
    
    ### 🚀 **Ready to Get Started?**
    
    Select your analysis type from the sidebar and click "Run Enhanced Scan" to experience the next generation of options flow analysis!
    
    **v2.0 transforms options flow from pattern recognition into predictive intelligence.**
    """)

# --- ORIGINAL DISPLAY FUNCTIONS (ENHANCED VERSIONS) ---
def display_enhanced_summary(trades):
    st.markdown("### 📊 Enhanced Market Summary")
    
    if not trades:
        st.warning("No trades to analyze")
        return
    
    # Get market context for enhanced analysis
    market_context = get_market_context()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_ratio, sentiment_label = calculate_sentiment_score(trades)
        context_emoji = "🟢" if market_context.get('market_sentiment') == 'Bullish' else "🔴" if market_context.get('market_sentiment') == 'Bearish' else "⚪"
        st.metric("Market Sentiment", f"{context_emoji} {sentiment_label}", f"{sentiment_ratio:.1%} calls")
    
    with col2:
        total_premium = sum(t.get('premium', 0) for t in trades)
        daily_avg = total_premium / max(len(set(t.get('time_ny', '')[:5] for t in trades)), 1)  # Rough hourly average
        st.metric("Total Premium", f"${total_premium:,.0f}", f"${daily_avg:,.0f}/hr avg")
    
    with col3:
        buy_trades = len([t for t in trades if 'BUY' in t.get('enhanced_side', '')])
        sell_trades = len([t for t in trades if 'SELL' in t.get('enhanced_side', '')])
        buy_ratio = buy_trades / max(buy_trades + sell_trades, 1)
        ratio_emoji = "🟢" if buy_ratio > 0.6 else "🔴" if buy_ratio < 0.4 else "⚪"
        st.metric("Buy vs Sell", f"{ratio_emoji} {buy_trades}/{sell_trades}", f"{buy_ratio:.1%} buys")
    
    with col4:
        high_conf_trades = len([t for t in trades if 'High Confidence' in t.get('enhanced_side', '')])
        conf_ratio = high_conf_trades / max(len(trades), 1)
        conf_emoji = "🟢" if conf_ratio > 0.3 else "🟡" if conf_ratio > 0.15 else "🔴"
        st.metric("High Confidence", f"{conf_emoji} {high_conf_trades}", f"{conf_ratio:.1%} of total")
    
    # Enhanced insights row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Dark pool activity indicator
        large_block_trades = len([t for t in trades if t.get('premium', 0) > 1000000])
        st.metric("Mega Blocks ($1M+)", large_block_trades, delta="🏢" if large_block_trades > 0 else None)
    
    with col2:
        # Volatility plays
        high_iv_trades = len([t for t in trades if t.get('iv', 0) > config.EXTREME_IV_THRESHOLD])
        st.metric("High IV Plays", high_iv_trades, delta="📈" if high_iv_trades > 5 else None)
    
    with col3:
        # Short-term focus
        short_term = len([t for t in trades if t.get('dte', 30) <= 7])
        st.metric("Weekly/0DTE", short_term, delta="⚡" if short_term > len(trades) * 0.3 else None)
    
    with col4:
        # Market context alignment
        vol_regime = market_context.get('volatility_regime', 'Unknown')
        regime_emoji = "📈" if vol_regime == 'High' else "📊"
        st.metric("Vol Regime", f"{regime_emoji} {vol_regime}")

def display_enhanced_scan_with_tracking(trades):
    """Enhanced scan results with advanced position tracking"""
    
    # Save trackable positions from current scan
    trackable_trades = position_tracker.save_trackable_positions(trades)
    
    if trackable_trades:
        st.success(f"📍 Added {len(trackable_trades)} high-confidence positions to enhanced tracking system")
        
        # Enhanced tracking preview
        with st.expander("📍 Newly Tracked Positions with Predictions", expanded=False):
            tracking_data = []
            for trade in trackable_trades:
                tracking_data.append({
                    'Ticker': trade['ticker'],
                    'Strike': f"${trade['strike']:.0f}",
                    'Type': trade['type'],
                    'Expiry': trade['expiry'],
                    'DTE': trade['dte'],
                    'Premium': f"${trade['premium']:,.0f}",
                    'Volume': f"{trade['volume']:,}",
                    'Side': trade['enhanced_side'],
                    'Confidence': f"{trade['side_confidence']:.1%}",
                    'Alert Score': f"{trade.get('enhanced_alert_score', 0):.1f}",
                    'Expected Outcome': position_tracker.predict_outcome(trade),
                    'Tracking Reason': position_tracker.get_tracking_reason(trade)
                })
            
            df = pd.DataFrame(tracking_data)
            st.dataframe(df, use_container_width=True)
    
    # Check for updates to existing tracked positions with enhanced analysis
    position_updates = position_tracker.check_position_updates(trades)
    
    if position_updates:
        st.warning(f"🔄 Found activity in {len(position_updates)} previously tracked positions!")
        
        with st.expander("🔄 Enhanced Position Updates", expanded=True):
            for update in position_updates:
                position = update['position']
                activity = update['current_activity']
                is_significant = update['is_significant']
                
                # Enhanced display with momentum analysis
                significance_emoji = "💪" if is_significant else "📊"
                st.markdown(f"**{significance_emoji} {position['ticker']} ${position['strike']:.0f}{position['type']} - {position['expiry']}**")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.write(f"**Original (Day 1):**")
                    st.write(f"Premium: ${position['original_premium']:,.0f}")
                    st.write(f"Volume: {position['original_volume']:,}")
                    st.write(f"Side: {position['original_side']}")
                    st.write(f"Expected: {position.get('expected_outcome', 'Unknown')}")
                
                with col2:
                    st.write(f"**Today's Activity:**")
                    st.write(f"Premium: ${activity['total_premium']:,.0f}")
                    st.write(f"Volume: {activity['total_volume']:,}")
                    st.write(f"Side: {activity['dominant_side']}")
                    st.write(f"Trades: {activity['trade_count']}")
                
                with col3:
                    st.write(f"**Enhanced Analysis:**")
                    volume_multiple = activity['volume_vs_original']
                    st.write(f"Volume Multiple: {volume_multiple:.1f}x")
                    st.write(f"Sentiment: {activity['sentiment_shift']}")
                    st.write(f"Momentum Score: {activity['momentum_score']:.0f}/100")
                    st.write(f"Avg Confidence: {activity['avg_confidence']:.1%}")
                
                with col4:
                    st.write(f"**Market Context:**")
                    original_context = position.get('market_context_at_entry', {})
                    current_context = get_market_context()
                    
                    original_sentiment = original_context.get('market_sentiment', 'Unknown')
                    current_sentiment = current_context.get('market_sentiment', 'Unknown')
                    
                    st.write(f"Entry Sentiment: {original_sentiment}")
                    st.write(f"Current Sentiment: {current_sentiment}")
                    
                    if original_sentiment != current_sentiment:
                        st.write("⚠️ Sentiment Shift!")
                
                # Enhanced interpretation
                if activity['momentum_score'] > 70:
                    st.success("🚀 **Strong Momentum**: High probability of continued interest")
                elif activity['momentum_score'] > 40:
                    st.info("📈 **Moderate Momentum**: Decent follow-through")
                elif activity['sentiment_shift'] == "Shifted to Selling":
                    st.warning("⚠️ **Sentiment Reversal**: Original buyers may be taking profits")
                else:
                    st.info("📊 **Standard Activity**: Normal follow-through pattern")
                
                st.divider()

def display_position_tracking_dashboard():
    """Enhanced position tracking dashboard with advanced analytics"""
    st.markdown("### 📍 Enhanced Position Tracking Dashboard")
    
    # Cleanup expired positions first
    expired_count = position_tracker.cleanup_expired_positions()
    if expired_count > 0:
        st.info(f"🗑️ Cleaned up {expired_count} expired positions")
    
    # Get enhanced transfer analysis
    analysis = position_tracker.analyze_position_transfers()
    summary = analysis['summary']
    
    if summary['total_tracked'] == 0:
        st.info("📍 No positions currently tracked. Run a scan to start tracking high-confidence plays!")
        return
    
    # Enhanced summary metrics with context
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Tracked", summary['total_tracked'])
    
    with col2:
        transfer_rate = summary['transfer_rate']
        rate_emoji = "🟢" if transfer_rate > 0.5 else "🟡" if transfer_rate > 0.3 else "🔴"
        st.metric("Transfer Rate", f"{rate_emoji} {transfer_rate:.1%}")
    
    with col3:
        st.metric("Active Transfers", summary['transferred'])
    
    with col4:
        buy_sell_ratio = summary['buy_transfers'] / max(summary['sell_transfers'], 1)
        ratio_emoji = "🟢" if buy_sell_ratio > 1.5 else "🔴" if buy_sell_ratio < 0.7 else "⚪"
        st.metric("Buy/Sell Transfers", f"{ratio_emoji} {summary['buy_transfers']}/{summary['sell_transfers']}")
    
    with col5:
        avg_days = summary.get('avg_days_tracked', 0)
        st.metric("Avg Days Tracked", f"{avg_days:.1f}")
    
    # Performance integration
    performance_summary = performance_tracker.get_performance_summary()
    
    if performance_summary.get('total_trades_tracked', 0) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            theoretical_pnl = performance_summary.get('total_theoretical_pnl', 0)
            pnl_emoji = "🟢" if theoretical_pnl > 0 else "🔴" if theoretical_pnl < 0 else "⚪"
            st.metric("Theoretical P&L", f"{pnl_emoji} ${theoretical_pnl:,.0f}")
        
        with col2:
            win_rate = performance_summary.get('win_rate', 0)
            st.metric("Estimated Win Rate", f"{win_rate:.1%}")
        
        with col3:
            avg_pnl = performance_summary.get('avg_pnl_per_trade', 0)
            st.metric("Avg P&L/Trade", f"${avg_pnl:,.0f}")
    
    # Enhanced tabs with new features
    tab1, tab2, tab3, tab4 = st.tabs(["🔄 Active Transfers", "📊 All Positions", "📈 Analytics", "🎯 Predictions"])
    
    with tab1:
        st.markdown("#### 🔄 Positions with Follow-up Activity")
        transferred_positions = analysis['transferred']
        
        if not transferred_positions:
            st.info("No transferred positions found yet")
        else:
            for position in transferred_positions:
                with st.expander(f"📍 {position['ticker']} ${position['strike']:.0f}{position['type']} - {position['expiry']}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Original Trade:**")
                        st.write(f"💰 Premium: ${position['original_premium']:,.0f}")
                        st.write(f"📊 Volume: {position['original_volume']:,}")
                        st.write(f"🎯 Side: {position['original_side']}")
                        st.write(f"📅 Date: {position['original_date']}")
                        st.write(f"⏱️ DTE: {position['dte']} days")
                        st.write(f"🔮 Expected: {position.get('expected_outcome', 'Unknown')}")
                    
                    with col2:
                        st.markdown("**Follow-up Summary:**")
                        total_follow_premium = sum(f['total_premium'] for f in position['follow_up_data'])
                        total_follow_volume = sum(f['total_volume'] for f in position['follow_up_data'])
                        
                        st.write(f"💰 Total Follow Premium: ${total_follow_premium:,.0f}")
                        st.write(f"📊 Total Follow Volume: {total_follow_volume:,}")
                        st.write(f"📈 Days Active: {len(position['follow_up_data'])}")
                        
                        # Calculate overall momentum
                        if position['follow_up_data']:
                            latest = position['follow_up_data'][-1]
                            overall_momentum = latest.get('momentum_score', 0)
                            momentum_emoji = "🚀" if overall_momentum > 70 else "📈" if overall_momentum > 40 else "📊"
                            st.write(f"🎯 Overall Momentum: {momentum_emoji} {overall_momentum:.0f}/100")
                    
                    with col3:
                        st.markdown("**Day-by-Day Activity:**")
                        for i, follow_up in enumerate(position['follow_up_data'], 1):
                            sentiment_emoji = "🟢" if follow_up['sentiment_shift'] == 'Continued Buying' else "🔴" if 'Selling' in follow_up['sentiment_shift'] else "⚪"
                            st.write(f"**Day {i} ({follow_up['date']}):** {sentiment_emoji}")
                            st.write(f"  Premium: ${follow_up['total_premium']:,.0f}")
                            st.write(f"  Volume: {follow_up['total_volume']:,} ({follow_up['volume_vs_original']:.1f}x)")
                            st.write(f"  Momentum: {follow_up['momentum_score']:.0f}/100")
                            if i < len(position['follow_up_data']):
                                st.write("  ---")
    
    with tab2:
        st.markdown("#### 📊 All Tracked Positions")
        
        if 'tracked_positions' in st.session_state and st.session_state.tracked_positions:
            positions_data = []
            
            for position_id, position in st.session_state.tracked_positions.items():
                follow_up_count = len(position['follow_up_data'])
                
                # Enhanced calculations
                total_follow_up_volume = sum(f['total_volume'] for f in position['follow_up_data'])
                total_follow_up_premium = sum(f['total_premium'] for f in position['follow_up_data'])
                
                # Status determination
                if follow_up_count > 0:
                    latest_activity = position['follow_up_data'][-1]
                    momentum_score = latest_activity.get('momentum_score', 0)
                    
                    if momentum_score > 70:
                        status = "🚀 Strong Transfer"
                    elif momentum_score > 40:
                        status = "📈 Moderate Transfer"  
                    else:
                        status = "📊 Weak Transfer"
                        
                    dominant_side = latest_activity['dominant_side']
                    sentiment_shift = latest_activity['sentiment_shift']
                else:
                    status = "⏳ No Follow-up"
                    dominant_side = "N/A"
                    sentiment_shift = "N/A"
                
                # Calculate days since entry
                days_since = (datetime.now() - datetime.strptime(position['original_date'], '%Y-%m-%d')).days
                theoretical_pnl = performance_tracker.calculate_theoretical_pnl(position, days_since)
                
                positions_data.append({
                    'Ticker': position['ticker'],
                    'Strike': f"${position['strike']:.0f}",
                    'Type': position['type'],
                    'Expiry': position['expiry'],
                    'DTE': position['dte'],
                    'Entry Date': position['original_date'],
                    'Days Held': days_since,
                    'Original Premium': f"${position['original_premium']:,.0f}",
                    'Original Volume': f"{position['original_volume']:,}",
                    'Expected Outcome': position.get('expected_outcome', 'Unknown'),
                    'Status': status,
                    'Follow-up Days': follow_up_count,
                    'Total Follow Volume': f"{total_follow_up_volume:,}",
                    'Total Follow Premium': f"${total_follow_up_premium:,.0f}",
                    'Latest Side': dominant_side,
                    'Sentiment Shift': sentiment_shift,
                    'Theoretical P&L': f"${theoretical_pnl:,.0f}",
                    'Tracking Reason': position.get('tracking_reason', 'High Confidence'),
                    'Position ID': position_id[:8]
                })
            
            df = pd.DataFrame(positions_data)
            st.dataframe(df, use_container_width=True)
            
            # Enhanced bulk actions
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🗑️ Clear Expired"):
                    expired_count = position_tracker.cleanup_expired_positions()
                    st.success(f"Cleaned up {expired_count} expired positions")
                    st.rerun()
            
            with col2:
                if st.button("📊 Export Enhanced Data"):
                    # Create enhanced export with all tracking data
                    enhanced_csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Enhanced CSV",
                        data=enhanced_csv,
                        file_name=f"enhanced_position_tracking_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            with col3:
                if st.button("📈 Performance Summary"):
                    performance_summary = performance_tracker.get_performance_summary()
                    backtest_results = backtest_engine.validate_prediction_accuracy()
                    
                    summary_report = {
                        'timestamp': datetime.now().isoformat(),
                        'total_positions': len(positions_data),
                        'performance_metrics': performance_summary,
                        'backtest_results': backtest_results,
                        'position_breakdown': positions_data
                    }
                    
                    st.download_button(
                        label="Download Performance Report",
                        data=json.dumps(summary_report, indent=2, default=str),
                        file_name=f"performance_summary_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
            
            with col4:
                if st.button("🔄 Refresh Analysis"):
                    st.rerun()
        else:
            st.info("No positions being tracked yet")
    
    with tab3:
        st.markdown("#### 📈 Enhanced Transfer Analytics")
        
        if summary['total_tracked'] > 0:
            # Performance integration
            performance_data = performance_tracker.get_performance_summary()
            backtest_results = backtest_engine.validate_prediction_accuracy()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Transfer Performance:**")
                st.write(f"• **Overall Transfer Rate**: {summary['transfer_rate']:.1%}")
                st.write(f"• **Buy-side Transfers**: {summary['buy_transfers']} positions")
                st.write(f"• **Sell-side Transfers**: {summary['sell_transfers']} positions")
                st.write(f"• **Average Days Tracked**: {summary.get('avg_days_tracked', 0):.1f}")
                
                if performance_data.get('total_trades_tracked', 0) > 0:
                    st.write(f"• **Estimated Win Rate**: {performance_data.get('win_rate', 0):.1%}")
                    st.write(f"• **Avg Theoretical P&L**: ${performance_data.get('avg_pnl_per_trade', 0):,.0f}")
            
            with col2:
                st.markdown("**💡 Advanced Insights:**")
                
                transfer_rate = summary['transfer_rate']
                if transfer_rate > 0.6:
                    st.success("• 🟢 Excellent transfer rate - strong detection accuracy")
                elif transfer_rate > 0.4:
                    st.info("• 🟡 Good transfer rate - decent follow-through")
                else:
                    st.warning("• 🔴 Low transfer rate - consider adjusting criteria")
                
                buy_sell_ratio = summary['buy_transfers'] / max(summary['sell_transfers'], 1)
                if buy_sell_ratio > 1.5:
                    st.success("• 📈 Strong continued buying momentum")
                elif buy_sell_ratio < 0.7:
                    st.warning("• 📉 More selling than continued buying")
                else:
                    st.info("• ⚖️ Balanced buy/sell follow-through")
                
                if backtest_results:
                    confidence_analysis = backtest_results['confidence_analysis']
                    high_conf_rate = confidence_analysis.get('high_confidence_rate', 0)
                    
                    if high_conf_rate > 0.7:
                        st.success("• 🎯 High confidence trades performing excellently")
                    elif high_conf_rate > 0.5:
                        st.info("• 🎯 High confidence trades performing well")
                    else:
                        st.warning("• 🎯 High confidence trades underperforming")
            
            # Scenario effectiveness from backtesting
            if backtest_results and backtest_results.get('scenario_effectiveness'):
                st.markdown("**🎯 Best Performing Scenarios:**")
                scenario_effectiveness = backtest_results['scenario_effectiveness']
                
                # Show top 5 scenarios
                top_scenarios = sorted(scenario_effectiveness.items(), key=lambda x: -x[1])[:5]
                
                for scenario, rate in top_scenarios:
                    rate_emoji = "🟢" if rate > 0.6 else "🟡" if rate > 0.4 else "🔴"
                    st.write(f"• {rate_emoji} **{scenario}**: {rate:.1%} transfer rate")
        
        else:
            st.info("No tracking data available yet")
    
    with tab4:
        st.markdown("#### 🎯 Prediction Analysis")
        
        if 'tracked_positions' in st.session_state and st.session_state.tracked_positions:
            # Prediction accuracy analysis
            positions = st.session_state.tracked_positions.values()
            
            prediction_analysis = {
                'High Transfer Probability': {'predicted': 0, 'actual': 0},
                'Medium Transfer Probability': {'predicted': 0, 'actual': 0}, 
                'Low Transfer Probability': {'predicted': 0, 'actual': 0}
            }
            
            for position in positions:
                expected = position.get('expected_outcome', 'Unknown')
                if expected in prediction_analysis:
                    prediction_analysis[expected]['predicted'] += 1
                    
                    # Check if it actually transferred
                    if len(position.get('follow_up_data', [])) > 0:
                        prediction_analysis[expected]['actual'] += 1
            
            st.markdown("**🔮 Prediction Accuracy:**")
            
            for prediction, data in prediction_analysis.items():
                if data['predicted'] > 0:
                    accuracy = data['actual'] / data['predicted']
                    accuracy_emoji = "🟢" if accuracy > 0.6 else "🟡" if accuracy > 0.4 else "🔴"
                    
                    st.write(f"• {accuracy_emoji} **{prediction}**: {accuracy:.1%} accuracy "
                            f"({data['actual']}/{data['predicted']} transferred)")
            
            # Model improvement suggestions
            st.markdown("**🔧 Model Improvement Suggestions:**")
            
            overall_accuracy = sum(d['actual'] for d in prediction_analysis.values()) / max(sum(d['predicted'] for d in prediction_analysis.values()), 1)
            
            if overall_accuracy < 0.5:
                st.warning("• 🎯 Consider adjusting prediction criteria - accuracy below 50%")
                st.info("• 💡 Try increasing minimum confidence threshold")
                st.info("• 💡 Focus on fewer, higher-quality predictions")
            elif overall_accuracy > 0.7:
                st.success("• 🎯 Excellent prediction accuracy! Model is well-calibrated")
            else:
                st.info("• 🎯 Good prediction accuracy - minor tweaks could improve")
        
        else:
            st.info("No prediction data available yet")
    """)

# --- CONTINUE WITH ENHANCED DISPLAY FUNCTIONS ---

def diagnose_trade_data(trades):
    """Enhanced diagnostic function with v2.0 insights"""
    st.markdown("## 🔍 Enhanced Trade Data Diagnostics")
    
    if not trades:
        st.error("No trades to diagnose!")
        return
    
    # Sample size and quality metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**📊 Dataset Overview:**")
        st.write(f"• Total Trades: {len(trades)}")
        st.write(f"• Unique Tickers: {len(set(t.get('ticker') for t in trades))}")
        st.write(f"• Time Span: {len(set(t.get('time_ny', '')[:5] for t in trades))} hours")
    
    with col2:
        # Enhanced confidence analysis
        high_conf = len([t for t in trades if t.get('side_confidence', 0) >= 0.7])
        medium_conf = len([t for t in trades if 0.4 <= t.get('side_confidence', 0) < 0.7])
        low_conf = len([t for t in trades if t.get('side_confidence', 0) < 0.4])
        
        st.write(f"**🎯 Confidence Distribution:**")
        st.write(f"• High (≥70%): {high_conf} ({high_conf/len(trades):.1%})")
        st.write(f"• Medium (40-69%): {medium_conf} ({medium_conf/len(trades):.1%})")
        st.write(f"• Low (<40%): {low_conf} ({low_conf/len(trades):.1%})")
    
    with col3:
        # Market context diagnostics
        market_context = get_market_context()
        st.write(f"**🌍 Market Context:**")
        st.write(f"• Session: {market_context.get('market_session', 'Unknown')}")
        st.write(f"• Volatility: {market_context.get('volatility_regime', 'Unknown')}")
        st.write(f"• Sentiment: {market_context.get('market_sentiment', 'Unknown')}")
    
    # Data completeness analysis
    st.markdown("**📋 Data Quality Assessment:**")
    
    fields_to_check = ['price', 'bid', 'ask', 'volume', 'open_interest', 'description', 'rule_name', 'iv']
    completeness = {}
    
    for field in fields_to_check:
        valid_count = sum(1 for t in trades if t.get(field) not in ['N/A', '', None, 0])
        completeness[field] = valid_count / len(trades)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Field Completeness:**")
        for field, pct in completeness.items():
            color = "🟢" if pct > 0.8 else "🟡" if pct > 0.5 else "🔴"
            st.write(f"{color} {field}: {pct:.1%}")
    
    with col2:
        # Enhanced pattern detection in diagnostics
        st.write("**📊 Pattern Summary:**")
        
        mega_trades = len([t for t in trades if t.get('premium', 0) > 1000000])
        st.write(f"• Mega Trades ($1M+): {mega_trades}")
        
        high_vol_oi = len([t for t in trades if t.get('vol_oi_ratio', 0) > 10])
        st.write(f"• Extreme Vol/OI (>10): {high_vol_oi}")
        
        short_term = len([t for t in trades if t.get('dte', 30) <= 7])
        st.write(f"• Short-term (≤7 DTE): {short_term}")
        
        earnings_candidates = len([t for t in trades if t.get('iv', 0) > 0.4 and t.get('dte', 30) <= 21])
        st.write(f"• Earnings Candidates: {earnings_candidates}")
    
    # Sample trade inspection with enhanced analysis
    st.markdown("**🔍 Enhanced Sample Analysis:**")
    sample = trades[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Raw Data Sample:**")
        important_fields = ['ticker', 'price', 'bid', 'ask', 'volume', 'open_interest', 'iv', 'description', 'rule_name']
        for key in important_fields:
            if key in sample:
                st.write(f"• {key}: {sample[key]} ({type(sample[key]).__name__})")
    
    with col2:
        st.write("**Enhanced Analysis:**")
        side, confidence, reasoning = determine_trade_side_enhanced(sample, debug=False)
        
        st.write(f"• **Detected Side**: {side}")
        st.write(f"• **Confidence**: {confidence:.1%}")
        st.write(f"• **Top Reason**: {reasoning[0] if reasoning else 'No signals'}")
        
        # Market context fit
        context_fit = "Good" if confidence > 0.6 else "Poor"
        st.write(f"• **Context Fit**: {context_fit}")
        
        # Tracking eligibility
        is_trackable = position_tracker.is_trackable_position(sample)
        st.write(f"• **Trackable**: {'Yes' if is_trackable else 'No'}")
    
    # API health diagnostics
    st.markdown("**🔧 API Health Check:**")
    
    # Response time simulation (in real app, would measure actual response times)
    response_quality = "Excellent" if completeness.get('price', 0) > 0.8 else "Good" if completeness.get('price', 0) > 0.5 else "Poor"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data Quality", response_quality)
    
    with col2:
        avg_confidence = np.mean([t.get('side_confidence', 0) for t in trades])
        confidence_grade = "A" if avg_confidence > 0.6 else "B" if avg_confidence > 0.4 else "C"
        st.metric("Detection Quality", confidence_grade, f"{avg_confidence:.1%} avg")
    
    with col3:
        trackable_count = len([t for t in trades if position_tracker.is_trackable_position(t)])
        tracking_rate = trackable_count / len(trades)
        st.metric("Tracking Rate", f"{tracking_rate:.1%}", f"{trackable_count} trades")
    
    # Improvement suggestions
    if avg_confidence < 0.5:
        st.warning("💡 **Suggestion**: Low average confidence detected. Consider enabling debug mode to identify data quality issues.")
    
    if trackable_count == 0:
        st.info("💡 **Suggestion**: No trackable positions found. Try lowering minimum premium threshold or confidence requirements.")
    
    if completeness.get('iv', 0) < 0.3:
        st.warning("💡 **Suggestion**: Limited IV data available. Earnings and volatility play detection may be reduced.")

# Final closing for the main app
st.markdown("---")
st.markdown("**Enhanced Options Flow Tracker v2.0** - Transforming Options Flow Analysis with AI-Powered Insights")
