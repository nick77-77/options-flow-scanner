import streamlit as st
import httpx
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo
import sqlite3
import json
import os

# --- ENHANCED CONFIGURATION ---
@dataclass
class Config:
    # API Configuration
    UW_TOKEN: str = "e6e8601a-0746-4cec-a07d-c3eabfc13926"
    BASE_URL: str = "https://api.unusualwhales.com/api/option-trades/flow-alerts"
    TIMEOUT: int = 30
    
    # Filtering Configuration
    EXCLUDE_TICKERS: set = field(default_factory=lambda: {'MSTR', 'CRCL', 'COIN', 'META', 'NVDA', 'AMD', 'TSLA'})
    ALLOWED_TICKERS: set = field(default_factory=lambda: {'QQQ', 'SPY', 'IWM'})
    
    # Thresholds
    MIN_PREMIUM: int = 100000
    LIMIT: int = 500
    HIGH_IV_THRESHOLD: float = 0.30
    EXTREME_IV_THRESHOLD: float = 0.50
    HIGH_VOL_OI_RATIO: float = 5.0
    
    # Database Settings
    DB_PATH: str = "options_tracking.db"
    HIGH_CONFIDENCE_THRESHOLD: float = 0.7
    TRACKING_DAYS: int = 7  # Track for 7 days

class TradeSideAnalyzer:
    """Enhanced trade side detection"""
    
    def analyze_trade_side(self, trade_data: Dict) -> Tuple[str, float, List[str]]:
        """Comprehensive trade side analysis"""
        
        signals = []
        confidence_scores = []
        reasoning = []
        
        # Extract and validate data
        price = self._safe_float(trade_data.get('price', 0))
        bid = self._safe_float(trade_data.get('bid', 0))
        ask = self._safe_float(trade_data.get('ask', 0))
        volume = self._safe_float(trade_data.get('volume', 0))
        oi = max(self._safe_float(trade_data.get('open_interest', 1)), 1)
        
        # Method 1: Bid/Ask Analysis (Most reliable)
        if all([price > 0, bid > 0, ask > 0]) and ask > bid:
            mid = (bid + ask) / 2
            
            if price >= ask * 0.98:  # Within 2% of ask
                confidence_scores.append(0.9)
                reasoning.append(f"Price {price:.2f} at ask {ask:.2f}")
                signals.append("BUY")
            elif price <= bid * 1.02:  # Within 2% of bid
                confidence_scores.append(0.9)
                reasoning.append(f"Price {price:.2f} at bid {bid:.2f}")
                signals.append("SELL")
            elif price > mid * 1.02:
                confidence_scores.append(0.6)
                reasoning.append(f"Price {price:.2f} above mid {mid:.2f}")
                signals.append("BUY")
            elif price < mid * 0.98:
                confidence_scores.append(0.6)
                reasoning.append(f"Price {price:.2f} below mid {mid:.2f}")
                signals.append("SELL")
            else:
                confidence_scores.append(0.3)
                reasoning.append(f"Price {price:.2f} near mid {mid:.2f}")
                signals.append("NEUTRAL")
        
        # Method 2: Volume/OI Analysis
        vol_oi_ratio = volume / oi
        if vol_oi_ratio > 10:
            confidence_scores.append(0.8)
            reasoning.append(f"Extreme vol/OI ratio: {vol_oi_ratio:.1f}")
            signals.append("BUY")
        elif vol_oi_ratio > 5:
            confidence_scores.append(0.7)
            reasoning.append(f"High vol/OI ratio: {vol_oi_ratio:.1f}")
            signals.append("BUY")
        elif vol_oi_ratio > 2:
            confidence_scores.append(0.5)
            reasoning.append(f"Moderate vol/OI ratio: {vol_oi_ratio:.1f}")
            signals.append("BUY")
        
        # Method 3: Description Analysis
        description = trade_data.get('description', '').lower()
        rule_name = trade_data.get('rule_name', '').lower()
        
        strong_buy_keywords = ['sweep', 'aggressive', 'lifted', 'taken', 'market buy', 'block buy']
        if any(keyword in description for keyword in strong_buy_keywords):
            confidence_scores.append(0.8)
            reasoning.append("Strong buy keywords detected")
            signals.append("BUY")
        
        strong_sell_keywords = ['sold', 'offer hit', 'market sell', 'hit bid', 'block sell']
        if any(keyword in description for keyword in strong_sell_keywords):
            confidence_scores.append(0.8)
            reasoning.append("Strong sell keywords detected")
            signals.append("SELL")
        
        if 'ascending' in rule_name:
            confidence_scores.append(0.5)
            reasoning.append("Ascending fill pattern")
            signals.append("BUY")
        elif 'descending' in rule_name:
            confidence_scores.append(0.5)
            reasoning.append("Descending fill pattern")
            signals.append("SELL")
        
        # Aggregate results
        if not signals:
            return "UNKNOWN", 0.1, ["No clear signals detected"]
        
        # Find most common signal
        signal_counts = {}
        total_confidence = 0
        
        for signal, conf in zip(signals, confidence_scores):
            if signal in signal_counts:
                signal_counts[signal] += conf
            else:
                signal_counts[signal] = conf
            total_confidence += conf
        
        # Get the signal with highest total confidence
        final_signal = max(signal_counts.keys(), key=signal_counts.get)
        final_confidence = min(signal_counts[final_signal] / len([s for s in signals if s == final_signal]), 1.0)
        
        # Add confidence qualifier
        if final_confidence >= 0.8:
            final_signal += " (High Confidence)"
        elif final_confidence >= 0.6:
            final_signal += " (Medium Confidence)"
        elif final_confidence >= 0.4:
            final_signal += " (Low Confidence)"
        else:
            final_signal += " (Very Low Confidence)"
        
        return final_signal, final_confidence, reasoning
    
    def _safe_float(self, value: Union[str, int, float]) -> float:
        """Safely convert value to float"""
        if value in [None, 'N/A', '', 'NaN']:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

# --- DATABASE MANAGER ---
class OptionsTracker:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for tracking options"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS options_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                strike REAL NOT NULL,
                option_type TEXT NOT NULL,
                expiry TEXT NOT NULL,
                first_seen DATE NOT NULL,
                last_updated DATE NOT NULL,
                initial_oi INTEGER,
                current_oi INTEGER,
                initial_volume INTEGER,
                total_volume INTEGER,
                initial_premium REAL,
                total_premium REAL,
                confidence_level REAL,
                side TEXT,
                days_tracked INTEGER DEFAULT 1,
                oi_buildup REAL DEFAULT 0,
                volume_transfer REAL DEFAULT 0,
                status TEXT DEFAULT 'ACTIVE',
                UNIQUE(ticker, strike, option_type, expiry)
            )
        ''')
        
        # Create daily snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tracking_id INTEGER,
                date DATE NOT NULL,
                volume INTEGER,
                open_interest INTEGER,
                premium REAL,
                price REAL,
                confidence REAL,
                FOREIGN KEY (tracking_id) REFERENCES options_tracking (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_or_update_tracking(self, trade_data: Dict, analysis: Tuple[str, float, List[str]]):
        """Add new trade to tracking or update existing"""
        side, confidence, reasoning = analysis
        
        # Only track high confidence trades
        if confidence < 0.7:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        ticker = trade_data.get('ticker', '')
        strike = float(trade_data.get('strike', 0))
        option_type = trade_data.get('type', '')
        expiry = trade_data.get('expiry', '')
        volume = int(trade_data.get('volume', 0))
        oi = int(trade_data.get('open_interest', 0))
        premium = float(trade_data.get('premium', 0))
        price = float(trade_data.get('price', 0)) if trade_data.get('price') != 'N/A' else 0
        
        today = date.today().isoformat()
        
        # Check if this option is already being tracked
        cursor.execute('''
            SELECT id, initial_oi, total_volume, total_premium, days_tracked 
            FROM options_tracking 
            WHERE ticker = ? AND strike = ? AND option_type = ? AND expiry = ?
        ''', (ticker, strike, option_type, expiry))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing tracking
            tracking_id, initial_oi, total_volume, total_premium, days_tracked = existing
            
            new_total_volume = total_volume + volume
            new_total_premium = total_premium + premium
            oi_buildup = ((oi - initial_oi) / max(initial_oi, 1)) * 100
            volume_transfer = new_total_volume / max(oi, 1)
            
            cursor.execute('''
                UPDATE options_tracking 
                SET last_updated = ?, current_oi = ?, total_volume = ?, 
                    total_premium = ?, confidence_level = ?, days_tracked = ?,
                    oi_buildup = ?, volume_transfer = ?
                WHERE id = ?
            ''', (today, oi, new_total_volume, new_total_premium, confidence, 
                  days_tracked + 1, oi_buildup, volume_transfer, tracking_id))
        else:
            # Insert new tracking
            cursor.execute('''
                INSERT INTO options_tracking 
                (ticker, strike, option_type, expiry, first_seen, last_updated,
                 initial_oi, current_oi, initial_volume, total_volume, 
                 initial_premium, total_premium, confidence_level, side)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (ticker, strike, option_type, expiry, today, today,
                  oi, oi, volume, volume, premium, premium, confidence, side))
            
            tracking_id = cursor.lastrowid
        
        # Add daily snapshot
        cursor.execute('''
            INSERT INTO daily_snapshots 
            (tracking_id, date, volume, open_interest, premium, price, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (tracking_id, today, volume, oi, premium, price, confidence))
        
        conn.commit()
        conn.close()
    
    def get_tracked_options(self, days_back: int = 7) -> pd.DataFrame:
        """Get all tracked options from the last N days"""
        conn = sqlite3.connect(self.db_path)
        
        cutoff_date = (date.today() - timedelta(days=days_back)).isoformat()
        
        query = '''
            SELECT ticker, strike, option_type, expiry, first_seen, last_updated,
                   initial_oi, current_oi, total_volume, total_premium,
                   confidence_level, side, days_tracked, oi_buildup, volume_transfer
            FROM options_tracking 
            WHERE first_seen >= ? AND status = 'ACTIVE'
            ORDER BY oi_buildup DESC, total_premium DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        conn.close()
        
        return df
    
    def get_daily_progression(self, ticker: str, strike: float, option_type: str, expiry: str) -> pd.DataFrame:
        """Get daily progression for a specific option"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT ds.date, ds.volume, ds.open_interest, ds.premium, ds.price, ds.confidence
            FROM daily_snapshots ds
            JOIN options_tracking ot ON ds.tracking_id = ot.id
            WHERE ot.ticker = ? AND ot.strike = ? AND ot.option_type = ? AND ot.expiry = ?
            ORDER BY ds.date
        '''
        
        df = pd.read_sql_query(query, conn, params=(ticker, strike, option_type, expiry))
        conn.close()
        
        return df
    
    def cleanup_old_records(self, days_to_keep: int = 30):
        """Clean up old tracking records"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (date.today() - timedelta(days=days_to_keep)).isoformat()
        
        # Mark old records as inactive instead of deleting
        cursor.execute('''
            UPDATE options_tracking 
            SET status = 'INACTIVE' 
            WHERE last_updated < ?
        ''', (cutoff_date,))
        
        conn.commit()
        conn.close()

# --- MAIN PROCESSING FUNCTIONS ---
def parse_option_chain(opt_str):
    try:
        ticker = ''.join([c for c in opt_str if c.isalpha()])[:-1]
        date_start = len(ticker) + 1
        date_str = opt_str[date_start:date_start+6]
        expiry_date = date(2000 + int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6]))
        dte = (expiry_date - date.today()).days
        option_type = opt_str[date_start+6].upper()
        strike = int(opt_str[date_start+7:]) / 1000
        return ticker, expiry_date.strftime('%Y-%m-%d'), dte, option_type, strike
    except Exception:
        return None, None, None, None, None

def fetch_general_flow(config: Config):
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
    
    try:
        response = httpx.get(config.BASE_URL, headers=headers, params=params, timeout=config.TIMEOUT)
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code}")
            return []
        
        data = response.json().get('data', [])
        result = []
        analyzer = TradeSideAnalyzer()
        
        for trade in data:
            option_chain = trade.get('option_chain', '')
            ticker, expiry, dte, opt_type, strike = parse_option_chain(option_chain)

            if not ticker or ticker in config.EXCLUDE_TICKERS:
                continue

            premium = float(trade.get('total_premium', 0))
            if premium < config.MIN_PREMIUM:
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

            # Enhanced trade side analysis
            enhanced_side, confidence, reasoning = analyzer.analyze_trade_side(trade)

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
                'time_ny': ny_time_str,
                'rule_name': trade.get('rule_name', ''),
                'description': trade.get('description', ''),
                'underlying_price': trade.get('underlying_price', strike),
                'enhanced_side': enhanced_side,
                'side_confidence': confidence,
                'side_reasoning': reasoning,
                'vol_oi_ratio': float(trade.get('volume', 0)) / max(float(trade.get('open_interest', 1)), 1),
                'iv': float(trade.get('iv', 0)) if trade.get('iv') not in ['N/A', '', None] else 0,
                'bid': float(trade.get('bid', 0)) if trade.get('bid') not in ['N/A', '', None] else 0,
                'ask': float(trade.get('ask', 0)) if trade.get('ask') not in ['N/A', '', None] else 0
            }
            
            result.append(trade_data)

        return result

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return []

def display_enhanced_summary(trades):
    """Display enhanced summary with tracking info"""
    st.markdown("### 📊 Enhanced Market Summary")
    
    if not trades:
        st.warning("No trades to analyze")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_premium = sum(t.get('premium', 0) for t in trades)
        st.metric("Total Premium", f"${total_premium:,.0f}")
    
    with col2:
        buy_trades = len([t for t in trades if 'BUY' in t.get('enhanced_side', '')])
        sell_trades = len([t for t in trades if 'SELL' in t.get('enhanced_side', '')])
        st.metric("Buy vs Sell", f"{buy_trades}/{sell_trades}")
    
    with col3:
        high_conf_trades = len([t for t in trades if t.get('side_confidence', 0) >= 0.7])
        st.metric("High Confidence", high_conf_trades)
    
    with col4:
        avg_confidence = np.mean([t.get('side_confidence', 0) for t in trades]) if trades else 0
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")

def display_calls_and_puts_sections(trades, tracker):
    """Display the original 2-section layout for calls and puts"""
    st.markdown("### 📋 Options Flow Analysis")
    
    if not trades:
        st.info("No trades found")
        return
    
    # Separate calls and puts
    calls = [t for t in trades if t['type'] == 'C']
    puts = [t for t in trades if t['type'] == 'P']
    
    def create_trade_table(trade_list, trade_type_name):
        if not trade_list:
            st.info(f"No {trade_type_name.lower()} found")
            return
        
        # Sort by premium descending
        sorted_trades = sorted(trade_list, key=lambda x: x.get('premium', 0), reverse=True)
        
        table_data = []
        for trade in sorted_trades[:25]:  # Show top 25 per section
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
            
            # Calculate moneyness
            strike = trade.get('strike', 0)
            underlying = trade.get('underlying_price', strike)
            if underlying > 0:
                diff_pct = ((strike - underlying) / underlying) * 100
                if abs(diff_pct) < 2:
                    moneyness = "ATM"
                elif trade['type'] == 'C':
                    moneyness = f"OTM +{diff_pct:.1f}%" if diff_pct > 0 else f"ITM {diff_pct:.1f}%"
                else:
                    moneyness = f"OTM {abs(diff_pct):.1f}%" if diff_pct < 0 else f"ITM +{diff_pct:.1f}%"
            else:
                moneyness = "Unknown"
            
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
                'IV': f"{trade['iv']:.1%}" if trade['iv'] > 0 else "N/A",
                'Moneyness': moneyness,
                'Time': trade['time_ny']
            })
            
            # Add to tracking if high confidence
            if confidence >= 0.7:
                tracker.add_or_update_tracking(trade, (enhanced_side, confidence, trade.get('side_reasoning', [])))
        
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True)
    
    # Display in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🟢 CALLS")
        create_trade_table(calls, "Calls")
    
    with col2:
        st.markdown("#### 🔴 PUTS")
        create_trade_table(puts, "Puts")

def display_tracking_dashboard(tracker):
    """Display tracking dashboard for high confidence plays"""
    st.markdown("### 📈 High Confidence Tracking Dashboard")
    
    # Get tracked options
    tracked_df = tracker.get_tracked_options(days_back=7)
    
    if tracked_df.empty:
        st.info("No high confidence plays being tracked yet. Run analysis to start tracking.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tracked Positions", len(tracked_df))
    
    with col2:
        avg_buildup = tracked_df['oi_buildup'].mean() if len(tracked_df) > 0 else 0
        st.metric("Avg OI Buildup", f"{avg_buildup:.1f}%")
    
    with col3:
        total_premium = tracked_df['total_premium'].sum()
        st.metric("Total Tracked Premium", f"${total_premium:,.0f}")
    
    with col4:
        high_buildup = len(tracked_df[tracked_df['oi_buildup'] > 20])
        st.metric("High Buildup (>20%)", high_buildup)
    
    # Top OI buildup positions
    st.markdown("#### 🎯 Top Open Interest Buildup Positions")
    
    if len(tracked_df) > 0:
        # Prepare display data
        display_data = []
        for _, row in tracked_df.head(15).iterrows():
            # Side indicator
            side = row.get('side', 'UNKNOWN')
            if 'BUY' in str(side):
                side_emoji = "🟢"
            elif 'SELL' in str(side):
                side_emoji = "🔴"
            else:
                side_emoji = "⚪"
            
            # Buildup indicator
            buildup = row['oi_buildup']
            if buildup > 50:
                buildup_emoji = "🔥"
            elif buildup > 20:
                buildup_emoji = "📈"
            elif buildup > 0:
                buildup_emoji = "📊"
            else:
                buildup_emoji = "📉"
            
            display_data.append({
                'Ticker': row['ticker'],
                'Type': row['option_type'],
                'Side': f"{side_emoji} {side}",
                'Strike': f"${row['strike']:.0f}",
                'Expiry': row['expiry'],
                'Days Tracked': row['days_tracked'],
                'Initial OI': f"{row['initial_oi']:,}",
                'Current OI': f"{row['current_oi']:,}",
                'OI Buildup': f"{buildup_emoji} {buildup:.1f}%",
                'Volume Transfer': f"{row['volume_transfer']:.1f}",
                'Total Premium': f"${row['total_premium']:,.0f}",
                'Confidence': f"{row['confidence_level']:.0%}",
                'First Seen': row['first_seen'],
                'Last Updated': row['last_updated']
            })
        
        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True)
        
        # Volume transfer analysis
        st.markdown("#### 📊 Volume Transfer Analysis")
        st.write("**Volume Transfer Ratio**: Total Volume / Current Open Interest")
        st.write("• **> 2.0**: High volume transfer, significant new positions")
        st.write("• **1.0-2.0**: Moderate activity, some position building")
        st.write("• **< 1.0**: Low activity relative to open interest")
        
        # Highlight significant buildups
        significant_buildups = tracked_df[tracked_df['oi_buildup'] > 25]
        if not significant_buildups.empty:
            st.markdown("#### 🔥 Significant OI Buildups (>25%)")
            for _, row in significant_buildups.head(5).iterrows():
                st.write(f"**{row['ticker']} {row['strike']:.0f}{row['option_type']}** "
                        f"({row['expiry']}) - {row['oi_buildup']:.1f}% buildup over {row['days_tracked']} days")
    
    # Option to view detailed progression for specific strikes
    if st.checkbox("Show Detailed Position Progression"):
        if not tracked_df.empty:
            # Select a position to analyze
            position_options = []
            for _, row in tracked_df.iterrows():
                option_key = f"{row['ticker']} {row['strike']:.0f}{row['option_type']} {row['expiry']}"
                position_options.append((option_key, row))
            
            if position_options:
                selected_option = st.selectbox(
                    "Select position to analyze:",
                    options=range(len(position_options)),
                    format_func=lambda x: position_options[x][0]
                )
                
                if selected_option is not None:
                    selected_row = position_options[selected_option][1]
                    
                    # Get daily progression
                    progression_df = tracker.get_daily_progression(
                        selected_row['ticker'],
                        selected_row['strike'],
                        selected_row['option_type'],
                        selected_row['expiry']
                    )
                    
                    if not progression_df.empty:
                        st.markdown(f"#### 📈 Daily Progression: {position_options[selected_option][0]}")
                        
                        # Format progression data
                        prog_display = []
                        for _, day_row in progression_df.iterrows():
                            prog_display.append({
                                'Date': day_row['date'],
                                'Volume': f"{day_row['volume']:,}",
                                'Open Interest': f"{day_row['open_interest']:,}",
                                'Premium': f"${day_row['premium']:,.0f}",
                                'Price': f"${day_row['price']:.2f}" if day_row['price'] > 0 else "N/A",
                                'Confidence': f"{day_row['confidence']:.0%}"
                            })
                        
                        prog_df = pd.DataFrame(prog_display)
                        st.dataframe(prog_df, use_container_width=True)
                    else:
                        st.info("No daily progression data available for this position.")

def apply_filters(trades, premium_filter, confidence_filter, side_filter, dte_filter):
    """Apply filtering logic"""
    filtered = trades
    
    # Premium filter
    if premium_filter != "All":
        if premium_filter == "Under $250K":
            filtered = [t for t in filtered if t.get('premium', 0) < 250000]
        elif premium_filter == "$250K - $500K":
            filtered = [t for t in filtered if 250000 <= t.get('premium', 0) < 500000]
        elif premium_filter == "Above $500K":
            filtered = [t for t in filtered if t.get('premium', 0) >= 500000]
        elif premium_filter == "Above $1M":
            filtered = [t for t in filtered if t.get('premium', 0) >= 1000000]
    
    # Confidence filter
    if confidence_filter != "All":
        if confidence_filter == "High Confidence Only":
            filtered = [t for t in filtered if t.get('side_confidence', 0) >= 0.7]
        elif confidence_filter == "Medium+ Confidence":
            filtered = [t for t in filtered if t.get('side_confidence', 0) >= 0.4]
    
    # Side filter
    if side_filter != "All":
        if side_filter == "Buy Only":
            filtered = [t for t in filtered if 'BUY' in t.get('enhanced_side', '')]
        elif side_filter == "Sell Only":
            filtered = [t for t in filtered if 'SELL' in t.get('enhanced_side', '')]
    
    # DTE filter
    if dte_filter != "All":
        if dte_filter == "0DTE":
            filtered = [t for t in filtered if t.get('dte', 0) == 0]
        elif dte_filter == "Weekly (≤7d)":
            filtered = [t for t in filtered if t.get('dte', 0) <= 7]
        elif dte_filter == "Monthly (≤30d)":
            filtered = [t for t in filtered if t.get('dte', 0) <= 30]
    
    return filtered

def save_to_csv(trades, filename_prefix):
    if not trades:
        st.warning("No data to save")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.csv"
    
    csv_data = []
    for trade in trades:
        row = trade.copy()
        if isinstance(row.get('side_reasoning'), list):
            row['side_reasoning'] = ', '.join(row['side_reasoning'])
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

# --- MAIN APPLICATION ---
def main():
    st.set_page_config(
        page_title="Enhanced Options Flow Tracker",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Enhanced Options Flow Tracker with OI Tracking")
    st.markdown("### Real-time Options Analysis with High Confidence Position Tracking")
    
    # Initialize components
    config = Config()
    if 'UW_TOKEN' in st.secrets:
        config.UW_TOKEN = st.secrets['UW_TOKEN']
    
    # Initialize database tracker
    tracker = OptionsTracker(config.DB_PATH)
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            [
                "🔍 Main Flow Analysis",
                "📈 High Confidence Tracking",
                "📊 Combined Analysis"
            ]
        )
        
        # Filters
        st.markdown("### 🎯 Filters")
        
        premium_filter = st.selectbox(
            "Premium Range:",
            ["All", "Under $250K", "$250K - $500K", "Above $500K", "Above $1M"],
            index=0
        )
        
        confidence_filter = st.selectbox(
            "Confidence Level:",
            ["All", "High Confidence Only", "Medium+ Confidence"],
            index=0
        )
        
        side_filter = st.selectbox(
            "Trade Side:",
            ["All", "Buy Only", "Sell Only"],
            index=0
        )
        
        dte_filter = st.selectbox(
            "Days to Expiry:",
            ["All", "0DTE", "Weekly (≤7d)", "Monthly (≤30d)"],
            index=0
        )
        
        # Database management
        st.markdown("### 🗄️ Database Management")
        if st.button("🧹 Cleanup Old Records", help="Remove records older than 30 days"):
            tracker.cleanup_old_records()
            st.success("Old records cleaned up!")
        
        # Action button
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Main content
    if run_analysis:
        with st.spinner("🔄 Fetching and analyzing options flow data..."):
            # Fetch data
            trades = fetch_general_flow(config)
            
            if not trades:
                st.error("❌ No data received. Please check your connection and API token.")
                return
            
            # Apply filters
            filtered_trades = apply_filters(trades, premium_filter, confidence_filter, side_filter, dte_filter)
            
            st.info(f"📊 Found {len(trades)} trades → {len(filtered_trades)} after filtering")
            
            # Display based on analysis type
            if "Main Flow" in analysis_type:
                display_enhanced_summary(filtered_trades)
                display_calls_and_puts_sections(filtered_trades, tracker)
                
                with st.expander("💾 Export Data"):
                    save_to_csv(filtered_trades, "main_flow_analysis")
            
            elif "High Confidence Tracking" in analysis_type:
                display_tracking_dashboard(tracker)
            
            elif "Combined Analysis" in analysis_type:
                display_enhanced_summary(filtered_trades)
                display_calls_and_puts_sections(filtered_trades, tracker)
                st.divider()
                display_tracking_dashboard(tracker)
                
                with st.expander("💾 Export Data"):
                    save_to_csv(filtered_trades, "combined_analysis")
    
    else:
        st.markdown("""
        ## Welcome to Enhanced Options Flow Tracker! 👋
        
        ### 🆕 New Features:
        
        #### 📈 **High Confidence Position Tracking**
        - **Automatic Database Storage**: High confidence trades (70%+) are automatically tracked
        - **Open Interest Buildup Monitoring**: Track OI changes over multiple days
        - **Volume Transfer Analysis**: Monitor volume relative to open interest
        - **Daily Progression Tracking**: See how positions develop over time
        
        #### 🎯 **Key Tracking Metrics**:
        - **OI Buildup %**: Percentage increase in open interest since first detection
        - **Volume Transfer Ratio**: Total volume divided by current open interest
        - **Days Tracked**: How long we've been monitoring this position
        - **Confidence Level**: Average confidence of trade direction detection
        
        #### 🔍 **Analysis Types**:
        
        **🔍 Main Flow Analysis**
        - Traditional calls/puts sections layout
        - Enhanced buy/sell detection
        - Automatic tracking of high confidence plays
        
        **📈 High Confidence Tracking**
        - View all tracked positions from past 7 days
        - See OI buildup and volume transfer patterns
        - Detailed daily progression for specific strikes
        
        **📊 Combined Analysis**
        - Both main flow and tracking in one view
        - Complete market overview with historical context
        
        ### 🎛️ **Enhanced Filters**:
        - **Premium Range**: Filter by trade size
        - **Confidence Level**: Focus on high-quality signals
        - **Trade Side**: Buy-only or sell-only analysis
        - **Time to Expiry**: From 0DTE to monthly plays
        
        ### 📊 **Understanding Volume Transfer**:
        - **High (>2.0)**: Significant new position building
        - **Moderate (1.0-2.0)**: Normal activity levels
        - **Low (<1.0)**: Minimal new activity relative to existing OI
        
        **Ready to start tracking? Select your analysis type and click 'Run Analysis'!**
        """)

if __name__ == "__main__":
    main()
