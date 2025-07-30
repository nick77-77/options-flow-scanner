import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import sqlite3
import httpx
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from zoneinfo import ZoneInfo

# --- OI TRACKING DATABASE SETUP ---
def init_oi_tracking_db():
    """Initialize SQLite database for OI tracking"""
    conn = sqlite3.connect('oi_tracking.db')
    cursor = conn.cursor()
    
    # Create table for tracked trades
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tracked_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            option_chain TEXT NOT NULL,
            strike REAL NOT NULL,
            expiry DATE NOT NULL,
            option_type TEXT NOT NULL,
            trade_date DATE NOT NULL,
            trade_time TEXT NOT NULL,
            enhanced_side TEXT NOT NULL,
            side_confidence REAL NOT NULL,
            premium REAL NOT NULL,
            volume INTEGER NOT NULL,
            initial_oi INTEGER NOT NULL,
            predicted_oi_change TEXT NOT NULL,
            alert_score INTEGER DEFAULT 0,
            scenarios TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create table for daily OI snapshots
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS oi_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            option_chain TEXT NOT NULL,
            strike REAL NOT NULL,
            expiry DATE NOT NULL,
            option_type TEXT NOT NULL,
            snapshot_date DATE NOT NULL,
            open_interest INTEGER NOT NULL,
            volume INTEGER DEFAULT 0,
            price REAL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(option_chain, snapshot_date)
        )
    ''')
    
    # Create table for tracking results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tracking_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tracked_trade_id INTEGER NOT NULL,
            days_tracked INTEGER NOT NULL,
            initial_oi INTEGER NOT NULL,
            current_oi INTEGER NOT NULL,
            oi_change INTEGER NOT NULL,
            oi_change_pct REAL NOT NULL,
            prediction_accuracy TEXT NOT NULL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (tracked_trade_id) REFERENCES tracked_trades (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def save_high_confidence_trades(trades):
    """Save high-confidence trades to tracking database"""
    if not trades:
        return 0
    
    conn = sqlite3.connect('oi_tracking.db')
    cursor = conn.cursor()
    
    saved_count = 0
    for trade in trades:
        # Only save high-confidence trades (70%+ confidence)
        if trade.get('side_confidence', 0) < 0.7:
            continue
            
        # Skip if already exists
        cursor.execute('''
            SELECT id FROM tracked_trades 
            WHERE option_chain = ? AND trade_date = DATE(?)
        ''', (trade.get('option', ''), trade.get('time_utc', '')[:10]))
        
        if cursor.fetchone():
            continue
            
        # Predict OI change based on our analysis
        vol_oi_ratio = trade.get('vol_oi_ratio', 0)
        enhanced_side = trade.get('enhanced_side', '')
        volume = trade.get('volume', 0)
        
        if vol_oi_ratio > 5 and 'BUY' in enhanced_side:
            predicted_change = "Major Increase (5x+ ratio)"
        elif vol_oi_ratio > 2 and 'BUY' in enhanced_side:
            predicted_change = "Moderate Increase (2-5x ratio)"
        elif 'BUY' in enhanced_side and volume > 100:
            predicted_change = "Small Increase (buying detected)"
        elif 'SELL' in enhanced_side and vol_oi_ratio < 0.5:
            predicted_change = "Decrease (selling detected)"
        else:
            predicted_change = "Minimal Change"
        
        try:
            cursor.execute('''
                INSERT INTO tracked_trades (
                    ticker, option_chain, strike, expiry, option_type,
                    trade_date, trade_time, enhanced_side, side_confidence,
                    premium, volume, initial_oi, predicted_oi_change,
                    alert_score, scenarios
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.get('ticker', ''),
                trade.get('option', ''),
                trade.get('strike', 0),
                trade.get('expiry', ''),
                trade.get('type', ''),
                trade.get('time_utc', '')[:10],
                trade.get('time_ny', ''),
                trade.get('enhanced_side', ''),
                trade.get('side_confidence', 0),
                trade.get('premium', 0),
                trade.get('volume', 0),
                trade.get('open_interest', 0),
                predicted_change,
                trade.get('alert_score', 0),
                ', '.join(trade.get('scenarios', []))
            ))
            saved_count += 1
        except Exception as e:
            st.error(f"Error saving trade: {e}")
            continue
    
    conn.commit()
    conn.close()
    return saved_count

def fetch_current_oi_data(option_chains):
    """Fetch current OI data for tracked options"""
    if not option_chains:
        return {}
    
    # This would connect to your options data API
    # For now, we'll simulate the API call structure
    headers = {
        'Accept': 'application/json, text/plain',
        'Authorization': st.secrets.get("UW_TOKEN", "your-token-here")
    }
    
    current_oi_data = {}
    
    # In a real implementation, you'd batch these requests
    for option_chain in option_chains[:10]:  # Limit for demo
        try:
            # Simulated API call - replace with actual endpoint
            # url = f"https://api.unusualwhales.com/api/options/{option_chain}/current"
            # response = httpx.get(url, headers=headers, timeout=30)
            
            # For demo, we'll simulate some OI changes
            # In real implementation, parse the actual API response
            import random
            simulated_oi = random.randint(100, 5000)
            simulated_volume = random.randint(0, 500)
            simulated_price = random.uniform(0.5, 50.0)
            
            current_oi_data[option_chain] = {
                'open_interest': simulated_oi,
                'volume': simulated_volume,
                'price': simulated_price,
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            st.warning(f"Could not fetch OI for {option_chain}: {e}")
            continue
    
    return current_oi_data

def update_oi_snapshots(current_oi_data):
    """Update daily OI snapshots"""
    if not current_oi_data:
        return 0
    
    conn = sqlite3.connect('oi_tracking.db')
    cursor = conn.cursor()
    
    today = date.today()
    updated_count = 0
    
    for option_chain, data in current_oi_data.items():
        try:
            # Parse option chain to get components
            # This should match your existing parse_option_chain function
            ticker = option_chain[:4] if len(option_chain) > 4 else option_chain
            
            cursor.execute('''
                INSERT OR REPLACE INTO oi_snapshots (
                    ticker, option_chain, strike, expiry, option_type,
                    snapshot_date, open_interest, volume, price
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker,
                option_chain,
                0,  # Would parse actual strike
                today,  # Would parse actual expiry
                'C',  # Would parse actual type
                today,
                data['open_interest'],
                data['volume'],
                data['price']
            ))
            updated_count += 1
        except Exception as e:
            continue
    
    conn.commit()
    conn.close()
    return updated_count

def calculate_tracking_results():
    """Calculate and update tracking results for all monitored trades"""
    conn = sqlite3.connect('oi_tracking.db')
    cursor = conn.cursor()
    
    # Get all tracked trades that haven't expired
    cursor.execute('''
        SELECT t.*, s.open_interest as current_oi
        FROM tracked_trades t
        LEFT JOIN oi_snapshots s ON t.option_chain = s.option_chain
        WHERE t.expiry >= DATE('now')
        AND s.snapshot_date = DATE('now')
    ''')
    
    tracked_trades = cursor.fetchall()
    results = []
    
    for trade in tracked_trades:
        trade_id = trade[0]
        initial_oi = trade[12]  # initial_oi column
        current_oi = trade[-1]  # current_oi from join
        predicted_change = trade[13]  # predicted_oi_change column
        
        if current_oi is None:
            continue
            
        oi_change = current_oi - initial_oi
        oi_change_pct = (oi_change / max(initial_oi, 1)) * 100
        
        # Determine prediction accuracy
        if "Major Increase" in predicted_change and oi_change_pct > 50:
            accuracy = "Correct - Major Increase"
        elif "Moderate Increase" in predicted_change and 10 <= oi_change_pct <= 50:
            accuracy = "Correct - Moderate Increase"
        elif "Small Increase" in predicted_change and 0 < oi_change_pct < 10:
            accuracy = "Correct - Small Increase"
        elif "Decrease" in predicted_change and oi_change_pct < 0:
            accuracy = "Correct - Decrease"
        elif "Minimal Change" in predicted_change and abs(oi_change_pct) < 5:
            accuracy = "Correct - Minimal Change"
        else:
            accuracy = "Incorrect Prediction"
        
        # Calculate days since trade
        trade_date = datetime.strptime(trade[6], '%Y-%m-%d').date()
        days_tracked = (date.today() - trade_date).days
        
        # Update or insert tracking result
        cursor.execute('''
            INSERT OR REPLACE INTO tracking_results (
                tracked_trade_id, days_tracked, initial_oi, current_oi,
                oi_change, oi_change_pct, prediction_accuracy
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_id, days_tracked, initial_oi, current_oi,
            oi_change, oi_change_pct, accuracy
        ))
        
        results.append({
            'trade_id': trade_id,
            'ticker': trade[1],
            'option_chain': trade[2],
            'enhanced_side': trade[7],
            'side_confidence': trade[8],
            'predicted_change': predicted_change,
            'initial_oi': initial_oi,
            'current_oi': current_oi,
            'oi_change': oi_change,
            'oi_change_pct': oi_change_pct,
            'accuracy': accuracy,
            'days_tracked': days_tracked
        })
    
    conn.commit()
    conn.close()
    return results

def get_tracking_performance_metrics():
    """Get overall performance metrics for our OI predictions"""
    conn = sqlite3.connect('oi_tracking.db')
    cursor = conn.cursor()
    
    # Overall accuracy stats
    cursor.execute('''
        SELECT 
            prediction_accuracy,
            COUNT(*) as count,
            AVG(oi_change_pct) as avg_oi_change_pct
        FROM tracking_results
        GROUP BY prediction_accuracy
    ''')
    
    accuracy_stats = cursor.fetchall()
    
    # Performance by confidence level
    cursor.execute('''
        SELECT 
            CASE 
                WHEN t.side_confidence >= 0.9 THEN 'Very High (90%+)'
                WHEN t.side_confidence >= 0.8 THEN 'High (80-89%)'
                WHEN t.side_confidence >= 0.7 THEN 'Medium-High (70-79%)'
                ELSE 'Other'
            END as confidence_level,
            COUNT(*) as total_trades,
            SUM(CASE WHEN r.prediction_accuracy LIKE 'Correct%' THEN 1 ELSE 0 END) as correct_predictions,
            AVG(ABS(r.oi_change_pct)) as avg_abs_oi_change
        FROM tracked_trades t
        JOIN tracking_results r ON t.id = r.tracked_trade_id
        GROUP BY confidence_level
    ''')
    
    confidence_performance = cursor.fetchall()
    
    # Performance by trade side
    cursor.execute('''
        SELECT 
            CASE 
                WHEN t.enhanced_side LIKE '%BUY%' THEN 'BUY'
                WHEN t.enhanced_side LIKE '%SELL%' THEN 'SELL'
                ELSE 'OTHER'
            END as trade_side,
            COUNT(*) as total_trades,
            SUM(CASE WHEN r.prediction_accuracy LIKE 'Correct%' THEN 1 ELSE 0 END) as correct_predictions,
            AVG(r.oi_change_pct) as avg_oi_change_pct
        FROM tracked_trades t
        JOIN tracking_results r ON t.id = r.tracked_trade_id
        GROUP BY trade_side
    ''')
    
    side_performance = cursor.fetchall()
    
    conn.close()
    
    return {
        'accuracy_stats': accuracy_stats,
        'confidence_performance': confidence_performance,
        'side_performance': side_performance
    }

def display_oi_tracking_dashboard():
    """Display the OI tracking dashboard"""
    st.markdown("### 📊 Open Interest Tracking Dashboard")
    
    # Initialize database if needed
    init_oi_tracking_db()
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Update OI Data", use_container_width=True):
            with st.spinner("Fetching current OI data..."):
                # Get tracked option chains
                conn = sqlite3.connect('oi_tracking.db')
                cursor = conn.cursor()
                cursor.execute('SELECT DISTINCT option_chain FROM tracked_trades WHERE expiry >= DATE("now")')
                option_chains = [row[0] for row in cursor.fetchall()]
                conn.close()
                
                if option_chains:
                    current_oi_data = fetch_current_oi_data(option_chains)
                    updated_count = update_oi_snapshots(current_oi_data)
                    st.success(f"Updated OI data for {updated_count} options")
                else:
                    st.info("No tracked trades found")
    
    with col2:
        if st.button("🔄 Refresh Results", use_container_width=True):
            with st.spinner("Calculating tracking results..."):
                results = calculate_tracking_results()
                st.success(f"Updated results for {len(results)} tracked trades")
    
    with col3:
        if st.button("🧹 Clear Old Data", use_container_width=True):
            conn = sqlite3.connect('oi_tracking.db')
            cursor = conn.cursor()
            # Remove expired trades older than 30 days
            cursor.execute('DELETE FROM tracked_trades WHERE expiry < DATE("now", "-30 days")')
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            st.success(f"Cleaned up {deleted} expired trades")
    
    # Get current tracking statistics
    conn = sqlite3.connect('oi_tracking.db')
    
    # Summary metrics
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM tracked_trades WHERE expiry >= DATE("now")')
    active_trades = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(*) FROM tracking_results')
    tracked_results = cursor.fetchone()[0]
    
    cursor.execute('''
        SELECT COUNT(*) FROM tracking_results 
        WHERE prediction_accuracy LIKE "Correct%"
    ''')
    correct_predictions = cursor.fetchone()[0]
    
    accuracy_rate = (correct_predictions / max(tracked_results, 1)) * 100
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Tracks", active_trades)
    with col2:
        st.metric("Total Results", tracked_results)
    with col3:
        st.metric("Correct Predictions", correct_predictions)
    with col4:
        st.metric("Accuracy Rate", f"{accuracy_rate:.1f}%")
    
    # Performance metrics
    if tracked_results > 0:
        metrics = get_tracking_performance_metrics()
        
        st.markdown("#### 🎯 Prediction Performance Analysis")
        
        # Accuracy breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Prediction Accuracy Breakdown:**")
            accuracy_df = pd.DataFrame(metrics['accuracy_stats'], 
                                     columns=['Prediction Result', 'Count', 'Avg OI Change %'])
            st.dataframe(accuracy_df, use_container_width=True)
        
        with col2:
            st.markdown("**Performance by Confidence Level:**")
            conf_df = pd.DataFrame(metrics['confidence_performance'], 
                                 columns=['Confidence Level', 'Total Trades', 'Correct', 'Avg OI Change %'])
            conf_df['Accuracy %'] = (conf_df['Correct'] / conf_df['Total Trades'] * 100).round(1)
            st.dataframe(conf_df, use_container_width=True)
        
        # Performance by trade side
        st.markdown("**Performance by Trade Side:**")
        side_df = pd.DataFrame(metrics['side_performance'], 
                             columns=['Trade Side', 'Total Trades', 'Correct', 'Avg OI Change %'])
        side_df['Accuracy %'] = (side_df['Correct'] / side_df['Total Trades'] * 100).round(1)
        st.dataframe(side_df, use_container_width=True)
    
    # Recent tracking results
    st.markdown("#### 📈 Recent Tracking Results")
    
    cursor.execute('''
        SELECT 
            t.ticker,
            t.option_chain,
            t.enhanced_side,
            t.side_confidence,
            t.predicted_oi_change,
            r.initial_oi,
            r.current_oi,
            r.oi_change,
            r.oi_change_pct,
            r.prediction_accuracy,
            r.days_tracked
        FROM tracked_trades t
        JOIN tracking_results r ON t.id = r.tracked_trade_id
        ORDER BY r.last_updated DESC
        LIMIT 20
    ''')
    
    recent_results = cursor.fetchall()
    
    if recent_results:
        results_data = []
        for result in recent_results:
            # Color coding for accuracy
            accuracy = result[9]
            if "Correct" in accuracy:
                accuracy_emoji = "✅"
            else:
                accuracy_emoji = "❌"
            
            # Side emoji
            if "BUY" in result[2]:
                side_emoji = "🟢"
            elif "SELL" in result[2]:
                side_emoji = "🔴"
            else:
                side_emoji = "⚪"
            
            results_data.append({
                'Ticker': result[0],
                'Option': result[1][:20] + "..." if len(result[1]) > 20 else result[1],
                'Side': f"{side_emoji} {result[2]}",
                'Confidence': f"{result[3]:.0%}",
                'Predicted': result[4],
                'Initial OI': f"{result[5]:,}",
                'Current OI': f"{result[6]:,}",
                'OI Change': f"{result[7]:+,}",
                'Change %': f"{result[8]:+.1f}%",
                'Accuracy': f"{accuracy_emoji} {accuracy}",
                'Days': result[10]
            })
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
    else:
        st.info("No tracking results available. Start by saving some high-confidence trades!")
    
    # Visualization of OI changes
    if recent_results:
        st.markdown("#### 📊 OI Change Distribution")
        
        # Create histogram of OI changes
        oi_changes = [result[8] for result in recent_results]
        
        fig = px.histogram(
            x=oi_changes,
            nbins=20,
            title="Distribution of OI Changes (%)",
            labels={'x': 'OI Change %', 'y': 'Count'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Accuracy by confidence level chart
        if len(recent_results) > 5:
            confidence_data = []
            accuracy_data = []
            
            for result in recent_results:
                confidence_data.append(result[3] * 100)  # Convert to percentage
                accuracy_data.append(1 if "Correct" in result[9] else 0)
            
            fig = px.scatter(
                x=confidence_data,
                y=accuracy_data,
                title="Prediction Accuracy vs Confidence Level",
                labels={'x': 'Confidence %', 'y': 'Correct (1) vs Incorrect (0)'},
                hover_data={'x': confidence_data}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    conn.close()

def run_oi_tracking_for_trades(trades):
    """Run OI tracking for a batch of trades and return tracking info"""
    if not trades:
        return {"saved": 0, "message": "No trades to track"}
    
    # Initialize database
    init_oi_tracking_db()
    
    # Save high-confidence trades
    saved_count = save_high_confidence_trades(trades)
    
    # Get summary of what was saved
    if saved_count > 0:
        message = f"🎯 Started tracking {saved_count} high-confidence trades for future OI analysis"
        
        # Show breakdown of saved trades
        high_conf_trades = [t for t in trades if t.get('side_confidence', 0) >= 0.7]
        buy_count = len([t for t in high_conf_trades if 'BUY' in t.get('enhanced_side', '')])
        sell_count = len([t for t in high_conf_trades if 'SELL' in t.get('enhanced_side', '')])
        
        breakdown = f"📊 Breakdown: {buy_count} BUY trades, {sell_count} SELL trades"
        
        return {
            "saved": saved_count,
            "message": message,
            "breakdown": breakdown,
            "high_conf_trades": high_conf_trades
        }
    else:
        return {
            "saved": 0,
            "message": "⚠️ No high-confidence trades found to track (need 70%+ confidence)",
            "breakdown": f"Total trades analyzed: {len(trades)}"
        }

def display_tracking_setup_info():
    """Display information about the OI tracking system"""
    st.markdown("""
    ### 🎯 Open Interest Tracking System
    
    **How it works:**
    1. **High-Confidence Detection**: Only tracks trades with 70%+ confidence in buy/sell direction
    2. **OI Prediction**: Makes predictions about OI changes based on volume/OI ratios and trade direction
    3. **Daily Monitoring**: Tracks actual OI changes over time to validate predictions
    4. **Performance Analysis**: Measures accuracy of our buy/sell detection system
    
    **Prediction Categories:**
    - **Major Increase**: Vol/OI > 5x + BUY signal (expect 50%+ OI increase)
    - **Moderate Increase**: Vol/OI 2-5x + BUY signal (expect 10-50% OI increase)
    - **Small Increase**: BUY signal + volume >100 (expect <10% OI increase)
    - **Decrease**: SELL signal + low Vol/OI (expect OI decrease)
    - **Minimal Change**: Mixed signals (expect <5% change)
    
    **Benefits:**
    - **Validate Detection**: See if our buy/sell classification actually predicts OI changes
    - **Improve Accuracy**: Learn which signals are most reliable
    - **Market Intelligence**: Understand which trades create lasting positions
    - **Performance Tracking**: Measure success rate of high-confidence predictions
    """)
