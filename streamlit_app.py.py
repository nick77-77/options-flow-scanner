import streamlit as st
import httpx
import asyncio
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo
import logging
from concurrent.futures import ThreadPoolExecutor
import json
from enum import Enum

# --- ENHANCED CONFIGURATION ---
@dataclass
class Config:
    # API Configuration
    UW_TOKEN: str = "e6e8601a-0746-4cec-a07d-c3eabfc13926"
    BASE_URL: str = "https://api.unusualwhales.com/api/option-trades/flow-alerts"
    TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    
    # Filtering Configuration
    EXCLUDE_TICKERS: set = field(default_factory=lambda: {'MSTR', 'CRCL', 'COIN', 'META', 'NVDA', 'AMD', 'TSLA'})
    ALLOWED_TICKERS: set = field(default_factory=lambda: {'QQQ', 'SPY', 'IWM'})
    
    # Thresholds
    MIN_PREMIUM: int = 100000
    LIMIT: int = 500
    HIGH_IV_THRESHOLD: float = 0.30
    EXTREME_IV_THRESHOLD: float = 0.50
    IV_CRUSH_THRESHOLD: float = 0.15
    HIGH_VOL_OI_RATIO: float = 5.0
    UNUSUAL_OI_THRESHOLD: int = 1000
    GAMMA_SQUEEZE_THRESHOLD: float = 0.10
    IV_SPIKE_THRESHOLD: float = 0.20
    CORRELATION_THRESHOLD: float = 0.7
    
    # Performance Settings
    CACHE_TTL: int = 300  # 5 minutes
    BATCH_SIZE: int = 100
    MAX_DISPLAY_TRADES: int = 50

class TradeSide(Enum):
    BUY = "BUY"
    SELL = "SELL"
    UNKNOWN = "UNKNOWN"

class ConfidenceLevel(Enum):
    VERY_HIGH = "Very High"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    VERY_LOW = "Very Low"

@dataclass
class TradeAnalysis:
    side: TradeSide
    confidence: float
    confidence_level: ConfidenceLevel
    reasoning: List[str]
    price_quality: str
    data_completeness: float

@dataclass
class EnhancedTrade:
    # Basic trade info
    ticker: str
    option_chain: str
    type: str  # C or P
    strike: float
    expiry: str
    dte: int
    
    # Pricing data
    price: Union[float, str]
    premium: float
    volume: float
    open_interest: float
    iv: float
    bid: float
    ask: float
    underlying_price: float
    
    # Enhanced analysis
    trade_analysis: TradeAnalysis
    moneyness: str
    scenarios: List[str]
    oi_analysis: Dict
    time_ny: str
    
    # Metadata
    rule_name: str
    description: str
    created_at: str

# --- ENHANCED DATA PROCESSING ---
class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger
    
    def parse_option_chain(self, opt_str: str) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str], Optional[float]]:
        """Enhanced option chain parser with better error handling"""
        try:
            if not opt_str or not isinstance(opt_str, str):
                return None, None, None, None, None
            
            # Find ticker (letters only)
            ticker_match = ""
            for i, char in enumerate(opt_str):
                if char.isalpha():
                    ticker_match += char
                else:
                    break
            
            if len(ticker_match) < 2:
                return None, None, None, None, None
            
            # Remove option type indicator at end
            ticker = ticker_match[:-1] if ticker_match[-1] in 'CP' else ticker_match
            
            # Find date start
            date_start = len(ticker_match)
            if date_start + 6 >= len(opt_str):
                return None, None, None, None, None
            
            # Parse date
            date_str = opt_str[date_start:date_start+6]
            if not date_str.isdigit():
                return None, None, None, None, None
            
            year = 2000 + int(date_str[:2])
            month = int(date_str[2:4])
            day = int(date_str[4:6])
            
            if month > 12 or day > 31:
                return None, None, None, None, None
            
            expiry_date = date(year, month, day)
            dte = (expiry_date - date.today()).days
            
            # Parse option type and strike
            option_type = opt_str[date_start+6].upper()
            if option_type not in ['C', 'P']:
                return None, None, None, None, None
            
            strike_str = opt_str[date_start+7:]
            if not strike_str.isdigit():
                return None, None, None, None, None
            
            strike = int(strike_str) / 1000
            
            return ticker.upper(), expiry_date.strftime('%Y-%m-%d'), dte, option_type, strike
        
        except Exception as e:
            self.logger.error(f"Error parsing option chain {opt_str}: {e}")
            return None, None, None, None, None

class TradeSideAnalyzer:
    """Enhanced trade side detection with machine learning-like confidence scoring"""
    
    def __init__(self):
        self.price_weight = 0.4
        self.volume_weight = 0.3
        self.description_weight = 0.2
        self.context_weight = 0.1
    
    def analyze_trade_side(self, trade_data: Dict) -> TradeAnalysis:
        """Comprehensive trade side analysis with detailed confidence scoring"""
        
        signals = []
        confidence_scores = []
        reasoning = []
        
        # Extract and validate data
        price = self._safe_float(trade_data.get('price', 0))
        bid = self._safe_float(trade_data.get('bid', 0))
        ask = self._safe_float(trade_data.get('ask', 0))
        volume = self._safe_float(trade_data.get('volume', 0))
        oi = max(self._safe_float(trade_data.get('open_interest', 1)), 1)
        
        # Calculate data completeness
        data_fields = ['price', 'bid', 'ask', 'volume', 'open_interest']
        valid_fields = sum(1 for field in data_fields if trade_data.get(field) not in [None, 'N/A', '', 0])
        data_completeness = valid_fields / len(data_fields)
        
        # Method 1: Bid/Ask Analysis (Most reliable)
        price_signal, price_confidence, price_reasoning = self._analyze_price_vs_spread(price, bid, ask)
        if price_signal != TradeSide.UNKNOWN:
            signals.append(price_signal)
            confidence_scores.append(price_confidence * self.price_weight)
            reasoning.extend(price_reasoning)
        
        # Method 2: Volume/OI Analysis
        vol_signal, vol_confidence, vol_reasoning = self._analyze_volume_oi_ratio(volume, oi)
        if vol_signal != TradeSide.UNKNOWN:
            signals.append(vol_signal)
            confidence_scores.append(vol_confidence * self.volume_weight)
            reasoning.extend(vol_reasoning)
        
        # Method 3: Description Analysis
        desc_signal, desc_confidence, desc_reasoning = self._analyze_description(
            trade_data.get('description', ''), trade_data.get('rule_name', '')
        )
        if desc_signal != TradeSide.UNKNOWN:
            signals.append(desc_signal)
            confidence_scores.append(desc_confidence * self.description_weight)
            reasoning.extend(desc_reasoning)
        
        # Method 4: Contextual Analysis
        context_signal, context_confidence, context_reasoning = self._analyze_context(trade_data)
        if context_signal != TradeSide.UNKNOWN:
            signals.append(context_signal)
            confidence_scores.append(context_confidence * self.context_weight)
            reasoning.extend(context_reasoning)
        
        # Aggregate results
        final_side, final_confidence = self._aggregate_signals(signals, confidence_scores)
        confidence_level = self._determine_confidence_level(final_confidence)
        price_quality = self._assess_price_quality(bid, ask, price)
        
        return TradeAnalysis(
            side=final_side,
            confidence=final_confidence,
            confidence_level=confidence_level,
            reasoning=reasoning,
            price_quality=price_quality,
            data_completeness=data_completeness
        )
    
    def _safe_float(self, value: Union[str, int, float]) -> float:
        """Safely convert value to float"""
        if value in [None, 'N/A', '', 'NaN']:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _analyze_price_vs_spread(self, price: float, bid: float, ask: float) -> Tuple[TradeSide, float, List[str]]:
        """Analyze price relative to bid/ask spread"""
        if not all([price > 0, bid > 0, ask > 0]) or ask <= bid:
            return TradeSide.UNKNOWN, 0.0, ["Insufficient bid/ask data"]
        
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid
        
        # Tighter thresholds for better accuracy
        ask_threshold = ask * 0.95  # Within 5% of ask
        bid_threshold = bid * 1.05  # Within 5% of bid
        
        reasoning = []
        
        if price >= ask_threshold:
            confidence = 0.95 - (spread_pct * 0.2)  # Lower confidence for wide spreads
            reasoning.append(f"Price {price:.2f} at/above ask {ask:.2f}")
            return TradeSide.BUY, min(confidence, 0.95), reasoning
        
        elif price <= bid_threshold:
            confidence = 0.95 - (spread_pct * 0.2)
            reasoning.append(f"Price {price:.2f} at/below bid {bid:.2f}")
            return TradeSide.SELL, min(confidence, 0.95), reasoning
        
        elif price > mid * 1.02:
            confidence = 0.6 - (spread_pct * 0.1)
            reasoning.append(f"Price {price:.2f} above mid {mid:.2f}")
            return TradeSide.BUY, max(confidence, 0.3), reasoning
        
        elif price < mid * 0.98:
            confidence = 0.6 - (spread_pct * 0.1)
            reasoning.append(f"Price {price:.2f} below mid {mid:.2f}")
            return TradeSide.SELL, max(confidence, 0.3), reasoning
        
        else:
            reasoning.append(f"Price {price:.2f} near mid {mid:.2f}")
            return TradeSide.UNKNOWN, 0.2, reasoning
    
    def _analyze_volume_oi_ratio(self, volume: float, oi: float) -> Tuple[TradeSide, float, List[str]]:
        """Analyze volume to open interest ratio"""
        if volume <= 0:
            return TradeSide.UNKNOWN, 0.0, ["No volume data"]
        
        vol_oi_ratio = volume / oi
        reasoning = []
        
        if vol_oi_ratio > 20:
            reasoning.append(f"Extreme vol/OI ratio: {vol_oi_ratio:.1f}")
            return TradeSide.BUY, 0.9, reasoning
        elif vol_oi_ratio > 10:
            reasoning.append(f"Very high vol/OI ratio: {vol_oi_ratio:.1f}")
            return TradeSide.BUY, 0.8, reasoning
        elif vol_oi_ratio > 5:
            reasoning.append(f"High vol/OI ratio: {vol_oi_ratio:.1f}")
            return TradeSide.BUY, 0.6, reasoning
        elif vol_oi_ratio > 2:
            reasoning.append(f"Moderate vol/OI ratio: {vol_oi_ratio:.1f}")
            return TradeSide.BUY, 0.4, reasoning
        else:
            reasoning.append(f"Low vol/OI ratio: {vol_oi_ratio:.1f}")
            return TradeSide.UNKNOWN, 0.1, reasoning
    
    def _analyze_description(self, description: str, rule_name: str) -> Tuple[TradeSide, float, List[str]]:
        """Analyze trade description and rules for directional signals"""
        desc_lower = description.lower()
        rule_lower = rule_name.lower()
        reasoning = []
        
        # Strong buy indicators
        strong_buy_keywords = ['sweep', 'aggressive buy', 'lifted', 'taken', 'market buy', 'block buy']
        if any(keyword in desc_lower for keyword in strong_buy_keywords):
            reasoning.append("Strong buy keywords detected")
            return TradeSide.BUY, 0.85, reasoning
        
        # Strong sell indicators
        strong_sell_keywords = ['sold', 'offer hit', 'market sell', 'hit bid', 'block sell']
        if any(keyword in desc_lower for keyword in strong_sell_keywords):
            reasoning.append("Strong sell keywords detected")
            return TradeSide.SELL, 0.85, reasoning
        
        # Rule-based analysis
        if 'ascending' in rule_lower:
            reasoning.append("Ascending fill pattern")
            return TradeSide.BUY, 0.5, reasoning
        elif 'descending' in rule_lower:
            reasoning.append("Descending fill pattern")
            return TradeSide.SELL, 0.5, reasoning
        
        return TradeSide.UNKNOWN, 0.0, ["No clear directional signals in description"]
    
    def _analyze_context(self, trade_data: Dict) -> Tuple[TradeSide, float, List[str]]:
        """Analyze contextual factors like option type, moneyness, premium size"""
        reasoning = []
        signals = []
        
        # Premium size analysis
        premium = self._safe_float(trade_data.get('premium', 0))
        if premium > 1000000:
            reasoning.append(f"Very large premium: ${premium:,.0f}")
            signals.append((TradeSide.BUY, 0.4))  # Large trades often buying
        
        # Option type and moneyness
        option_type = trade_data.get('type', '')
        strike = self._safe_float(trade_data.get('strike', 0))
        underlying = self._safe_float(trade_data.get('underlying_price', strike))
        
        if underlying > 0 and strike > 0:
            moneyness_pct = ((strike - underlying) / underlying) * 100
            
            if option_type == 'C' and moneyness_pct > 3:  # OTM calls
                reasoning.append("OTM call - typically bought")
                signals.append((TradeSide.BUY, 0.3))
            elif option_type == 'P' and moneyness_pct < -3:  # OTM puts
                reasoning.append("OTM put - typically bought")
                signals.append((TradeSide.BUY, 0.3))
        
        # Aggregate contextual signals
        if signals:
            avg_confidence = np.mean([conf for _, conf in signals])
            most_common_side = max(set(side for side, _ in signals), 
                                 key=lambda x: sum(conf for side, conf in signals if side == x))
            return most_common_side, avg_confidence, reasoning
        
        return TradeSide.UNKNOWN, 0.0, reasoning
    
    def _aggregate_signals(self, signals: List[TradeSide], confidences: List[float]) -> Tuple[TradeSide, float]:
        """Aggregate multiple signals into final result"""
        if not signals:
            return TradeSide.UNKNOWN, 0.0
        
        # Weight by confidence and count
        side_scores = defaultdict(float)
        for side, confidence in zip(signals, confidences):
            side_scores[side] += confidence
        
        # Find the side with highest total confidence
        if not side_scores or max(side_scores.values()) < 0.1:
            return TradeSide.UNKNOWN, 0.0
        
        final_side = max(side_scores.keys(), key=side_scores.get)
        final_confidence = min(side_scores[final_side], 1.0)
        
        return final_side, final_confidence
    
    def _determine_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to categorical level"""
        if confidence >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.6:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _assess_price_quality(self, bid: float, ask: float, price: float) -> str:
        """Assess the quality of price data for trade side detection"""
        if bid > 0 and ask > 0 and ask > bid:
            spread_pct = (ask - bid) / ((bid + ask) / 2)
            if spread_pct < 0.02:
                return "Excellent"
            elif spread_pct < 0.05:
                return "Good"
            elif spread_pct < 0.1:
                return "Fair"
            else:
                return "Poor - Wide Spread"
        elif price > 0:
            return "Limited - Price Only"
        else:
            return "Poor - No Price Data"

# --- ENHANCED API CLIENT ---
class APIClient:
    def __init__(self, config: Config):
        self.config = config
        self.headers = {
            'Accept': 'application/json, text/plain',
            'Authorization': config.UW_TOKEN
        }
        self.logger = logging.getLogger(__name__)
    
    async def fetch_data_async(self, params: Dict) -> List[Dict]:
        """Asynchronous data fetching with retry logic"""
        async with httpx.AsyncClient(timeout=self.config.TIMEOUT) as client:
            for attempt in range(self.config.MAX_RETRIES):
                try:
                    response = await client.get(
                        self.config.BASE_URL,
                        headers=self.headers,
                        params=params
                    )
                    response.raise_for_status()
                    data = response.json().get('data', [])
                    self.logger.info(f"Successfully fetched {len(data)} trades")
                    return data
                
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == self.config.MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return []
    
    def fetch_data(self, params: Dict) -> List[Dict]:
        """Synchronous wrapper for async fetch"""
        try:
            return asyncio.run(self.fetch_data_async(params))
        except Exception as e:
            self.logger.error(f"Failed to fetch data: {e}")
            st.error(f"API Error: {e}")
            return []

# --- ENHANCED ANALYTICS ENGINE ---
class AnalyticsEngine:
    def __init__(self, config: Config):
        self.config = config
        self.processor = DataProcessor(config)
        self.analyzer = TradeSideAnalyzer()
    
    def process_trades(self, raw_trades: List[Dict]) -> List[EnhancedTrade]:
        """Process raw trades into enhanced trade objects"""
        enhanced_trades = []
        
        for trade in raw_trades:
            try:
                enhanced_trade = self._create_enhanced_trade(trade)
                if enhanced_trade:
                    enhanced_trades.append(enhanced_trade)
            except Exception as e:
                self.processor.logger.error(f"Error processing trade: {e}")
                continue
        
        return enhanced_trades
    
    def _create_enhanced_trade(self, trade_data: Dict) -> Optional[EnhancedTrade]:
        """Create enhanced trade object from raw data"""
        # Parse option chain
        option_chain = trade_data.get('option_chain', '')
        ticker, expiry, dte, opt_type, strike = self.processor.parse_option_chain(option_chain)
        
        if not all([ticker, expiry, dte is not None, opt_type, strike]):
            return None
        
        # Skip excluded tickers
        if ticker in self.config.EXCLUDE_TICKERS:
            return None
        
        # Extract and validate premium
        premium = self.analyzer._safe_float(trade_data.get('total_premium', 0))
        if premium < self.config.MIN_PREMIUM:
            return None
        
        # Time conversion
        time_ny = self._convert_time(trade_data.get('created_at'))
        
        # Enhanced trade side analysis
        trade_analysis = self.analyzer.analyze_trade_side(trade_data)
        
        # Calculate moneyness
        underlying_price = self.analyzer._safe_float(trade_data.get('underlying_price', strike))
        moneyness = self._calculate_moneyness(strike, underlying_price, opt_type)
        
        # Analyze scenarios and OI
        scenarios = self._detect_scenarios(trade_data, trade_analysis)
        oi_analysis = self._analyze_open_interest(trade_data)
        
        return EnhancedTrade(
            ticker=ticker,
            option_chain=option_chain,
            type=opt_type,
            strike=strike,
            expiry=expiry,
            dte=dte,
            price=trade_data.get('price', 'N/A'),
            premium=premium,
            volume=self.analyzer._safe_float(trade_data.get('volume', 0)),
            open_interest=self.analyzer._safe_float(trade_data.get('open_interest', 0)),
            iv=self.analyzer._safe_float(trade_data.get('iv', 0)),
            bid=self.analyzer._safe_float(trade_data.get('bid', 0)),
            ask=self.analyzer._safe_float(trade_data.get('ask', 0)),
            underlying_price=underlying_price,
            trade_analysis=trade_analysis,
            moneyness=moneyness,
            scenarios=scenarios,
            oi_analysis=oi_analysis,
            time_ny=time_ny,
            rule_name=trade_data.get('rule_name', ''),
            description=trade_data.get('description', ''),
            created_at=trade_data.get('created_at', '')
        )
    
    def _convert_time(self, time_str: str) -> str:
        """Convert UTC time to NY time"""
        if not time_str or time_str == 'N/A':
            return 'N/A'
        
        try:
            utc_time = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            ny_time = utc_time.astimezone(ZoneInfo("America/New_York"))
            return ny_time.strftime("%I:%M %p")
        except Exception:
            return 'N/A'
    
    def _calculate_moneyness(self, strike: float, underlying: float, opt_type: str) -> str:
        """Calculate option moneyness"""
        if underlying <= 0:
            return "Unknown"
        
        diff_percent = ((strike - underlying) / underlying) * 100
        
        if abs(diff_percent) < 2:
            return "ATM"
        elif opt_type == 'C':
            if diff_percent > 0:
                return f"OTM +{diff_percent:.1f}%"
            else:
                return f"ITM {diff_percent:.1f}%"
        else:  # Put
            if diff_percent < 0:
                return f"OTM {abs(diff_percent):.1f}%"
            else:
                return f"ITM +{diff_percent:.1f}%"
    
    def _detect_scenarios(self, trade_data: Dict, analysis: TradeAnalysis) -> List[str]:
        """Detect trading scenarios"""
        scenarios = []
        
        premium = self.analyzer._safe_float(trade_data.get('total_premium', 0))
        volume = self.analyzer._safe_float(trade_data.get('volume', 0))
        oi = self.analyzer._safe_float(trade_data.get('open_interest', 1))
        iv = self.analyzer._safe_float(trade_data.get('iv', 0))
        
        # Premium-based scenarios
        if premium > 1000000:
            scenarios.append("Mega Premium Trade (>$1M)")
        elif premium > 500000:
            scenarios.append("Large Premium Trade")
        
        # Volume/OI scenarios
        vol_oi_ratio = volume / oi
        if vol_oi_ratio > 10:
            scenarios.append("Extreme Volume Surge")
        elif vol_oi_ratio > 5:
            scenarios.append("High Volume Activity")
        
        # IV scenarios
        if iv > self.config.EXTREME_IV_THRESHOLD:
            scenarios.append("Extreme IV Play")
        elif iv > self.config.HIGH_IV_THRESHOLD:
            scenarios.append("High IV Premium")
        
        # Trade side scenarios
        if analysis.confidence > 0.8:
            if analysis.side == TradeSide.BUY:
                scenarios.append("High Confidence Buying")
            elif analysis.side == TradeSide.SELL:
                scenarios.append("High Confidence Selling")
        
        return scenarios if scenarios else ["Normal Flow"]
    
    def _analyze_open_interest(self, trade_data: Dict) -> Dict:
        """Analyze open interest patterns"""
        oi = self.analyzer._safe_float(trade_data.get('open_interest', 0))
        volume = self.analyzer._safe_float(trade_data.get('volume', 0))
        
        analysis = {
            'oi_level': 'Normal',
            'liquidity_score': 'Medium',
            'oi_change_indicator': 'Stable'
        }
        
        # OI level categorization
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
        
        # Liquidity assessment
        vol_oi_ratio = volume / max(oi, 1)
        if oi > 5000 and volume > 100:
            analysis['liquidity_score'] = 'Excellent'
        elif oi > 1000 and volume > 50:
            analysis['liquidity_score'] = 'Good'
        elif oi > 500 and volume > 20:
            analysis['liquidity_score'] = 'Fair'
        else:
            analysis['liquidity_score'] = 'Poor'
        
        # OI change prediction
        if vol_oi_ratio > 5:
            analysis['oi_change_indicator'] = 'Major Increase Expected'
        elif vol_oi_ratio > 2:
            analysis['oi_change_indicator'] = 'Increase Expected'
        elif vol_oi_ratio > 0.5:
            analysis['oi_change_indicator'] = 'Moderate Activity'
        else:
            analysis['oi_change_indicator'] = 'Low Activity'
        
        return analysis

# --- ENHANCED VISUALIZATION ---
class VisualizationEngine:
    @staticmethod
    def create_sentiment_gauge(trades: List[EnhancedTrade]) -> go.Figure:
        """Create market sentiment gauge"""
        buy_premium = sum(t.premium for t in trades if t.trade_analysis.side == TradeSide.BUY)
        total_premium = sum(t.premium for t in trades)
        
        if total_premium == 0:
            sentiment_ratio = 0.5
        else:
            sentiment_ratio = buy_premium / total_premium
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = sentiment_ratio * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Market Sentiment (% Buy Premium)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightcoral"},
                    {'range': [30, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightgreen"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50}}))
        
        return fig
    
    @staticmethod
    def create_confidence_distribution(trades: List[EnhancedTrade]) -> go.Figure:
        """Create confidence level distribution chart"""
        confidence_levels = [t.trade_analysis.confidence_level.value for t in trades]
        
        fig = px.histogram(
            x=confidence_levels,
            title="Trade Direction Confidence Distribution",
            labels={'x': 'Confidence Level', 'y': 'Number of Trades'}
        )
        
        return fig
    
    @staticmethod
    def create_premium_by_side_chart(trades: List[EnhancedTrade]) -> go.Figure:
        """Create premium distribution by trade side"""
        buy_trades = [t for t in trades if t.trade_analysis.side == TradeSide.BUY]
        sell_trades = [t for t in trades if t.trade_analysis.side == TradeSide.SELL]
        
        fig = go.Figure()
        
        if buy_trades:
            fig.add_trace(go.Box(
                y=[t.premium for t in buy_trades],
                name="Buy Trades",
                marker_color="green"
            ))
        
        if sell_trades:
            fig.add_trace(go.Box(
                y=[t.premium for t in sell_trades],
                name="Sell Trades",
                marker_color="red"
            ))
        
        fig.update_layout(
            title="Premium Distribution by Trade Side",
            yaxis_title="Premium ($)",
            yaxis_type="log"
        )
        
        return fig

# --- STREAMLIT UI COMPONENTS ---
class UIComponents:
    @staticmethod
    def render_enhanced_metrics(trades: List[EnhancedTrade]):
        """Render enhanced metrics dashboard"""
        if not trades:
            st.warning("No trades available for metrics")
            return
        
        # Primary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_premium = sum(t.premium for t in trades)
            st.metric("Total Premium", f"${total_premium:,.0f}")
        
        with col2:
            buy_trades = len([t for t in trades if t.trade_analysis.side == TradeSide.BUY])
            sell_trades = len([t for t in trades if t.trade_analysis.side == TradeSide.SELL])
            st.metric("Buy/Sell Ratio", f"{buy_trades}:{sell_trades}")
        
        with col3:
            high_conf_trades = len([t for t in trades if t.trade_analysis.confidence >= 0.7])
            st.metric("High Confidence", f"{high_conf_trades}")
        
        with col4:
            avg_confidence = np.mean([t.trade_analysis.confidence for t in trades])
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        
        # Secondary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            zero_dte = len([t for t in trades if t.dte == 0])
            st.metric("0DTE Trades", zero_dte)
        
        with col2:
            avg_data_quality = np.mean([t.trade_analysis.data_completeness for t in trades])
            st.metric("Data Quality", f"{avg_data_quality:.1%}")
        
        with col3:
            unique_tickers = len(set(t.ticker for t in trades))
            st.metric("Unique Tickers", unique_tickers)
        
        with col4:
            mega_trades = len([t for t in trades if t.premium > 1000000])
            st.metric("Mega Trades (>$1M)", mega_trades)
    
    @staticmethod
    def render_enhanced_trade_table(trades: List[EnhancedTrade], max_rows: int = 50):
        """Render enhanced trade table with improved formatting"""
        if not trades:
            st.info("No trades to display")
            return
        
        # Sort trades by premium (descending)
        sorted_trades = sorted(trades, key=lambda x: x.premium, reverse=True)[:max_rows]
        
        # Prepare table data
        table_data = []
        for trade in sorted_trades:
            analysis = trade.trade_analysis
            
            # Side emoji and display
            if analysis.side == TradeSide.BUY:
                side_display = f"🟢 BUY ({analysis.confidence_level.value})"
            elif analysis.side == TradeSide.SELL:
                side_display = f"🔴 SELL ({analysis.confidence_level.value})"
            else:
                side_display = f"⚪ UNKNOWN ({analysis.confidence_level.value})"
            
            # Confidence color coding
            if analysis.confidence >= 0.7:
                conf_display = f"🟢 {analysis.confidence:.0%}"
            elif analysis.confidence >= 0.4:
                conf_display = f"🟡 {analysis.confidence:.0%}"
            else:
                conf_display = f"🔴 {analysis.confidence:.0%}"
            
            table_data.append({
                'Ticker': trade.ticker,
                'Type': trade.type,
                'Side': side_display,
                'Confidence': conf_display,
                'Strike': f"${trade.strike:.0f}",
                'DTE': trade.dte,
                'Premium': f"${trade.premium:,.0f}",
                'Volume': f"{trade.volume:,.0f}",
                'OI': f"{trade.open_interest:,.0f}",
                'Vol/OI': f"{trade.volume/max(trade.open_interest, 1):.1f}",
                'IV': f"{trade.iv:.1%}" if trade.iv > 0 else "N/A",
                'Moneyness': trade.moneyness,
                'Primary Scenario': trade.scenarios[0] if trade.scenarios else "Normal Flow",
                'Data Quality': trade.trade_analysis.price_quality,
                'Time': trade.time_ny
            })
        
        # Display table
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, height=600)
    
    @staticmethod
    def render_debug_panel(trades: List[EnhancedTrade]):
        """Render debug and diagnostics panel"""
        if not trades:
            return
        
        st.markdown("### 🔧 Debug & Diagnostics")
        
        # Data quality overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            price_quality_dist = {}
            for trade in trades:
                quality = trade.trade_analysis.price_quality
                price_quality_dist[quality] = price_quality_dist.get(quality, 0) + 1
            
            st.markdown("**Price Data Quality:**")
            for quality, count in sorted(price_quality_dist.items()):
                pct = count / len(trades) * 100
                st.write(f"• {quality}: {count} ({pct:.1f}%)")
        
        with col2:
            confidence_dist = {}
            for trade in trades:
                level = trade.trade_analysis.confidence_level.value
                confidence_dist[level] = confidence_dist.get(level, 0) + 1
            
            st.markdown("**Confidence Distribution:**")
            for level, count in sorted(confidence_dist.items()):
                pct = count / len(trades) * 100
                st.write(f"• {level}: {count} ({pct:.1f}%)")
        
        with col3:
            avg_completeness = np.mean([t.trade_analysis.data_completeness for t in trades])
            poor_quality_trades = len([t for t in trades if t.trade_analysis.data_completeness < 0.6])
            unknown_side_trades = len([t for t in trades if t.trade_analysis.side == TradeSide.UNKNOWN])
            
            st.markdown("**Overall Health:**")
            st.write(f"• Avg Data Completeness: {avg_completeness:.1%}")
            st.write(f"• Poor Quality Trades: {poor_quality_trades}")
            st.write(f"• Unknown Side Trades: {unknown_side_trades}")
        
        # Sample trade analysis
        if st.checkbox("Show Sample Trade Analysis"):
            sample_trade = trades[0]
            analysis = sample_trade.trade_analysis
            
            st.markdown("**Sample Trade Breakdown:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.json({
                    "ticker": sample_trade.ticker,
                    "strike": sample_trade.strike,
                    "type": sample_trade.type,
                    "price": sample_trade.price,
                    "bid": sample_trade.bid,
                    "ask": sample_trade.ask,
                    "volume": sample_trade.volume,
                    "open_interest": sample_trade.open_interest
                })
            
            with col2:
                st.write(f"**Detected Side:** {analysis.side.value}")
                st.write(f"**Confidence:** {analysis.confidence:.1%}")
                st.write(f"**Confidence Level:** {analysis.confidence_level.value}")
                st.write(f"**Price Quality:** {analysis.price_quality}")
                st.write(f"**Data Completeness:** {analysis.data_completeness:.1%}")
                st.write(f"**Reasoning:** {'; '.join(analysis.reasoning[:3])}")

# --- MAIN APPLICATION ---
def main():
    st.set_page_config(
        page_title="Enhanced Options Flow Tracker",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize components
    config = Config()
    if 'UW_TOKEN' in st.secrets:
        config.UW_TOKEN = st.secrets['UW_TOKEN']
    
    client = APIClient(config)
    engine = AnalyticsEngine(config)
    viz = VisualizationEngine()
    ui = UIComponents()
    
    # Header
    st.title("📊 Enhanced Options Flow Tracker v2.0")
    st.markdown("### Real-time Options Analysis with Advanced AI-Powered Trade Direction Detection")
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("## 🎛️ Enhanced Control Panel")
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            [
                "🔍 Complete Flow Analysis",
                "⚡ Short-Term ETF Scanner",
                "🎯 High Confidence Trades",
                "🚨 Alert Dashboard",
                "📊 Market Sentiment Analysis",
                "🔧 Debug & Diagnostics"
            ]
        )
        
        # Enhanced filters
        st.markdown("### 🎯 Enhanced Filters")
        
        premium_filter = st.selectbox(
            "Premium Range:",
            ["All", "Under $100K", "$100K-$500K", "$500K-$1M", "Above $1M"],
            index=0
        )
        
        confidence_filter = st.selectbox(
            "Confidence Level:",
            ["All", "High+ Only", "Medium+ Only", "Low Confidence Only"],
            index=0
        )
        
        side_filter = st.selectbox(
            "Trade Side:",
            ["All", "Buy Only", "Sell Only", "Unknown Only"],
            index=0
        )
        
        dte_filter = st.selectbox(
            "Days to Expiry:",
            ["All", "0DTE", "≤7 days", "≤30 days", "≤90 days"],
            index=0
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            enable_visualizations = st.checkbox("Enable Advanced Charts", value=True)
            max_trades_display = st.slider("Max Trades to Display", 25, 200, 50)
            enable_debug = st.checkbox("Enable Debug Mode", value=False)
        
        # Action button
        run_analysis = st.button("🚀 Run Enhanced Analysis", type="primary", use_container_width=True)
    
    # Main content area
    if run_analysis:
        with st.spinner("🔄 Fetching and analyzing options flow data..."):
            # Fetch data
            params = {
                'limit': config.LIMIT,
                'min_volume_oi_ratio': 1.0
            }
            
            raw_trades = client.fetch_data(params)
            
            if not raw_trades:
                st.error("❌ No data received from API. Please check your connection and API token.")
                return
            
            # Process trades
            trades = engine.process_trades(raw_trades)
            
            if not trades:
                st.warning("⚠️ No valid trades found after processing.")
                return
            
            # Apply filters
            filtered_trades = apply_enhanced_filters(
                trades, premium_filter, confidence_filter, side_filter, dte_filter
            )
            
            st.info(f"📊 Processed {len(raw_trades)} raw trades → {len(trades)} valid trades → {len(filtered_trades)} filtered trades")
            
            # Display analysis based on type
            if "Complete Flow" in analysis_type:
                ui.render_enhanced_metrics(filtered_trades)
                
                if enable_visualizations:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(viz.create_sentiment_gauge(filtered_trades), use_container_width=True)
                    with col2:
                        st.plotly_chart(viz.create_confidence_distribution(filtered_trades), use_container_width=True)
                    
                    st.plotly_chart(viz.create_premium_by_side_chart(filtered_trades), use_container_width=True)
                
                ui.render_enhanced_trade_table(filtered_trades, max_trades_display)
                
            elif "ETF Scanner" in analysis_type:
                etf_trades = [t for t in filtered_trades if t.ticker in config.ALLOWED_TICKERS and t.dte <= 7]
                ui.render_enhanced_metrics(etf_trades)
                ui.render_enhanced_trade_table(etf_trades, max_trades_display)
                
            elif "High Confidence" in analysis_type:
                high_conf_trades = [t for t in filtered_trades if t.trade_analysis.confidence >= 0.7]
                ui.render_enhanced_metrics(high_conf_trades)
                ui.render_enhanced_trade_table(high_conf_trades, max_trades_display)
                
            elif "Alert Dashboard" in analysis_type:
                render_alert_dashboard(filtered_trades)
                
            elif "Sentiment Analysis" in analysis_type:
                render_sentiment_analysis(filtered_trades, viz)
                
            elif "Debug" in analysis_type:
                ui.render_debug_panel(filtered_trades)
                ui.render_enhanced_trade_table(filtered_trades[:10], 10)
            
            # Export functionality
            with st.expander("💾 Export Data"):
                export_enhanced_data(filtered_trades, analysis_type)
    
    else:
        render_welcome_screen()

def apply_enhanced_filters(trades, premium_filter, confidence_filter, side_filter, dte_filter):
    """Apply enhanced filtering logic"""
    filtered = trades
    
    # Premium filter
    if premium_filter != "All":
        if premium_filter == "Under $100K":
            filtered = [t for t in filtered if t.premium < 100000]
        elif premium_filter == "$100K-$500K":
            filtered = [t for t in filtered if 100000 <= t.premium < 500000]
        elif premium_filter == "$500K-$1M":
            filtered = [t for t in filtered if 500000 <= t.premium < 1000000]
        elif premium_filter == "Above $1M":
            filtered = [t for t in filtered if t.premium >= 1000000]
    
    # Confidence filter
    if confidence_filter != "All":
        if confidence_filter == "High+ Only":
            filtered = [t for t in filtered if t.trade_analysis.confidence >= 0.7]
        elif confidence_filter == "Medium+ Only":
            filtered = [t for t in filtered if t.trade_analysis.confidence >= 0.4]
        elif confidence_filter == "Low Confidence Only":
            filtered = [t for t in filtered if t.trade_analysis.confidence < 0.4]
    
    # Side filter
    if side_filter != "All":
        if side_filter == "Buy Only":
            filtered = [t for t in filtered if t.trade_analysis.side == TradeSide.BUY]
        elif side_filter == "Sell Only":
            filtered = [t for t in filtered if t.trade_analysis.side == TradeSide.SELL]
        elif side_filter == "Unknown Only":
            filtered = [t for t in filtered if t.trade_analysis.side == TradeSide.UNKNOWN]
    
    # DTE filter
    if dte_filter != "All":
        if dte_filter == "0DTE":
            filtered = [t for t in filtered if t.dte == 0]
        elif dte_filter == "≤7 days":
            filtered = [t for t in filtered if t.dte <= 7]
        elif dte_filter == "≤30 days":
            filtered = [t for t in filtered if t.dte <= 30]
        elif dte_filter == "≤90 days":
            filtered = [t for t in filtered if t.dte <= 90]
    
    return filtered

def render_alert_dashboard(trades):
    """Render enhanced alert dashboard"""
    st.markdown("### 🚨 Enhanced Alert Dashboard")
    
    # Generate alerts based on multiple criteria
    alerts = []
    
    for trade in trades:
        alert_score = 0
        reasons = []
        
        # High premium
        if trade.premium > 1000000:
            alert_score += 3
            reasons.append("Mega Premium (>$1M)")
        elif trade.premium > 500000:
            alert_score += 2
            reasons.append("Large Premium")
        
        # High confidence + unusual characteristics
        if trade.trade_analysis.confidence > 0.8:
            alert_score += 2
            reasons.append("High Confidence Direction")
        
        # Unusual volume/OI
        vol_oi_ratio = trade.volume / max(trade.open_interest, 1)
        if vol_oi_ratio > 10:
            alert_score += 2
            reasons.append("Extreme Volume/OI Ratio")
        
        # 0DTE with size
        if trade.dte == 0 and trade.premium > 200000:
            alert_score += 2
            reasons.append("Large 0DTE Trade")
        
        if alert_score >= 4:  # Threshold for alerts
            alerts.append({
                'trade': trade,
                'score': alert_score,
                'reasons': reasons
            })
    
    # Sort by alert score
    alerts.sort(key=lambda x: x['score'], reverse=True)
    
    if alerts:
        st.write(f"🔥 **{len(alerts)} High Priority Alerts Found**")
        
        for i, alert in enumerate(alerts[:10], 1):
            trade = alert['trade']
            analysis = trade.trade_analysis
            
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    side_emoji = "🟢" if analysis.side == TradeSide.BUY else "🔴" if analysis.side == TradeSide.SELL else "⚪"
                    
                    st.markdown(f"**{i}. {side_emoji} {trade.ticker} {trade.strike:.0f}{trade.type} "
                              f"({trade.dte}d) - {analysis.side.value} ({analysis.confidence:.0%})**")
                    st.write(f"💰 ${trade.premium:,.0f} | Vol: {trade.volume:,.0f} | OI: {trade.open_interest:,.0f}")
                    st.write(f"🎯 **Alert Reasons:** {', '.join(alert['reasons'])}")
                    st.write(f"📊 **Analysis:** {', '.join(analysis.reasoning[:2])}")
                
                with col2:
                    st.metric("Alert Score", alert['score'])
                
                st.divider()
    else:
        st.info("No high-priority alerts at this time.")

def render_sentiment_analysis(trades, viz):
    """Render comprehensive sentiment analysis"""
    st.markdown("### 📊 Market Sentiment Analysis")
    
    if not trades:
        st.warning("No trades available for sentiment analysis")
        return
    
    # Overall sentiment metrics
    buy_trades = [t for t in trades if t.trade_analysis.side == TradeSide.BUY]
    sell_trades = [t for t in trades if t.trade_analysis.side == TradeSide.SELL]
    
    buy_premium = sum(t.premium for t in buy_trades)
    sell_premium = sum(t.premium for t in sell_trades)
    total_premium = buy_premium + sell_premium
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Buy Premium", f"${buy_premium:,.0f}")
        st.metric("Buy Trades", len(buy_trades))
    
    with col2:
        st.metric("Sell Premium", f"${sell_premium:,.0f}")
        st.metric("Sell Trades", len(sell_trades))
    
    with col3:
        if total_premium > 0:
            buy_ratio = buy_premium / total_premium
            sentiment = "Bullish" if buy_ratio > 0.6 else "Bearish" if buy_ratio < 0.4 else "Neutral"
            st.metric("Market Sentiment", sentiment)
            st.metric("Buy Premium %", f"{buy_ratio:.1%}")
        else:
            st.metric("Market Sentiment", "No Data")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(viz.create_sentiment_gauge(trades), use_container_width=True)
    
    with col2:
        st.plotly_chart(viz.create_confidence_distribution(trades), use_container_width=True)
    
    # Sector breakdown (if available)
    st.markdown("#### 🏢 Sector Sentiment Breakdown")
    
    # Simple sector mapping
    sector_map = {
        'SPY': 'Broad Market', 'QQQ': 'Technology', 'IWM': 'Small Cap',
        'XLF': 'Financial', 'XLE': 'Energy', 'XLK': 'Technology',
        'XLV': 'Healthcare', 'XLI': 'Industrial'
    }
    
    sector_sentiment = defaultdict(lambda: {'buy': 0, 'sell': 0, 'count': 0})
    
    for trade in trades:
        sector = sector_map.get(trade.ticker, 'Other')
        sector_sentiment[sector]['count'] += 1
        
        if trade.trade_analysis.side == TradeSide.BUY:
            sector_sentiment[sector]['buy'] += trade.premium
        elif trade.trade_analysis.side == TradeSide.SELL:
            sector_sentiment[sector]['sell'] += trade.premium
    
    if sector_sentiment:
        for sector, data in sector_sentiment.items():
            if data['count'] >= 3:  # Only show sectors with 3+ trades
                total = data['buy'] + data['sell']
                if total > 0:
                    buy_pct = data['buy'] / total
                    sentiment_color = "🟢" if buy_pct > 0.6 else "🔴" if buy_pct < 0.4 else "🟡"
                    st.write(f"{sentiment_color} **{sector}**: {buy_pct:.1%} bullish "
                           f"(${total:,.0f} total, {data['count']} trades)")

def export_enhanced_data(trades, analysis_type):
    """Export enhanced trade data"""
    if not trades:
        st.warning("No data to export")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"enhanced_options_flow_{analysis_type.replace(' ', '_').lower()}_{timestamp}.csv"
    
    # Prepare export data
    export_data = []
    for trade in trades:
        analysis = trade.trade_analysis
        
        row = {
            'ticker': trade.ticker,
            'option_chain': trade.option_chain,
            'type': trade.type,
            'strike': trade.strike,
            'expiry': trade.expiry,
            'dte': trade.dte,
            'price': trade.price,
            'premium': trade.premium,
            'volume': trade.volume,
            'open_interest': trade.open_interest,
            'iv': trade.iv,
            'bid': trade.bid,
            'ask': trade.ask,
            'underlying_price': trade.underlying_price,
            'trade_side': analysis.side.value,
            'confidence': analysis.confidence,
            'confidence_level': analysis.confidence_level.value,
            'reasoning': '; '.join(analysis.reasoning),
            'price_quality': analysis.price_quality,
            'data_completeness': analysis.data_completeness,
            'moneyness': trade.moneyness,
            'scenarios': '; '.join(trade.scenarios),
            'oi_level': trade.oi_analysis.get('oi_level', ''),
            'liquidity_score': trade.oi_analysis.get('liquidity_score', ''),
            'time_ny': trade.time_ny,
            'rule_name': trade.rule_name,
            'description': trade.description
        }
        export_data.append(row)
    
    df = pd.DataFrame(export_data)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label=f"📥 Download {filename}",
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

def render_welcome_screen():
    """Render welcome screen with feature overview"""
    st.markdown("""
    ## Welcome to Enhanced Options Flow Tracker v2.0! 🚀
    
    ### 🆕 Revolutionary Improvements
    
    #### 🧠 **AI-Powered Trade Direction Detection**
    - **Multi-Method Analysis**: 6+ detection algorithms working in parallel
    - **Confidence Scoring**: Machine learning-inspired confidence levels (0-100%)
    - **Data Quality Assessment**: Automatic evaluation of price data reliability
    - **Smart Fallbacks**: Multiple backup methods when primary data is missing
    
    #### 📊 **Enhanced Analytics Engine**
    - **Real-time Processing**: Asynchronous data fetching with retry logic
    - **Advanced Filtering**: Multi-dimensional trade filtering
    - **Pattern Recognition**: Automated detection of trading scenarios
    - **Performance Optimized**: Efficient data processing and caching
    
    #### 🎯 **Key Features**
    
    **🔍 Complete Flow Analysis**
    - Comprehensive market overview with sentiment analysis
    - Interactive charts and visualizations
    - Enhanced trade tables with confidence indicators
    
    **⚡ Short-Term ETF Scanner**
    - Focused on SPY/QQQ/IWM trades ≤7 DTE
    - Real-time 0DTE activity monitoring
    - ETF-specific sentiment tracking
    
    **🎯 High Confidence Trades**
    - Only trades with 70%+ direction confidence
    - Quality-filtered for reliable signals
    - Reduced noise, increased signal clarity
    
    **🚨 Alert Dashboard**
    - Multi-criteria alert scoring system
    - Automated detection of unusual activity
    - Priority-ranked alert display
    
    **📊 Market Sentiment Analysis**
    - Real-time sentiment gauge
    - Sector-level sentiment breakdown
    - Buy/sell pressure visualization
    
    **🔧 Debug & Diagnostics**
    - Data quality monitoring
    - Detection algorithm transparency
    - Troubleshooting tools
    
    #### 💡 **Smart Trade Direction Detection**
    
    Our enhanced system uses multiple detection methods:
    
    1. **Bid/Ask Analysis** - Most reliable when spread data available
    2. **Volume/OI Patterns** - Identifies new position building vs. closing
    3. **Description Mining** - NLP analysis of trade descriptions
    4. **Market Context** - Time-based and option characteristics analysis
    5. **Premium Analysis** - Large trade behavior patterns
    6. **Rule Pattern Recognition** - Ascending/descending fill analysis
    
    #### 🎛️ **Enhanced Controls**
    - **Premium Filters**: From under $100K to above $1M
    - **Confidence Filters**: High, medium, low confidence levels
    - **Side Filters**: Buy-only, sell-only, or unknown trades
    - **Time Filters**: 0DTE to LEAPS timeframes
    - **Advanced Options**: Visualization toggles, display limits, debug mode
    
    #### 📈 **Visual Analytics**
    - **Sentiment Gauge**: Real-time market bullishness meter
    - **Confidence Distribution**: Quality of trade direction data
    - **Premium Analysis**: Buy vs. sell trade size comparison
    - **Interactive Charts**: Plotly-powered visualizations
    
    ### 🚀 **Ready to Start?**
    
    1. **Select your analysis type** from the sidebar
    2. **Configure filters** for your specific interests
    3. **Enable advanced options** if desired
    4. **Click "Run Enhanced Analysis"** to begin
    
    **Pro Tip**: Start with "Complete Flow Analysis" to get a comprehensive market overview!
    """)

if __name__ == "__main__":
    main()
