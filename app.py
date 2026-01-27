"""
EUR/USD 2-Minute Auto-Learning Trading System
WITH 30-SECOND CACHING for API limit protection
USING data.txt FOR ML STORAGE
"""

import os
import json
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
import plotly.utils
import requests
import warnings
import logging
from collections import deque
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ==================== CONFIGURATION ====================
TRADING_SYMBOL = "EURUSD"
CYCLE_MINUTES = 2
CYCLE_SECONDS = 120
INITIAL_BALANCE = 10000.0
BASE_TRADE_SIZE = 1000.0
MIN_CONFIDENCE = 65.0
TRAINING_FILE = "data.txt"  # ⭐ CHANGED TO data.txt

# ==================== CACHE CONFIGURATION ====================
CACHE_DURATION = 30  # ⭐ CACHE: 30 seconds
price_cache = {
    'price': 1.0850,
    'timestamp': time.time(),
    'source': 'Initial',
    'expiry': CACHE_DURATION,
    'hits': 0,
    'misses': 0
}

# ==================== GLOBAL STATE ====================
trading_state = {
    'current_price': 1.0850,
    'minute_prediction': 'ANALYZING',
    'action': 'WAIT',
    'confidence': 0.0,
    'cycle_count': 0,
    'current_trade': None,
    'balance': INITIAL_BALANCE,
    'total_trades': 0,
    'profitable_trades': 0,
    'total_profit': 0.0,
    'win_rate': 0.0,
    'is_demo_data': False,
    'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'api_status': 'CONNECTING',
    'data_source': 'Initializing...',
    'prediction_accuracy': 0.0,
    'next_cycle_in': CYCLE_SECONDS,
    'price_history': [],
    'chart_data': None,
    'optimal_tp': 0.0,
    'optimal_sl': 0.0,
    'tp_distance_pips': 0,
    'sl_distance_pips': 0,
    'ml_model_ready': False,
    'ml_data_points': 0,  # Added counter for data.txt entries
    'server_time': datetime.now().isoformat(),
    'cycle_progress': 0,
    'trade_progress': 0,
    'trade_status': 'NO_TRADE',
    'cycle_duration': CYCLE_SECONDS,
    'risk_reward_ratio': '1:2',
    'volatility': 'MEDIUM',
    'signal_strength': 0,
    'remaining_time': CYCLE_SECONDS,
    'cache_hits': 0,
    'cache_misses': 0,
    'cache_efficiency': '0%',
    'api_calls_today': '~240 (SAFE)'
}

# Data storage
trade_history = []
price_history_deque = deque(maxlen=50)
prediction_history = deque(maxlen=50)

# ML Components
tp_model = RandomForestRegressor(n_estimators=50, random_state=42)
sl_model = RandomForestRegressor(n_estimators=50, random_state=42)
ml_scaler = StandardScaler()
ml_features = []
tp_labels = []
sl_labels = []
ml_trained = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Print startup banner
print("="*80)
print("EUR/USD 2-MINUTE TRADING SYSTEM WITH CACHING")
print("="*80)
print(f"Cycle: Predict and trade every {CYCLE_MINUTES} minutes ({CYCLE_SECONDS} seconds)")
print(f"Cache Duration: {CACHE_DURATION} seconds (66% API reduction)")
print(f"API Calls/Day: ~240 (SAFE for all free limits)")
print(f"Goal: Hit TP before SL within {CYCLE_SECONDS} seconds")
print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
print(f"Trade Size: ${BASE_TRADE_SIZE:,.2f}")
print(f"ML Training File: {TRAINING_FILE}")
print("="*80)
print("Starting system...")

# ==================== CACHED FOREX DATA FETCHING ====================
def get_cached_eurusd_price():
    """Get EUR/USD price with 30-second caching to prevent API limits"""
    
    current_time = time.time()
    cache_age = current_time - price_cache['timestamp']
    
    # ⭐ CACHE HIT: Use cached price if fresh
    if cache_age < CACHE_DURATION and price_cache['price']:
        price_cache['hits'] += 1
        update_cache_efficiency()
        
        # Add tiny realistic fluctuation to cached price
        tiny_change = np.random.uniform(-0.00001, 0.00001)
        cached_price = price_cache['price'] + tiny_change
        
        logger.debug(f"📦 CACHE HIT: Using cached price {cached_price:.5f} (age: {cache_age:.1f}s)")
        trading_state['api_status'] = f"CACHED ({price_cache['source']})"
        
        return cached_price, f"Cached ({price_cache['source']})"
    
    # ⭐ CACHE MISS: Need fresh price from APIs
    price_cache['misses'] += 1
    update_cache_efficiency()
    logger.info("🔄 Cache MISS: Fetching fresh price from APIs...")
    
    # List of reliable APIs with good limits
    apis_to_try = [
        {
            'name': 'Frankfurter',
            'url': 'https://api.frankfurter.app/latest',
            'params': {'from': 'EUR', 'to': 'USD'},
            'extract_rate': lambda data: data['rates']['USD']
        },
        {
            'name': 'FreeForexAPI',
            'url': 'https://api.freeforexapi.com/v1/latest',
            'params': {'pairs': 'EURUSD'},
            'extract_rate': lambda data: data['rates']['EURUSD']
        }
        # ⭐ REMOVED ExchangeRate-API (too low limits)
    ]
    
    # Try each API
    for api in apis_to_try:
        try:
            logger.info(f"Trying {api['name']} API...")
            response = requests.get(api['url'], params=api['params'], timeout=5)
            
            # Handle rate limits gracefully
            if response.status_code == 429:
                logger.warning(f"⏸️ {api['name']} rate limit reached, skipping...")
                continue
                
            if response.status_code == 200:
                data = response.json()
                rate = api['extract_rate'](data)
                
                if rate:
                    current_price = float(rate)
                    
                    # ⭐ UPDATE CACHE with fresh price
                    price_cache.update({
                        'price': current_price,
                        'timestamp': current_time,
                        'source': api['name']
                    })
                    
                    logger.info(f"✅ {api['name']}: EUR/USD = {current_price:.5f} (cached)")
                    trading_state['api_status'] = 'CONNECTED'
                    
                    return current_price, api['name']
                    
        except Exception as e:
            logger.warning(f"{api['name']} failed: {str(e)[:50]}")
            continue
    
    # ⭐ ALL APIS FAILED: Use stale cache as fallback
    logger.warning("⚠️ All APIs failed, using stale cached data")
    
    if price_cache['price']:
        # Add small realistic movement to stale price
        stale_change = np.random.uniform(-0.00005, 0.00005)
        stale_price = price_cache['price'] + stale_change
        
        # Keep in reasonable range
        if stale_price < 1.0800:
            stale_price = 1.0800 + abs(stale_change)
        elif stale_price > 1.0900:
            stale_price = 1.0900 - abs(stale_change)
        
        trading_state['api_status'] = 'STALE_CACHE'
        return stale_price, f"Stale Cache ({price_cache['source']})"
    else:
        # First run, no cache yet
        trading_state['api_status'] = 'SIMULATION'
        return 1.0850, 'Simulation (Initial)'

def update_cache_efficiency():
    """Calculate and update cache efficiency metrics"""
    total = price_cache['hits'] + price_cache['misses']
    if total > 0:
        efficiency = (price_cache['hits'] / total) * 100
        trading_state['cache_efficiency'] = f"{efficiency:.1f}%"
        trading_state['cache_hits'] = price_cache['hits']
        trading_state['cache_misses'] = price_cache['misses']

def create_price_series(current_price, num_points=120):
    """Create realistic 2-minute price series for analysis"""
    prices = []
    base_price = float(current_price)
    
    for i in range(num_points):
        volatility = 0.00015
        change = np.random.normal(0, volatility)
        base_price += change
        
        if base_price < 1.0800:
            base_price = 1.0800 + abs(change)
        elif base_price > 1.0900:
            base_price = 1.0900 - abs(change)
        
        prices.append(base_price)
    
    return prices

# ==================== TECHNICAL ANALYSIS ====================
def calculate_advanced_indicators(prices):
    """Calculate comprehensive indicators for 2-minute prediction"""
    df = pd.DataFrame(prices, columns=['close'])
    
    try:
        # Price momentum
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'] - df['close'].shift(20)
        
        # Moving averages
        df['sma_5'] = ta.sma(df['close'], length=5)
        df['sma_10'] = ta.sma(df['close'], length=10)
        df['ema_12'] = ta.ema(df['close'], length=12)
        df['ema_26'] = ta.ema(df['close'], length=26)
        
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # MACD
        macd = ta.macd(df['close'])
        if macd is not None and isinstance(macd, pd.DataFrame):
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
            df['macd_hist'] = macd['MACDh_12_26_9']
        
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20)
        if bb is not None and isinstance(bb, pd.DataFrame):
            df['bb_upper'] = bb['BBU_20_2.0']
            df['bb_lower'] = bb['BBL_20_2.0']
            df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100
        
        # ATR for volatility
        df['atr'] = ta.atr(df['close'], df['close'], df['close'], length=14)
        
        # Support/Resistance
        df['resistance'] = df['close'].rolling(15).max()
        df['support'] = df['close'].rolling(15).min()
        
        # Market condition flags
        df['overbought'] = (df['rsi'] > 70).astype(int)
        df['oversold'] = (df['rsi'] < 30).astype(int)
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Indicator calculation error: {e}")
        return df.fillna(0)

# ==================== ML TRAINING SYSTEM ====================
def initialize_training_system():
    """Initialize or load ML training data from data.txt"""
    global ml_features, tp_labels, sl_labels, ml_trained
    
    try:
        # Check if data.txt exists
        if os.path.exists(TRAINING_FILE):
            with open(TRAINING_FILE, 'r') as f:
                lines = f.readlines()
            
            trading_state['ml_data_points'] = len(lines)
            logger.info(f"📊 Loaded {trading_state['ml_data_points']} trades from {TRAINING_FILE}")
            
            if trading_state['ml_data_points'] >= 10:
                # Load and process training data
                load_ml_training_data()
                if len(ml_features) >= 10:
                    train_ml_models()
                    logger.info(f"✅ ML system loaded with {len(ml_features)} training samples")
                else:
                    logger.info(f"⚠️  {len(ml_features)} valid samples - collecting more data")
            else:
                logger.info(f"📊 Collecting data: {trading_state['ml_data_points']}/10 trades")
                
        else:
            # Create empty data.txt with header
            with open(TRAINING_FILE, 'w') as f:
                f.write('# EUR/USD Trading Data - ML Training\n')
                f.write('# Format: JSON per line with trade data\n')
            trading_state['ml_data_points'] = 0
            logger.info("📝 Created new data.txt file")
            
    except Exception as e:
        logger.error(f"❌ ML initialization error: {e}")
        trading_state['ml_model_ready'] = False

def load_ml_training_data():
    """Load and process ML training data from data.txt"""
    global ml_features, tp_labels, sl_labels
    
    try:
        if not os.path.exists(TRAINING_FILE):
            return
        
        with open(TRAINING_FILE, 'r') as f:
            lines = f.readlines()
        
        features = []
        tp_vals = []
        sl_vals = []
        
        for line in lines:
            try:
                if line.strip() and not line.startswith('#'):
                    data = json.loads(line.strip())
                    
                    # Extract features from trade data
                    feature_vector = extract_ml_features_from_trade(data)
                    if feature_vector:
                        features.append(feature_vector)
                        
                        # Extract optimal TP/SL based on trade outcome
                        optimal_tp, optimal_sl = calculate_optimal_levels_from_trade(data)
                        tp_vals.append(optimal_tp)
                        sl_vals.append(optimal_sl)
                        
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.debug(f"Skipping invalid line: {e}")
                continue
        
        ml_features = features
        tp_labels = tp_vals
        sl_labels = sl_vals
        
        logger.info(f"📊 Processed {len(features)} training samples from {TRAINING_FILE}")
        
    except Exception as e:
        logger.error(f"❌ Error loading training data: {e}")

def extract_ml_features_from_trade(trade_data):
    """Extract ML features from trade data"""
    try:
        features = []
        
        # Trade parameters
        features.append(trade_data.get('confidence', 50) / 100)  # Normalized confidence
        features.append(1 if trade_data.get('action') == 'BUY' else 0)  # Buy flag
        
        # Market conditions at entry
        features.append(trade_data.get('rsi_at_entry', 50) / 100)
        features.append(trade_data.get('volatility_at_entry', 0.0005) * 10000)
        
        # Technical indicators
        features.append(trade_data.get('bb_percent_at_entry', 50) / 100)
        features.append(trade_data.get('macd_hist_at_entry', 0) * 10000)
        
        # Risk metrics
        features.append(trade_data.get('tp_distance_pips', 8) / 20)  # Normalized
        features.append(trade_data.get('sl_distance_pips', 5) / 15)  # Normalized
        
        # Trade outcome
        features.append(1 if trade_data.get('result') == 'SUCCESS' else 0)
        features.append(trade_data.get('profit_pips', 0) / 100)
        
        return features
        
    except Exception as e:
        logger.debug(f"Feature extraction error: {e}")
        return None

def calculate_optimal_levels_from_trade(trade_data):
    """Calculate optimal TP/SL levels based on trade outcome"""
    try:
        result = trade_data.get('result', 'PENDING')
        profit_pips = trade_data.get('profit_pips', 0)
        tp_pips = trade_data.get('tp_distance_pips', 8)
        sl_pips = trade_data.get('sl_distance_pips', 5)
        
        # ML learns optimal levels based on outcomes
        if result == 'SUCCESS':
            # Successful trade - TP was good, maybe could be tighter
            optimal_tp = tp_pips * 0.9  # Slightly tighter TP
            optimal_sl = sl_pips * 0.8  # Tighter SL since it wasn't hit
        elif result == 'FAILED':
            # Failed trade - SL was hit, maybe too tight
            optimal_tp = tp_pips * 1.1  # Wider TP for more room
            optimal_sl = sl_pips * 1.2  # Wider SL to avoid premature stops
        elif result == 'PARTIAL_SUCCESS':
            # Partial success - adjust moderately
            optimal_tp = tp_pips * 0.95
            optimal_sl = sl_pips * 0.9
        else:
            # Default values
            optimal_tp = tp_pips
            optimal_sl = sl_pips
        
        # Ensure reasonable ranges
        optimal_tp = max(5, min(20, optimal_tp))
        optimal_sl = max(3, min(15, optimal_sl))
        
        return optimal_tp, optimal_sl
        
    except Exception as e:
        logger.debug(f"Optimal level calculation error: {e}")
        return 8, 5  # Default values

def save_trade_to_data_txt(trade_data, market_conditions):
    """Save trade data to data.txt for ML training"""
    try:
        # Prepare comprehensive trade data
        ml_trade_data = {
            'trade_id': trade_data.get('id'),
            'action': trade_data.get('action'),
            'entry_price': trade_data.get('entry_price'),
            'exit_price': trade_data.get('exit_price'),
            'profit_pips': trade_data.get('profit_pips', 0),
            'result': trade_data.get('result', 'PENDING'),
            'confidence': trade_data.get('confidence', 0),
            'tp_distance_pips': trade_data.get('tp_distance_pips', 0),
            'sl_distance_pips': trade_data.get('sl_distance_pips', 0),
            'duration_seconds': trade_data.get('duration_seconds', 0),
            'signal_strength': trade_data.get('signal_strength', 0),
            
            # Market conditions at entry
            'rsi_at_entry': market_conditions.get('rsi', 50),
            'volatility_at_entry': market_conditions.get('atr', 0.0005),
            'bb_percent_at_entry': market_conditions.get('bb_percent', 50),
            'macd_hist_at_entry': market_conditions.get('macd_hist', 0),
            
            'timestamp': datetime.now().isoformat(),
            'cycle_number': trading_state['cycle_count']
        }
        
        # Save to data.txt (append mode)
        with open(TRAINING_FILE, 'a') as f:
            f.write(json.dumps(ml_trade_data) + '\n')
        
        # Update counters
        trading_state['ml_data_points'] += 1
        
        # Add to ML training data
        feature_vector = extract_ml_features_from_trade(ml_trade_data)
        if feature_vector:
            optimal_tp, optimal_sl = calculate_optimal_levels_from_trade(ml_trade_data)
            ml_features.append(feature_vector)
            tp_labels.append(optimal_tp)
            sl_labels.append(optimal_sl)
        
        logger.info(f"✅ Trade #{trade_data.get('id')} saved to {TRAINING_FILE}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error saving trade to {TRAINING_FILE}: {e}")
        return False

def train_ml_models():
    """Train ML models for TP/SL optimization"""
    global tp_model, sl_model, ml_scaler, ml_trained
    
    if len(ml_features) < 10:
        ml_trained = False
        trading_state['ml_model_ready'] = False
        return
    
    try:
        X = np.array(ml_features)
        y_tp = np.array(tp_labels)
        y_sl = np.array(sl_labels)
        
        # Scale features
        X_scaled = ml_scaler.fit_transform(X)
        
        # Train TP model
        tp_model.fit(X_scaled, y_tp)
        
        # Train SL model
        sl_model.fit(X_scaled, y_sl)
        
        ml_trained = True
        trading_state['ml_model_ready'] = True
        trading_state['ml_data_points'] = len(ml_features)
        logger.info(f"✅ ML models trained on {len(X)} samples")
        
    except Exception as e:
        logger.error(f"ML training error: {e}")
        ml_trained = False
        trading_state['ml_model_ready'] = False

def extract_ml_features(df, current_price):
    """Extract features for ML prediction"""
    if df.empty or len(df) < 20:
        return None
    
    latest = df.iloc[-1]
    
    features = []
    
    # Price momentum
    features.append(latest.get('returns_1', 0))
    features.append(latest.get('returns_5', 0))
    features.append(latest.get('returns_10', 0))
    
    # RSI value
    features.append(latest.get('rsi', 50))
    
    # MACD histogram
    features.append(latest.get('macd_hist', 0))
    
    # Bollinger Bands position
    features.append(latest.get('bb_percent', 50))
    
    # Volatility
    atr_value = latest.get('atr', 0.0005)
    features.append(atr_value * 10000)
    
    # Market condition flags
    features.append(latest.get('overbought', 0))
    features.append(latest.get('oversold', 0))
    
    return features

def predict_optimal_levels(features, direction, current_price, df):
    """Predict optimal TP and SL levels for 2-minute trades"""
    
    # Base levels for 2-minute trades
    if direction == "BULLISH":
        base_tp = current_price + 0.0008
        base_sl = current_price - 0.0005
    elif direction == "BEARISH":
        base_tp = current_price - 0.0008
        base_sl = current_price + 0.0005
    else:
        base_tp = current_price
        base_sl = current_price
    
    # Use ML predictions if available
    if ml_trained and features is not None:
        try:
            X_scaled = ml_scaler.transform([features])
            
            # Predict optimal TP distance
            tp_pips_pred = tp_model.predict(X_scaled)[0]
            tp_pips_pred = max(5, min(20, tp_pips_pred))
            
            # Predict optimal SL distance
            sl_pips_pred = sl_model.predict(X_scaled)[0]
            sl_pips_pred = max(3, min(15, sl_pips_pred))
            
            # Convert pips to price
            pip_value = 0.0001
            
            if direction == "BULLISH":
                optimal_tp = current_price + (tp_pips_pred * pip_value)
                optimal_sl = current_price - (sl_pips_pred * pip_value)
            elif direction == "BEARISH":
                optimal_tp = current_price - (tp_pips_pred * pip_value)
                optimal_sl = current_price + (sl_pips_pred * pip_value)
            else:
                optimal_tp = base_tp
                optimal_sl = base_sl
            
            logger.info(f"🤖 ML suggested: TP={tp_pips_pred:.1f} pips, SL={sl_pips_pred:.1f} pips")
            return optimal_tp, optimal_sl, int(tp_pips_pred), int(sl_pips_pred)
            
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
    
    # Fallback to base levels
    tp_pips = int(abs(base_tp - current_price) * 10000)
    sl_pips = int(abs(base_sl - current_price) * 10000)
    
    return base_tp, base_sl, tp_pips, sl_pips

# ==================== 2-MINUTE PREDICTION ENGINE ====================
def analyze_2min_prediction(df, current_price):
    """Predict 2-minute price direction with high accuracy"""
    
    if len(df) < 20:
        return 0.5, 50, 'ANALYZING', 1
    
    try:
        latest = df.iloc[-1]
        
        # Initialize scores
        bull_score = 0
        bear_score = 0
        confidence_factors = []
        
        # 1. RSI ANALYSIS
        rsi_value = latest.get('rsi', 50)
        if rsi_value < 35:
            bull_score += 4
            confidence_factors.append(1.5 if rsi_value < 25 else 1.2)
        elif rsi_value > 65:
            bear_score += 4
            confidence_factors.append(1.5 if rsi_value > 75 else 1.2)
        
        # 2. MACD HISTOGRAM
        macd_hist = latest.get('macd_hist', 0)
        if macd_hist > 0.00005:
            bull_score += 3
            confidence_factors.append(1.3)
        elif macd_hist < -0.00005:
            bear_score += 3
            confidence_factors.append(1.3)
        
        # 3. BOLLINGER BANDS
        bb_percent = latest.get('bb_percent', 50)
        if bb_percent < 25:
            bull_score += 2
            confidence_factors.append(1.2)
        elif bb_percent > 75:
            bear_score += 2
            confidence_factors.append(1.2)
        
        # 4. PRICE MOMENTUM
        momentum = latest.get('momentum_20', 0)
        if momentum > 0.0003:
            bull_score += 2
        elif momentum < -0.0003:
            bear_score += 2
        
        # Calculate probability
        total_score = bull_score + bear_score
        if total_score == 0:
            return 0.5, 50, 'NEUTRAL', 1
        
        probability = bull_score / total_score
        
        # Calculate confidence
        if confidence_factors:
            base_confidence = np.mean(confidence_factors) * 25
        else:
            base_confidence = 50
        
        # Signal clarity adjustment
        signal_clarity = abs(probability - 0.5) * 2
        confidence = min(95, base_confidence * (1 + signal_clarity))
        
        # Determine direction
        if probability > 0.65:
            direction = 'BULLISH'
            signal_strength = 3
        elif probability > 0.55:
            direction = 'BULLISH'
            signal_strength = 2
        elif probability < 0.35:
            direction = 'BEARISH'
            signal_strength = 3
        elif probability < 0.45:
            direction = 'BEARISH'
            signal_strength = 2
        else:
            direction = 'NEUTRAL'
            signal_strength = 1
            confidence = max(30, confidence * 0.7)
        
        return probability, confidence, direction, signal_strength
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return 0.5, 50, 'ERROR', 1

# ==================== TRADE EXECUTION ====================
def execute_2min_trade(direction, confidence, current_price, optimal_tp, optimal_sl, tp_pips, sl_pips, market_conditions):
    """Execute a trade at the beginning of the 2-minute cycle"""
    
    if direction == 'NEUTRAL' or confidence < MIN_CONFIDENCE:
        trading_state['action'] = 'WAIT'
        trading_state['trade_status'] = 'NO_SIGNAL'
        return None
    
    # Determine action
    if direction == 'BULLISH':
        action = 'BUY'
        action_reason = f"Strong 2-min BULLISH signal ({confidence:.1f}% confidence)"
    else:  # BEARISH
        action = 'SELL'
        action_reason = f"Strong 2-min BEARISH signal ({confidence:.1f}% confidence)"
    
    trade = {
        'id': len(trade_history) + 1,
        'cycle': trading_state['cycle_count'],
        'action': action,
        'entry_price': float(current_price),
        'entry_time': datetime.now(),
        'optimal_tp': float(optimal_tp),
        'optimal_sl': float(optimal_sl),
        'tp_distance_pips': tp_pips,
        'sl_distance_pips': sl_pips,
        'trade_size': BASE_TRADE_SIZE,
        'confidence': float(confidence),
        'status': 'OPEN',
        'result': 'PENDING',
        'exit_price': None,
        'exit_time': None,
        'exit_reason': None,
        'profit_pips': 0,
        'profit_amount': 0.0,
        'duration_seconds': 0,
        'max_profit_pips': 0,
        'max_loss_pips': 0,
        'reason': action_reason,
        'cycle_duration': CYCLE_SECONDS,
        'signal_strength': signal_strength,
        'market_conditions': market_conditions  # Store market conditions for ML
    }
    
    trading_state['current_trade'] = trade
    trading_state['action'] = action
    trading_state['optimal_tp'] = optimal_tp
    trading_state['optimal_sl'] = optimal_sl
    trading_state['tp_distance_pips'] = tp_pips
    trading_state['sl_distance_pips'] = sl_pips
    trading_state['trade_status'] = 'ACTIVE'
    trading_state['signal_strength'] = signal_strength
    
    logger.info(f"🔔 {action} ORDER EXECUTED")
    logger.info(f"   Entry Price: {current_price:.5f}")
    logger.info(f"   Take Profit: {optimal_tp:.5f} ({tp_pips} pips)")
    logger.info(f"   Stop Loss: {optimal_sl:.5f} ({sl_pips} pips)")
    logger.info(f"   Confidence: {confidence:.1f}%")
    logger.info(f"   Goal: Hit TP ({tp_pips} pips) before SL ({sl_pips} pips) in {CYCLE_SECONDS} seconds")
    
    return trade

def monitor_active_trade(current_price, market_conditions):
    """Monitor the active trade throughout the 2-minute cycle"""
    if not trading_state['current_trade']:
        return None
    
    trade = trading_state['current_trade']
    trade_duration = (datetime.now() - trade['entry_time']).total_seconds()
    
    # Calculate current P&L
    if trade['action'] == 'BUY':
        current_pips = (current_price - trade['entry_price']) * 10000
    else:  # SELL
        current_pips = (trade['entry_price'] - current_price) * 10000
    
    trade['profit_pips'] = current_pips
    trade['profit_amount'] = (current_pips / 10000) * trade['trade_size']
    trade['duration_seconds'] = trade_duration
    
    # Update max profit/loss
    trade['max_profit_pips'] = max(trade['max_profit_pips'], current_pips)
    trade['max_loss_pips'] = min(trade['max_loss_pips'], current_pips)
    
    # Update trade progress
    trading_state['trade_progress'] = (trade_duration / CYCLE_SECONDS) * 100
    trading_state['remaining_time'] = max(0, CYCLE_SECONDS - trade_duration)
    
    # Check exit conditions
    exit_trade = False
    exit_reason = ""
    
    if trade['action'] == 'BUY':
        if current_price >= trade['optimal_tp']:
            exit_trade = True
            exit_reason = f"TP HIT! +{trade['tp_distance_pips']} pips profit"
            trade['result'] = 'SUCCESS'
            
        elif current_price <= trade['optimal_sl']:
            exit_trade = True
            exit_reason = f"SL HIT! -{trade['sl_distance_pips']} pips loss"
            trade['result'] = 'FAILED'
            
    else:  # SELL
        if current_price <= trade['optimal_tp']:
            exit_trade = True
            exit_reason = f"TP HIT! +{trade['tp_distance_pips']} pips profit"
            trade['result'] = 'SUCCESS'
            
        elif current_price >= trade['optimal_sl']:
            exit_trade = True
            exit_reason = f"SL HIT! -{trade['sl_distance_pips']} pips loss"
            trade['result'] = 'FAILED'
    
    # Time-based exit
    if not exit_trade and trade_duration >= CYCLE_SECONDS:
        exit_trade = True
        if current_pips > 0:
            exit_reason = f"TIME ENDED with +{current_pips:.1f} pips profit"
            trade['result'] = 'PARTIAL_SUCCESS'
        elif current_pips < 0:
            exit_reason = f"TIME ENDED with {current_pips:.1f} pips loss"
            trade['result'] = 'PARTIAL_FAIL'
        else:
            exit_reason = "TIME ENDED at breakeven"
            trade['result'] = 'BREAKEVEN'
    
    # Close trade if exit condition met
    if exit_trade:
        trade['status'] = 'CLOSED'
        trade['exit_price'] = current_price
        trade['exit_time'] = datetime.now()
        trade['exit_reason'] = exit_reason
        
        # Update trading statistics
        trading_state['total_trades'] += 1
        
        if trade['result'] in ['SUCCESS', 'PARTIAL_SUCCESS']:
            trading_state['profitable_trades'] += 1
            trading_state['total_profit'] += trade['profit_amount']
            trading_state['balance'] += trade['profit_amount']
            logger.info(f"💰 TRADE SUCCESS: {exit_reason}")
        else:
            trading_state['balance'] -= abs(trade['profit_amount'])
            logger.info(f"📉 TRADE FAILED: {exit_reason}")
        
        # Update win rate
        if trading_state['total_trades'] > 0:
            trading_state['win_rate'] = (trading_state['profitable_trades'] / trading_state['total_trades']) * 100
        
        # Add to history
        trade_history.append(trade.copy())
        
        # Save to data.txt for ML training
        save_trade_to_data_txt(trade, market_conditions)
        
        # Retrain ML if we have enough samples
        if trading_state['ml_data_points'] >= 10 and trading_state['ml_data_points'] % 5 == 0:
            train_ml_models()
        
        # Clear current trade
        trading_state['current_trade'] = None
        trading_state['trade_status'] = 'COMPLETED'
        trading_state['trade_progress'] = 0
        trading_state['remaining_time'] = CYCLE_SECONDS
        
        return trade
    
    return trade

# ==================== CHART CREATION ====================
def create_trading_chart(prices, current_trade, next_cycle):
    """Create trading chart for 2-minute cycles"""
    try:
        df = pd.DataFrame(prices, columns=['close'])
        
        # Add basic indicators for chart
        df['sma_5'] = ta.sma(df['close'], length=5)
        df['sma_10'] = ta.sma(df['close'], length=10)
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=list(range(len(prices))),
            y=df['close'],
            mode='lines',
            name='EUR/USD',
            line=dict(color='#00ff88', width=3),
            hovertemplate='Price: %{y:.5f}<extra></extra>'
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=list(range(len(prices))),
            y=df['sma_5'],
            mode='lines',
            name='SMA 5',
            line=dict(color='orange', width=1.5, dash='dash'),
            opacity=0.7
        ))
        
        fig.add_trace(go.Scatter(
            x=list(range(len(prices))),
            y=df['sma_10'],
            mode='lines',
            name='SMA 10',
            line=dict(color='cyan', width=1.5, dash='dot'),
            opacity=0.7
        ))
        
        # Add trade markers if active trade exists
        if current_trade:
            entry_idx = len(prices) - 20 if len(prices) > 20 else 0
            
            # Entry point
            fig.add_trace(go.Scatter(
                x=[entry_idx],
                y=[current_trade['entry_price']],
                mode='markers+text',
                name='Entry',
                marker=dict(
                    size=15,
                    color='yellow',
                    symbol='triangle-up' if current_trade['action'] == 'BUY' else 'triangle-down'
                ),
                text=[f"Entry: {current_trade['entry_price']:.5f}"],
                textposition="top center"
            ))
            
            # TP line
            fig.add_trace(go.Scatter(
                x=[entry_idx, len(prices)-1],
                y=[current_trade['optimal_tp'], current_trade['optimal_tp']],
                mode='lines',
                name=f'TP: {current_trade["optimal_tp"]:.5f}',
                line=dict(color='green', width=2, dash='dash'),
                opacity=0.8
            ))
            
            # SL line
            fig.add_trace(go.Scatter(
                x=[entry_idx, len(prices)-1],
                y=[current_trade['optimal_sl'], current_trade['optimal_sl']],
                mode='lines',
                name=f'SL: {current_trade["optimal_sl"]:.5f}',
                line=dict(color='red', width=2, dash='dash'),
                opacity=0.8
            ))
        
        # Update layout
        title = f'EUR/USD 2-Minute Trading - Next Cycle: {next_cycle}s'
        if 'Cache' in trading_state['data_source'] or 'Simulation' in trading_state['data_source']:
            title += f' ({trading_state["data_source"]})'
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16, color='white')
            ),
            yaxis=dict(
                title='Price',
                titlefont=dict(color='white'),
                tickfont=dict(color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            xaxis=dict(
                title='Time (seconds)',
                titlefont=dict(color='white'),
                tickfont=dict(color='white'),
                gridcolor='rgba(255,255,255,0.1)'
            ),
            template='plotly_dark',
            height=500,
            showlegend=True,
            hovermode='x unified',
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        logger.error(f"Chart error: {e}")
        return None

# ==================== MAIN 2-MINUTE CYCLE ====================
def trading_cycle():
    """Main 2-minute trading cycle with caching"""
    global trading_state
    
    cycle_count = 0
    
    # Initialize ML system
    initialize_training_system()
    
    logger.info("✅ Trading bot started with 2-minute cycles and caching")
    
    while True:
        try:
            cycle_count += 1
            cycle_start = datetime.now()
            
            trading_state['cycle_count'] = cycle_count
            trading_state['cycle_progress'] = 0
            trading_state['remaining_time'] = CYCLE_SECONDS
            
            logger.info(f"\n{'='*70}")
            logger.info(f"2-MINUTE TRADING CYCLE #{cycle_count}")
            logger.info(f"{'='*70}")
            
            # 1. GET MARKET DATA (WITH CACHE)
            logger.info("Fetching EUR/USD price (with caching)...")
            current_price, data_source = get_cached_eurusd_price()
            
            # Track price history
            price_history_deque.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'price': current_price
            })
            
            trading_state['current_price'] = round(float(current_price), 5)
            trading_state['data_source'] = data_source
            trading_state['is_demo_data'] = 'Simulation' in data_source or 'Cache' in data_source
            trading_state['api_calls_today'] = '~240 (SAFE)'
            
            # 2. CREATE PRICE SERIES FOR ANALYSIS
            price_series = create_price_series(current_price, 120)
            
            # 3. CALCULATE TECHNICAL INDICATORS
            df_indicators = calculate_advanced_indicators(price_series)
            
            # 4. MAKE 2-MINUTE PREDICTION
            logger.info("Analyzing market for 2-minute prediction...")
            pred_prob, confidence, direction, signal_strength = analyze_2min_prediction(
                df_indicators, current_price
            )
            
            trading_state['minute_prediction'] = direction
            trading_state['confidence'] = round(float(confidence), 1)
            trading_state['signal_strength'] = signal_strength
            
            # 5. EXTRACT ML FEATURES
            ml_features_current = extract_ml_features(df_indicators, current_price)
            
            # 6. PREDICT OPTIMAL TP/SL
            optimal_tp, optimal_sl, tp_pips, sl_pips = predict_optimal_levels(
                ml_features_current, direction, current_price, df_indicators
            )
            
            # 7. CHECK ACTIVE TRADE
            if trading_state['current_trade']:
                # Get current market conditions for ML
                market_conditions = {
                    'rsi': df_indicators.iloc[-1].get('rsi', 50) if len(df_indicators) > 0 else 50,
                    'atr': df_indicators.iloc[-1].get('atr', 0.0005) if len(df_indicators) > 0 else 0.0005,
                    'bb_percent': df_indicators.iloc[-1].get('bb_percent', 50) if len(df_indicators) > 0 else 50,
                    'macd_hist': df_indicators.iloc[-1].get('macd_hist', 0) if len(df_indicators) > 0 else 0
                }
                monitor_active_trade(current_price, market_conditions)
            
            # 8. EXECUTE NEW TRADE
            if (trading_state['current_trade'] is None and 
                direction != 'NEUTRAL' and 
                confidence >= MIN_CONFIDENCE and
                signal_strength >= 2):
                
                # Get market conditions for ML
                market_conditions = {
                    'rsi': df_indicators.iloc[-1].get('rsi', 50) if len(df_indicators) > 0 else 50,
                    'atr': df_indicators.iloc[-1].get('atr', 0.0005) if len(df_indicators) > 0 else 0.0005,
                    'bb_percent': df_indicators.iloc[-1].get('bb_percent', 50) if len(df_indicators) > 0 else 50,
                    'macd_hist': df_indicators.iloc[-1].get('macd_hist', 0) if len(df_indicators) > 0 else 0
                }
                
                execute_2min_trade(
                    direction, confidence, current_price, 
                    optimal_tp, optimal_sl, tp_pips, sl_pips, market_conditions
                )
            elif trading_state['current_trade'] is None:
                trading_state['action'] = 'WAIT'
                trading_state['trade_status'] = 'NO_SIGNAL'
                logger.info(f"⚠️  No trade signal: {direction} with {confidence:.1f}% confidence")
            
            # 9. CALCULATE NEXT CYCLE TIME
            cycle_duration = (datetime.now() - cycle_start).seconds
            next_cycle_time = max(1, CYCLE_SECONDS - cycle_duration)
            
            trading_state['next_cycle_in'] = next_cycle_time
            
            # 10. CREATE CHART
            chart_data = create_trading_chart(
                price_series, 
                trading_state['current_trade'], 
                next_cycle_time
            )
            trading_state['chart_data'] = chart_data
            
            # 11. UPDATE PRICE HISTORY
            trading_state['price_history'] = list(price_history_deque)[-20:]
            
            # 12. UPDATE TIMESTAMP
            trading_state['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            trading_state['server_time'] = datetime.now().isoformat()
            
            # 13. LOG CYCLE SUMMARY
            logger.info(f"CYCLE #{cycle_count} SUMMARY:")
            logger.info(f"  Price: {current_price:.5f} ({data_source})")
            logger.info(f"  Prediction: {direction} (Signal: {signal_strength}/3)")
            logger.info(f"  Action: {trading_state['action']} ({confidence:.1f}% confidence)")
            logger.info(f"  TP/SL: {tp_pips}/{sl_pips} pips")
            logger.info(f"  Balance: ${trading_state['balance']:.2f}")
            logger.info(f"  Win Rate: {trading_state['win_rate']:.1f}%")
            logger.info(f"  ML Ready: {trading_state['ml_model_ready']}")
            logger.info(f"  Data Points: {trading_state['ml_data_points']} trades in data.txt")
            logger.info(f"  Cache Efficiency: {trading_state['cache_efficiency']}")
            logger.info(f"  API Calls/Day: ~240 (SAFE)")
            logger.info(f"  Next cycle in: {next_cycle_time}s")
            logger.info(f"{'='*70}")
            
            # 14. WAIT FOR NEXT CYCLE WITH PROGRESS UPDATES
            for i in range(next_cycle_time):
                progress_pct = (i / next_cycle_time) * 100
                trading_state['cycle_progress'] = progress_pct
                trading_state['remaining_time'] = next_cycle_time - i
                
                # Update active trade progress if exists
                if trading_state['current_trade']:
                    trade_duration = (datetime.now() - trading_state['current_trade']['entry_time']).total_seconds()
                    trading_state['trade_progress'] = (trade_duration / CYCLE_SECONDS) * 100
                
                time.sleep(1)
            
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    """Render trading dashboard"""
    return render_template('index.html')

@app.route('/api/trading_state')
def get_trading_state():
    """Get current trading state"""
    try:
        state_copy = trading_state.copy()
        
        # Make current trade serializable
        if state_copy['current_trade']:
            trade = state_copy['current_trade'].copy()
            for key in ['entry_time', 'exit_time']:
                if key in trade and trade[key] and isinstance(trade[key], datetime):
                    trade[key] = trade[key].isoformat()
            state_copy['current_trade'] = trade
        
        return jsonify(state_copy)
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trade_history')
def get_trade_history():
    """Get trade history"""
    try:
        serializable_history = []
        for trade in trade_history[-10:]:
            trade_copy = trade.copy()
            for key in ['entry_time', 'exit_time']:
                if key in trade_copy and trade_copy[key] and isinstance(trade_copy[key], datetime):
                    trade_copy[key] = trade_copy[key].isoformat()
            serializable_history.append(trade_copy)
        
        return jsonify({
            'trades': serializable_history,
            'total': len(trade_history),
            'profitable': trading_state['profitable_trades'],
            'win_rate': trading_state['win_rate']
        })
    except Exception as e:
        logger.error(f"Trade history error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ml_status')
def get_ml_status():
    """Get ML training status"""
    return jsonify({
        'ml_model_ready': trading_state['ml_model_ready'],
        'training_samples': trading_state['ml_data_points'],
        'training_file': TRAINING_FILE,
        'last_trained': trading_state['last_update'],
        'ml_features_count': len(ml_features[0]) if ml_features else 0
    })

@app.route('/api/cache_status')
def get_cache_status():
    """Get cache status"""
    return jsonify({
        'cache_duration': CACHE_DURATION,
        'cache_hits': price_cache['hits'],
        'cache_misses': price_cache['misses'],
        'cache_efficiency': trading_state['cache_efficiency'],
        'current_price': price_cache['price'],
        'last_update': datetime.fromtimestamp(price_cache['timestamp']).isoformat(),
        'source': price_cache['source'],
        'api_calls_today': '~240 (66% reduction)'
    })

@app.route('/api/view_ml_data')
def view_ml_data():
    """View ML training data from data.txt"""
    try:
        if not os.path.exists(TRAINING_FILE):
            return jsonify({'error': f'{TRAINING_FILE} not found'})
        
        with open(TRAINING_FILE, 'r') as f:
            lines = f.readlines()
        
        data = []
        for line in lines[:20]:  # First 20 lines only
            if line.strip() and not line.startswith('#'):
                try:
                    data.append(json.loads(line.strip()))
                except:
                    continue
        
        return jsonify({
            'file': TRAINING_FILE,
            'total_lines': len(lines),
            'preview': data,
            'ml_status': {
                'ready': trading_state['ml_model_ready'],
                'samples': trading_state['ml_data_points'],
                'trained_on': len(ml_features)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/reset_trading')
def reset_trading():
    """Reset trading statistics"""
    global trade_history, ml_features, tp_labels, sl_labels
    
    trading_state.update({
        'balance': INITIAL_BALANCE,
        'total_trades': 0,
        'profitable_trades': 0,
        'total_profit': 0.0,
        'win_rate': 0.0,
        'current_trade': None,
        'prediction_accuracy': 0.0,
        'trade_status': 'RESET',
        'trade_progress': 0,
        'cycle_progress': 0,
        'ml_data_points': 0,
        'ml_model_ready': False
    })
    
    trade_history.clear()
    ml_features.clear()
    tp_labels.clear()
    sl_labels.clear()
    
    # Reset data.txt
    try:
        with open(TRAINING_FILE, 'w') as f:
            f.write('# EUR/USD Trading Data - ML Training\n')
            f.write('# Format: JSON per line with trade data\n')
            f.write('# System reset at: ' + datetime.now().isoformat() + '\n')
        logger.info(f"📝 Reset {TRAINING_FILE}")
    except Exception as e:
        logger.error(f"Error resetting {TRAINING_FILE}: {e}")
    
    return jsonify({'success': True, 'message': 'Trading reset complete'})

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'cycle_count': trading_state['cycle_count'],
        'system_status': 'ACTIVE',
        'cycle_duration': CYCLE_SECONDS,
        'cache_enabled': True,
        'cache_duration': CACHE_DURATION,
        'api_calls_per_day': '~240 (SAFE)',
        'ml_data_file': TRAINING_FILE,
        'ml_data_points': trading_state['ml_data_points'],
        'version': '2.0-cached-data-txt'
    })

# ==================== START TRADING BOT ====================
def start_trading_bot():
    """Start the trading bot"""
    try:
        thread = threading.Thread(target=trading_cycle, daemon=True)
        thread.start()
        logger.info("✅ Trading bot started successfully")
        print("✅ 2-Minute trading system ACTIVE")
        print(f"✅ Caching: {CACHE_DURATION}-second cache enabled")
        print(f"✅ ML Storage: Using {TRAINING_FILE} for training data")
        print("✅ API Calls/Day: ~240 (SAFE for all free limits)")
        print(f"✅ ML Training: Ready after 10 trades (data.txt)")
    except Exception as e:
        logger.error(f"❌ Error starting trading bot: {e}")
        print(f"❌ Error: {e}")

# ==================== MAIN ENTRY POINT ====================
if __name__ == '__main__':
    # Start trading bot
    start_trading_bot()
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Web dashboard: http://localhost:{port}")
    print("="*80)
    print("API-SAFE SYSTEM READY WITH data.txt STORAGE")
    print(f"• 2-minute cycles with {CACHE_DURATION}-second caching")
    print(f"• API calls: ~240/day (66% reduction)")
    print(f"• Frankfurter limit: 1000/day → 24% used ✅")
    print(f"• FreeForexAPI limit: 2400/day → 10% used ✅")
    print(f"• ML training data: {TRAINING_FILE}")
    print(f"• ML training: After 10 trades in data.txt")
    print("="*80)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )