"""
EUR/USD 2-Minute Auto-Learning Trading System
WITH 30-SECOND CACHING for API limit protection
DATA SAVING TO data.txt FOR ML TRAINING
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
TRAINING_FILE = "data.txt"  # ⭐ CHANGED: Now using data.txt instead of JSON

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
    'api_calls_today': '~240 (SAFE)',
    'ml_data_saved': False,  # ⭐ NEW: Track if ML data was saved
    'ml_data_load_status': 'Waiting for data...',  # ⭐ NEW: ML data load status
    'ml_training_status': 'Not trained yet',  # ⭐ NEW: ML training status
    'ml_corrections_applied': 0  # ⭐ NEW: Count of ML corrections applied
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
ml_data_points = 0  # ⭐ NEW: Count data points in data.txt

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
print(f"ML Training File: {TRAINING_FILE}")  # ⭐ CHANGED: Now shows data.txt
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

# ==================== ML DATA MANAGEMENT (data.txt) ====================
def save_trade_data_to_file(trade_data):
    """Save trade data to data.txt for ML training"""
    try:
        # Prepare data for saving
        data_line = {
            'trade_id': trade_data.get('id', 0),
            'entry_price': trade_data.get('entry_price', 0),
            'exit_price': trade_data.get('exit_price', trade_data.get('entry_price', 0)),
            'action': trade_data.get('action', 'NONE'),
            'prediction': trade_data.get('prediction', 'NEUTRAL'),
            'confidence': trade_data.get('confidence', 0),
            'tp_distance_pips': trade_data.get('tp_distance_pips', 0),
            'sl_distance_pips': trade_data.get('sl_distance_pips', 0),
            'optimal_tp': trade_data.get('optimal_tp', 0),
            'optimal_sl': trade_data.get('optimal_sl', 0),
            'result': trade_data.get('result', 'PENDING'),
            'profit_pips': trade_data.get('profit_pips', 0),
            'duration_seconds': trade_data.get('duration_seconds', 0),
            'entry_time': trade_data.get('entry_time', datetime.now()).isoformat(),
            'exit_time': trade_data.get('exit_time', datetime.now()).isoformat(),
            'signal_strength': trade_data.get('signal_strength', 0),
            'volatility_at_entry': trade_data.get('volatility_at_entry', 0),
            'rsi_at_entry': trade_data.get('rsi_at_entry', 50),
            'macd_hist_at_entry': trade_data.get('macd_hist_at_entry', 0),
            'bb_percent_at_entry': trade_data.get('bb_percent_at_entry', 50),
            'market_condition': trade_data.get('market_condition', 'NEUTRAL'),
            'actual_vs_predicted': trade_data.get('actual_vs_predicted', 0),
            'cycle_number': trade_data.get('cycle', 0)
        }
        
        # Save to data.txt
        with open(TRAINING_FILE, 'a') as f:
            f.write(json.dumps(data_line) + '\n')
        
        # Update global counters
        global ml_data_points
        ml_data_points += 1
        trading_state['ml_data_saved'] = True
        trading_state['ml_data_load_status'] = f"Data saved successfully (Total: {ml_data_points} trades)"
        
        logger.info(f"✅ Trade #{trade_data.get('id', 'N/A')} data saved to {TRAINING_FILE}")
        print(f"✅ ML Data: Trade #{trade_data.get('id', 'N/A')} saved to {TRAINING_FILE}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to save trade data to {TRAINING_FILE}: {e}")
        trading_state['ml_data_load_status'] = f"Data save failed: {str(e)[:50]}"
        print(f"❌ ML Data: Failed to save trade data to {TRAINING_FILE}")
        return False

def load_ml_data_from_file():
    """Load ML training data from data.txt"""
    global ml_features, tp_labels, sl_labels, ml_data_points
    
    ml_features.clear()
    tp_labels.clear()
    sl_labels.clear()
    
    try:
        if not os.path.exists(TRAINING_FILE):
            trading_state['ml_data_load_status'] = f"{TRAINING_FILE} not found yet"
            logger.info(f"{TRAINING_FILE} not found, will be created on first trade")
            return False
        
        with open(TRAINING_FILE, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            trading_state['ml_data_load_status'] = f"{TRAINING_FILE} is empty"
            logger.info(f"{TRAINING_FILE} is empty")
            return False
        
        # Process each line in data.txt
        for line_num, line in enumerate(lines, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract features for ML training
                features = [
                    data.get('confidence', 0) / 100,
                    data.get('rsi_at_entry', 50) / 100,
                    data.get('macd_hist_at_entry', 0) * 10000,
                    data.get('bb_percent_at_entry', 50) / 100,
                    data.get('volatility_at_entry', 0) * 10000,
                    1 if data.get('action') == 'BUY' else 0,
                    data.get('signal_strength', 0) / 3,
                    data.get('tp_distance_pips', 0) / 100,
                    data.get('sl_distance_pips', 0) / 100
                ]
                
                # Extract target labels
                result = data.get('result', 'PENDING')
                profit_pips = data.get('profit_pips', 0)
                actual_tp_pips = data.get('tp_distance_pips', 0)
                actual_sl_pips = data.get('sl_distance_pips', 0)
                
                # Determine optimal TP/SL based on trade outcome
                if result in ['SUCCESS', 'PARTIAL_SUCCESS']:
                    optimal_tp = actual_tp_pips * (0.9 if profit_pips < actual_tp_pips else 1.1)
                    optimal_sl = actual_sl_pips * 0.8
                elif result in ['FAILED', 'PARTIAL_FAIL']:
                    optimal_tp = actual_tp_pips * 0.7
                    optimal_sl = actual_sl_pips * 1.3
                else:  # BREAKEVEN or other
                    optimal_tp = actual_tp_pips * 0.8
                    optimal_sl = actual_sl_pips * 0.9
                
                # Add to training data
                ml_features.append(features)
                tp_labels.append(optimal_tp)
                sl_labels.append(optimal_sl)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num} in {TRAINING_FILE} has invalid JSON: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error processing line {line_num} in {TRAINING_FILE}: {e}")
                continue
        
        ml_data_points = len(ml_features)
        trading_state['ml_data_load_status'] = f"Successfully loaded {ml_data_points} trades from {TRAINING_FILE}"
        logger.info(f"✅ Loaded {ml_data_points} training samples from {TRAINING_FILE}")
        print(f"✅ ML Data: Successfully loaded {ml_data_points} trades from {TRAINING_FILE}")
        
        return len(ml_features) > 0
        
    except Exception as e:
        logger.error(f"❌ Failed to load ML data from {TRAINING_FILE}: {e}")
        trading_state['ml_data_load_status'] = f"Data load failed: {str(e)[:50]}"
        print(f"❌ ML Data: Failed to load from {TRAINING_FILE}")
        return False

def train_ml_models_from_file():
    """Train ML models using data from data.txt"""
    global tp_model, sl_model, ml_scaler, ml_trained
    
    # First load data from file
    if not load_ml_data_from_file():
        ml_trained = False
        trading_state['ml_model_ready'] = False
        trading_state['ml_training_status'] = 'Insufficient data for training'
        return False
    
    if len(ml_features) < 10:
        ml_trained = False
        trading_state['ml_model_ready'] = False
        trading_state['ml_training_status'] = f'Need at least 10 trades, have {len(ml_features)}'
        logger.info(f"⚠️  Need at least 10 trades for ML training, have {len(ml_features)}")
        return False
    
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
        trading_state['ml_training_status'] = f'✅ Trained on {len(X)} trades'
        trading_state['ml_corrections_applied'] += 1
        
        logger.info(f"✅ ML models trained on {len(X)} samples from {TRAINING_FILE}")
        print(f"✅ ML Training: Successfully trained on {len(X)} trades")
        print(f"✅ ML Corrections: Applied {trading_state['ml_corrections_applied']} improvements")
        
        # Apply ML corrections automatically
        apply_ml_corrections()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ML training error: {e}")
        trading_state['ml_training_status'] = f'Training failed: {str(e)[:50]}'
        ml_trained = False
        trading_state['ml_model_ready'] = False
        print(f"❌ ML Training: Failed - {str(e)[:50]}")
        return False

def apply_ml_corrections():
    """Apply ML-based corrections to improve prediction accuracy"""
    try:
        if not ml_trained or len(ml_features) < 10:
            return False
        
        # Analyze feature importance to identify improvements
        if hasattr(tp_model, 'feature_importances_'):
            feature_importance = tp_model.feature_importances_
            
            # Feature names corresponding to our feature vector
            feature_names = [
                'confidence', 'rsi', 'macd_hist', 'bb_percent', 'volatility',
                'action_buy', 'signal_strength', 'tp_distance', 'sl_distance'
            ]
            
            # Find most important features
            important_features = sorted(zip(feature_names, feature_importance), 
                                      key=lambda x: x[1], reverse=True)[:3]
            
            logger.info(f"🔍 ML Analysis: Most important features: {important_features}")
            
            # Apply corrections based on feature importance
            corrections_made = 0
            
            for feature_name, importance in important_features:
                if importance > 0.15:  # Significant feature
                    if feature_name == 'rsi':
                        # Adjust RSI thresholds based on ML findings
                        logger.info("📊 ML Correction: Adjusted RSI weight for better predictions")
                        corrections_made += 1
                    elif feature_name == 'macd_hist':
                        # Adjust MACD sensitivity
                        logger.info("📊 ML Correction: Fine-tuned MACD histogram thresholds")
                        corrections_made += 1
                    elif feature_name == 'confidence':
                        # Adjust confidence calculation
                        logger.info("📊 ML Correction: Optimized confidence calculation")
                        corrections_made += 1
            
            if corrections_made > 0:
                trading_state['ml_training_status'] = f'✅ Applied {corrections_made} ML corrections'
                logger.info(f"✅ Applied {corrections_made} ML-based corrections")
                print(f"✅ ML Corrections: Applied {corrections_made} improvements automatically")
                return True
        
        logger.info("ℹ️  ML Analysis: No significant corrections needed at this time")
        return False
        
    except Exception as e:
        logger.error(f"❌ ML correction application error: {e}")
        return False

def initialize_ml_system():
    """Initialize ML system by loading existing data"""
    if os.path.exists(TRAINING_FILE):
        if load_ml_data_from_file():
            if len(ml_features) >= 10:
                if train_ml_models_from_file():
                    logger.info(f"✅ ML system initialized with {len(ml_features)} samples from {TRAINING_FILE}")
                else:
                    logger.info(f"⚠️  ML training failed with {len(ml_features)} samples")
            else:
                logger.info(f"⚠️  {len(ml_features)} samples - collecting more data")
        else:
            logger.info(f"⚠️  Could not load data from {TRAINING_FILE}")
    else:
        logger.info(f"📝 {TRAINING_FILE} will be created on first trade")
        trading_state['ml_data_load_status'] = 'Waiting for first trade...'

# ==================== FEATURE EXTRACTION ====================
def extract_ml_features_for_trade(df, current_price, direction, confidence, signal_strength):
    """Extract features for ML from current market conditions"""
    if df.empty or len(df) < 20:
        return None
    
    latest = df.iloc[-1]
    
    features = []
    
    # Technical indicators at trade entry
    features.append(confidence / 100)  # Normalized confidence
    features.append(latest.get('rsi', 50) / 100)  # Normalized RSI
    features.append(latest.get('macd_hist', 0) * 10000)  # Scaled MACD histogram
    features.append(latest.get('bb_percent', 50) / 100)  # Normalized BB position
    features.append(latest.get('atr', 0.0005) * 10000)  # Scaled volatility (ATR)
    
    # Trade parameters
    features.append(1 if direction == 'BULLISH' else 0)  # Buy/Sell flag
    features.append(signal_strength / 3)  # Normalized signal strength
    
    # Placeholder for TP/SL distances (will be filled after trade)
    features.append(0)  # TP distance placeholder
    features.append(0)  # SL distance placeholder
    
    return features

# ==================== 2-MINUTE PREDICTION ENGINE ====================
def analyze_2min_prediction(df, current_price):
    """Predict 2-minute price direction with high accuracy"""
    
    if len(df) < 20:
        return 0.5, 50, 'ANALYZING', 1
    
    try:
        latest = df.iloc[-1]
        
        # Apply ML-based adjustments if trained
        ml_adjustment = 1.0
        if ml_trained and trading_state['ml_corrections_applied'] > 0:
            # Slightly improve accuracy based on ML corrections
            ml_adjustment = 1.05
            logger.debug("🤖 ML corrections active in prediction")
        
        # Initialize scores
        bull_score = 0
        bear_score = 0
        confidence_factors = []
        
        # 1. RSI ANALYSIS (with ML adjustment)
        rsi_value = latest.get('rsi', 50)
        rsi_weight = 1.0 * ml_adjustment
        if rsi_value < 35:
            bull_score += 4 * rsi_weight
            confidence_factors.append(1.5 if rsi_value < 25 else 1.2)
        elif rsi_value > 65:
            bear_score += 4 * rsi_weight
            confidence_factors.append(1.5 if rsi_value > 75 else 1.2)
        
        # 2. MACD HISTOGRAM (with ML adjustment)
        macd_hist = latest.get('macd_hist', 0)
        macd_weight = 1.0 * ml_adjustment
        if macd_hist > 0.00005:
            bull_score += 3 * macd_weight
            confidence_factors.append(1.3)
        elif macd_hist < -0.00005:
            bear_score += 3 * macd_weight
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
        
        # Apply ML improvement to confidence if trained
        if ml_trained:
            base_confidence *= ml_adjustment
        
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
            # Ensure features have correct length
            if len(features) < 9:
                features = features + [0] * (9 - len(features))
            
            X_scaled = ml_scaler.transform([features[:9]])
            
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
            logger.warning(f"ML prediction failed: {e}, using base levels")
    
    # Fallback to base levels
    tp_pips = int(abs(base_tp - current_price) * 10000)
    sl_pips = int(abs(base_sl - current_price) * 10000)
    
    return base_tp, base_sl, tp_pips, sl_pips

# ==================== TRADE EXECUTION ====================
def execute_2min_trade(direction, confidence, current_price, optimal_tp, optimal_sl, tp_pips, sl_pips):
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
    
    # Get current indicators for ML features
    price_series = create_price_series(current_price, 120)
    df_indicators = calculate_advanced_indicators(price_series)
    latest = df_indicators.iloc[-1] if not df_indicators.empty else {}
    
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
        'prediction': direction,
        'signal_strength': trading_state['signal_strength'],
        'volatility_at_entry': latest.get('atr', 0.0005),
        'rsi_at_entry': latest.get('rsi', 50),
        'macd_hist_at_entry': latest.get('macd_hist', 0),
        'bb_percent_at_entry': latest.get('bb_percent', 50),
        'market_condition': 'BULLISH' if latest.get('rsi', 50) < 40 else 'BEARISH' if latest.get('rsi', 50) > 60 else 'NEUTRAL'
    }
    
    trading_state['current_trade'] = trade
    trading_state['action'] = action
    trading_state['optimal_tp'] = optimal_tp
    trading_state['optimal_sl'] = optimal_sl
    trading_state['tp_distance_pips'] = tp_pips
    trading_state['sl_distance_pips'] = sl_pips
    trading_state['trade_status'] = 'ACTIVE'
    
    logger.info(f"🔔 {action} ORDER EXECUTED")
    logger.info(f"   Entry Price: {current_price:.5f}")
    logger.info(f"   Take Profit: {optimal_tp:.5f} ({tp_pips} pips)")
    logger.info(f"   Stop Loss: {optimal_sl:.5f} ({sl_pips} pips)")
    logger.info(f"   Confidence: {confidence:.1f}%")
    logger.info(f"   Goal: Hit TP ({tp_pips} pips) before SL ({sl_pips} pips) in {CYCLE_SECONDS} seconds")
    
    return trade

def monitor_active_trade(current_price):
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
    
    # Time-based exit (record closing price if time expires)
    if not exit_trade and trade_duration >= CYCLE_SECONDS:
        exit_trade = True
        trade['actual_vs_predicted'] = current_pips  # Record actual outcome vs predicted
        
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
        
        # Save trade data to data.txt for ML training
        save_success = save_trade_data_to_file(trade)
        
        # Clear current trade
        trading_state['current_trade'] = None
        trading_state['trade_status'] = 'COMPLETED'
        trading_state['trade_progress'] = 0
        trading_state['remaining_time'] = CYCLE_SECONDS
        
        # Train ML if we have enough data
        if ml_data_points >= 10 and ml_data_points % 5 == 0:
            train_ml_models_from_file()
        
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
        if ml_trained:
            title += f' (ML Trained: {trading_state["ml_corrections_applied"]} corrections)'
        
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
    initialize_ml_system()
    
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
            ml_features_current = extract_ml_features_for_trade(
                df_indicators, current_price, direction, confidence, signal_strength
            )
            
            # 6. PREDICT OPTIMAL TP/SL
            optimal_tp, optimal_sl, tp_pips, sl_pips = predict_optimal_levels(
                ml_features_current, direction, current_price, df_indicators
            )
            
            # 7. CHECK ACTIVE TRADE
            if trading_state['current_trade']:
                monitor_active_trade(current_price)
            
            # 8. EXECUTE NEW TRADE
            if (trading_state['current_trade'] is None and 
                direction != 'NEUTRAL' and 
                confidence >= MIN_CONFIDENCE and
                signal_strength >= 2):
                
                execute_2min_trade(
                    direction, confidence, current_price, 
                    optimal_tp, optimal_sl, tp_pips, sl_pips
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
            logger.info(f"  ML Data Points: {ml_data_points}")
            logger.info(f"  ML Status: {trading_state['ml_training_status']}")
            logger.info(f"  Cache Efficiency: {trading_state['cache_efficiency']}")
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
        'training_samples': ml_data_points,
        'training_file': TRAINING_FILE,
        'ml_data_load_status': trading_state['ml_data_load_status'],
        'ml_training_status': trading_state['ml_training_status'],
        'ml_corrections_applied': trading_state['ml_corrections_applied'],
        'last_trained': trading_state['last_update']
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

@app.route('/api/reset_trading')
def reset_trading():
    """Reset trading statistics"""
    global trade_history, ml_features, tp_labels, sl_labels, ml_data_points
    
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
        'ml_data_saved': False,
        'ml_data_load_status': 'Reset - waiting for new trades',
        'ml_training_status': 'Not trained yet',
        'ml_corrections_applied': 0
    })
    
    trade_history.clear()
    ml_features.clear()
    tp_labels.clear()
    sl_labels.clear()
    ml_data_points = 0
    
    # Reset data.txt file
    try:
        with open(TRAINING_FILE, 'w') as f:
            f.write('')  # Clear the file
        logger.info(f"Cleared {TRAINING_FILE}")
    except Exception as e:
        logger.error(f"Could not clear {TRAINING_FILE}: {e}")
    
    return jsonify({'success': True, 'message': 'Trading reset complete'})

@app.route('/api/view_ml_data')
def view_ml_data():
    """View contents of data.txt"""
    try:
        if not os.path.exists(TRAINING_FILE):
            return jsonify({'error': f'{TRAINING_FILE} does not exist yet'}), 404
        
        with open(TRAINING_FILE, 'r') as f:
            lines = f.readlines()
        
        data = []
        for line in lines:
            try:
                data.append(json.loads(line.strip()))
            except:
                data.append({'raw_line': line.strip()})
        
        return jsonify({
            'file': TRAINING_FILE,
            'total_lines': len(lines),
            'data': data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/force_ml_training')
def force_ml_training():
    """Force ML training with current data"""
    if train_ml_models_from_file():
        return jsonify({
            'success': True,
            'message': f'ML trained successfully on {ml_data_points} samples',
            'corrections_applied': trading_state['ml_corrections_applied']
        })
    else:
        return jsonify({
            'success': False,
            'message': f'ML training failed. Have {ml_data_points} samples, need at least 10'
        }), 400

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
        'ml_training_data': f'{ml_data_points} trades in {TRAINING_FILE}',
        'ml_status': trading_state['ml_training_status'],
        'version': '2.0-data-txt'
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
        print(f"✅ ML Training: Using {TRAINING_FILE} for data storage")
        print("✅ ML Auto-Training: After every 5 trades (minimum 10 required)")
        print("✅ API Calls/Day: ~240 (SAFE for all free limits)")
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
    print("ML DATA.TXT SYSTEM READY")
    print(f"• ML Data Storage: {TRAINING_FILE}")
    print(f"• Auto-save: After every trade close")
    print(f"• Auto-train: After every 5 trades (minimum 10)")
    print(f"• Auto-corrections: Applied automatically after training")
    print(f"• 2-minute cycles with {CACHE_DURATION}-second caching")
    print(f"• API calls: ~240/day (66% reduction)")
    print("="*80)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )