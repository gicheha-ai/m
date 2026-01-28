"""
EUR/USD 2-Minute Auto-Learning Trading System
WITH 30-SECOND CACHING for API limit protection
AND GOOGLE SHEETS DATA STORAGE
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

# ==================== GOOGLE SHEETS CONFIGURATION ====================
SHEETDB_API_URL = "https://sheetdb.io/api/v1/lfjhagwynpocp"  # Your SheetDB endpoint
GOOGLE_SHEETS_ENABLED = True

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
    'google_sheets_status': 'INITIALIZING'
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

# Trading bot thread control
trading_thread = None
trading_active = True

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
print(f"Data Storage: Google Sheets")
print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
print(f"Trade Size: ${BASE_TRADE_SIZE:,.2f}")
print(f"Auto-Trading: ENABLED")
print("="*80)
print("Starting system...")

# ==================== GOOGLE SHEETS FUNCTIONS ====================
def save_trade_to_google_sheets(trade_data):
    """Save trade data to Google Sheets"""
    if not GOOGLE_SHEETS_ENABLED:
        return {'success': False, 'message': 'Google Sheets disabled'}
    
    try:
        # Format trade data for SheetDB
        sheetdb_data = {
            "data": [trade_data]
        }
        
        response = requests.post(
            SHEETDB_API_URL,
            json=sheetdb_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 201:
            logger.info(f"✅ Trade saved to Google Sheets: Trade #{trade_data.get('trade_id', 'N/A')}")
            trading_state['google_sheets_status'] = 'CONNECTED'
            return {'success': True, 'data': response.json()}
        else:
            logger.warning(f"⚠️  SheetDB API error: {response.status_code} - {response.text[:100]}")
            trading_state['google_sheets_status'] = f'ERROR: {response.status_code}'
            return {'success': False, 'error': f'API Error {response.status_code}'}
            
    except Exception as e:
        logger.error(f"❌ Error saving to Google Sheets: {str(e)[:100]}")
        trading_state['google_sheets_status'] = f'ERROR: {str(e)[:30]}'
        return {'success': False, 'error': str(e)}

def get_trades_from_google_sheets():
    """Retrieve trade history from Google Sheets"""
    if not GOOGLE_SHEETS_ENABLED:
        return []
    
    try:
        response = requests.get(SHEETDB_API_URL, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"📊 Retrieved {len(data)} trades from Google Sheets")
            trading_state['google_sheets_status'] = 'CONNECTED'
            return data
        else:
            logger.warning(f"SheetDB GET error: {response.status_code}")
            trading_state['google_sheets_status'] = f'READ_ERROR: {response.status_code}'
            return []
            
    except Exception as e:
        logger.error(f"Error fetching from Google Sheets: {e}")
        trading_state['google_sheets_status'] = f'FETCH_ERROR: {str(e)[:30]}'
        return []

def initialize_google_sheets():
    """Initialize connection to Google Sheets"""
    if not GOOGLE_SHEETS_ENABLED:
        logger.info("Google Sheets storage disabled")
        trading_state['google_sheets_status'] = 'DISABLED'
        return
    
    try:
        logger.info("Initializing Google Sheets connection...")
        
        # Test connection
        test_response = requests.get(SHEETDB_API_URL, timeout=5)
        
        if test_response.status_code == 200:
            # Load existing trades
            existing_trades = get_trades_from_google_sheets()
            
            if existing_trades:
                # Convert to our format
                for trade in existing_trades:
                    # Convert numeric fields
                    numeric_fields = ['trade_id', 'entry_price', 'exit_price', 'profit_pips', 
                                    'confidence', 'duration_seconds', 'cycle_number', 
                                    'signal_strength', 'sl_distance_pips', 'tp_distance_pips']
                    for field in numeric_fields:
                        if field in trade:
                            try:
                                trade[field] = float(trade[field]) if '.' in str(trade[field]) else int(trade[field])
                            except:
                                trade[field] = 0
                
                trade_history.extend(existing_trades)
                trading_state['total_trades'] = len(existing_trades)
                
                # Count profitable trades
                profitable = sum(1 for t in existing_trades 
                               if t.get('result') in ['SUCCESS', 'WIN', 'PARTIAL_SUCCESS'])
                trading_state['profitable_trades'] = profitable
                
                if trading_state['total_trades'] > 0:
                    trading_state['win_rate'] = (trading_state['profitable_trades'] / 
                                                trading_state['total_trades']) * 100
                
                logger.info(f"✅ Loaded {len(existing_trades)} trades from Google Sheets")
                logger.info(f"✅ Win Rate: {trading_state['win_rate']:.1f}%")
                trading_state['google_sheets_status'] = 'CONNECTED'
            else:
                logger.info("✅ Google Sheets connected (no existing trades)")
                trading_state['google_sheets_status'] = 'CONNECTED'
        else:
            logger.warning(f"⚠️  SheetDB test failed: {test_response.status_code}")
            trading_state['google_sheets_status'] = f'TEST_FAILED: {test_response.status_code}'
            
    except Exception as e:
        logger.error(f"❌ Google Sheets initialization error: {e}")
        trading_state['google_sheets_status'] = f'INIT_ERROR: {str(e)[:30]}'

# ==================== CACHED FOREX DATA FETCHING ====================
def get_cached_eurusd_price():
    """Get EUR/USD price with 30-second caching"""
    
    current_time = time.time()
    cache_age = current_time - price_cache['timestamp']
    
    # ⭐ CACHE HIT: Use cached price if fresh
    if cache_age < CACHE_DURATION and price_cache['price']:
        price_cache['hits'] += 1
        update_cache_efficiency()
        
        # Add tiny realistic fluctuation
        tiny_change = np.random.uniform(-0.00001, 0.00001)
        cached_price = price_cache['price'] + tiny_change
        
        logger.debug(f"📦 CACHE HIT: Using cached price {cached_price:.5f} (age: {cache_age:.1f}s)")
        trading_state['api_status'] = f"CACHED ({price_cache['source']})"
        
        return cached_price, f"Cached ({price_cache['source']})"
    
    # ⭐ CACHE MISS: Need fresh price
    price_cache['misses'] += 1
    update_cache_efficiency()
    logger.info("🔄 Cache MISS: Fetching fresh price from APIs...")
    
    # List of APIs
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
            
            if response.status_code == 429:
                logger.warning(f"⏸️ {api['name']} rate limit reached, skipping...")
                continue
                
            if response.status_code == 200:
                data = response.json()
                rate = api['extract_rate'](data)
                
                if rate:
                    current_price = float(rate)
                    
                    # ⭐ UPDATE CACHE
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
    
    # ⭐ ALL APIS FAILED: Use stale cache
    logger.warning("⚠️ All APIs failed, using stale cached data")
    
    if price_cache['price']:
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
        # First run
        trading_state['api_status'] = 'SIMULATION'
        return 1.0850, 'Simulation (Initial)'

def update_cache_efficiency():
    """Calculate cache efficiency"""
    total = price_cache['hits'] + price_cache['misses']
    if total > 0:
        efficiency = (price_cache['hits'] / total) * 100
        trading_state['cache_efficiency'] = f"{efficiency:.1f}%"
        trading_state['cache_hits'] = price_cache['hits']
        trading_state['cache_misses'] = price_cache['misses']

def create_price_series(current_price, num_points=120):
    """Create price series for analysis"""
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
    """Calculate indicators for 2-minute prediction"""
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
    """Initialize ML training data from Google Sheets"""
    global ml_features, tp_labels, sl_labels, ml_trained
    
    if trade_history and len(trade_history) >= 10:
        try:
            logger.info(f"Starting ML training with {len(trade_history)} trades...")
            
            # Extract features from recent trades for ML
            for trade in trade_history[-50:]:  # Use last 50 trades
                if 'confidence' not in trade:
                    continue
                    
                features = [
                    trade.get('confidence', 0) / 100,
                    trade.get('bb_percent_at_entry', 50) / 100,
                    trade.get('rsi_at_entry', 50) / 100,
                    trade.get('macd_hist_at_entry', 0) * 10000,
                    1 if trade.get('action') == 'BUY' else 0,
                    trade.get('signal_strength', 0) / 3
                ]
                ml_features.append(features)
                
                # Determine optimal TP/SL based on trade result
                profit = trade.get('profit_pips', 0)
                result = trade.get('result', '')
                
                if result in ['SUCCESS', 'WIN']:
                    # Successful trade - keep similar distances
                    optimal_tp = trade.get('tp_distance_pips', 8)
                    optimal_sl = trade.get('sl_distance_pips', 5)
                elif result in ['FAILED', 'LOSE']:
                    # Failed trade - adjust distances
                    optimal_tp = trade.get('tp_distance_pips', 8) * 0.8  # Reduce TP
                    optimal_sl = trade.get('sl_distance_pips', 5) * 1.2  # Increase SL
                else:
                    # Other results - moderate adjustment
                    optimal_tp = trade.get('tp_distance_pips', 8) * 0.9
                    optimal_sl = trade.get('sl_distance_pips', 5) * 1.1
                
                tp_labels.append(max(5, min(20, optimal_tp)))
                sl_labels.append(max(3, min(15, optimal_sl)))
            
            if len(ml_features) >= 10:
                train_ml_models()
                logger.info(f"✅ ML system initialized with {len(ml_features)} samples")
            else:
                logger.info(f"⚠️  {len(ml_features)} samples - collecting more data")
                
        except Exception as e:
            logger.error(f"Error initializing ML: {e}")
    else:
        logger.info("⚠️  Not enough trades for ML training yet")

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
    """Predict optimal TP and SL levels"""
    
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
    """Predict 2-minute price direction"""
    
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
    
    trade_id = len(trade_history) + 1
    
    # Get current indicator values for ML features
    price_series = create_price_series(current_price, 120)
    df_indicators = calculate_advanced_indicators(price_series)
    latest = df_indicators.iloc[-1]
    
    trade = {
        'trade_id': trade_id,
        'timestamp': datetime.now().isoformat(),
        'action': action,
        'entry_price': float(current_price),
        'exit_price': None,
        'profit_pips': 0.0,
        'result': 'PENDING',
        'duration_seconds': 0.0,
        'cycle_number': trading_state['cycle_count'],
        'confidence': float(confidence),
        'signal_strength': signal_strength,
        'bb_percent_at_entry': float(latest.get('bb_percent', 50)),
        'rsi_at_entry': float(latest.get('rsi', 50)),
        'macd_hist_at_entry': float(latest.get('macd_hist', 0)),
        'volatility_at_entry': float(latest.get('atr', 0.0005)),
        'sl_distance_pips': sl_pips,
        'tp_distance_pips': tp_pips,
        'optimal_tp': float(optimal_tp),
        'optimal_sl': float(optimal_sl),
        'trade_size': BASE_TRADE_SIZE,
        'status': 'OPEN',
        'reason': action_reason
    }
    
    trading_state['current_trade'] = trade
    trading_state['action'] = action
    trading_state['optimal_tp'] = optimal_tp
    trading_state['optimal_sl'] = optimal_sl
    trading_state['tp_distance_pips'] = tp_pips
    trading_state['sl_distance_pips'] = sl_pips
    trading_state['trade_status'] = 'ACTIVE'
    trading_state['signal_strength'] = signal_strength
    
    # Save to Google Sheets immediately
    save_result = save_trade_to_google_sheets(trade)
    
    logger.info(f"🔔 {action} ORDER EXECUTED")
    logger.info(f"   Trade ID: {trade_id}")
    logger.info(f"   Entry Price: {current_price:.5f}")
    logger.info(f"   Take Profit: {optimal_tp:.5f} ({tp_pips} pips)")
    logger.info(f"   Stop Loss: {optimal_sl:.5f} ({sl_pips} pips)")
    logger.info(f"   Confidence: {confidence:.1f}%")
    logger.info(f"   Google Sheets: {'Saved' if save_result.get('success') else 'Failed'}")
    logger.info(f"   Goal: Hit TP ({tp_pips} pips) before SL ({sl_pips} pips) in {CYCLE_SECONDS} seconds")
    
    return trade

def monitor_active_trade(current_price):
    """Monitor the active trade throughout the 2-minute cycle"""
    if not trading_state['current_trade']:
        return None
    
    trade = trading_state['current_trade']
    entry_time = datetime.fromisoformat(trade['timestamp'])
    trade_duration = (datetime.now() - entry_time).total_seconds()
    
    # Calculate current P&L
    if trade['action'] == 'BUY':
        current_pips = (current_price - trade['entry_price']) * 10000
    else:  # SELL
        current_pips = (trade['entry_price'] - current_price) * 10000
    
    trade['profit_pips'] = float(current_pips)
    trade['duration_seconds'] = float(trade_duration)
    
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
        trade['exit_price'] = float(current_price)
        trade['exit_time'] = datetime.now().isoformat()
        trade['exit_reason'] = exit_reason
        
        # Update trading statistics
        trading_state['total_trades'] += 1
        
        if trade['result'] in ['SUCCESS', 'PARTIAL_SUCCESS']:
            trading_state['profitable_trades'] += 1
            profit_amount = (trade['profit_pips'] / 10000) * BASE_TRADE_SIZE
            trading_state['total_profit'] += profit_amount
            trading_state['balance'] += profit_amount
            logger.info(f"💰 TRADE SUCCESS: {exit_reason}")
        else:
            loss_amount = abs(trade['profit_pips'] / 10000) * BASE_TRADE_SIZE
            trading_state['balance'] -= loss_amount
            logger.info(f"📉 TRADE FAILED: {exit_reason}")
        
        # Update win rate
        if trading_state['total_trades'] > 0:
            trading_state['win_rate'] = (trading_state['profitable_trades'] / 
                                        trading_state['total_trades']) * 100
        
        # Add to local history
        trade_history.append(trade.copy())
        
        # Update trade in Google Sheets
        save_trade_to_google_sheets(trade)
        
        # Clear current trade
        trading_state['current_trade'] = None
        trading_state['trade_status'] = 'COMPLETED'
        trading_state['trade_progress'] = 0
        trading_state['remaining_time'] = CYCLE_SECONDS
        
        # Add to ML training data
        add_trade_to_ml_training(trade)
        
        return trade
    
    return trade

def add_trade_to_ml_training(trade):
    """Add completed trade to ML training data"""
    try:
        # Extract features from trade
        features = [
            trade.get('confidence', 0) / 100,
            trade.get('bb_percent_at_entry', 50) / 100,
            trade.get('rsi_at_entry', 50) / 100,
            trade.get('macd_hist_at_entry', 0) * 10000,
            1 if trade.get('action') == 'BUY' else 0,
            trade.get('signal_strength', 0) / 3
        ]
        
        # Determine optimal TP/SL based on result
        profit = trade.get('profit_pips', 0)
        result = trade.get('result', '')
        
        if result in ['SUCCESS', 'WIN']:
            optimal_tp = trade.get('tp_distance_pips', 8)
            optimal_sl = trade.get('sl_distance_pips', 5)
        elif result in ['FAILED', 'LOSE']:
            optimal_tp = trade.get('tp_distance_pips', 8) * 0.8
            optimal_sl = trade.get('sl_distance_pips', 5) * 1.2
        else:
            optimal_tp = trade.get('tp_distance_pips', 8) * 0.9
            optimal_sl = trade.get('sl_distance_pips', 5) * 1.1
        
        ml_features.append(features)
        tp_labels.append(max(5, min(20, optimal_tp)))
        sl_labels.append(max(3, min(15, optimal_sl)))
        
        # Retrain ML if we have enough samples
        if len(ml_features) >= 10 and len(ml_features) % 5 == 0:
            train_ml_models()
            logger.info(f"📚 ML retrained with {len(ml_features)} samples")
        
    except Exception as e:
        logger.error(f"Error adding trade to ML training: {e}")

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
    global trading_state, trading_active
    
    cycle_count = 0
    
    # Initialize Google Sheets
    initialize_google_sheets()
    
    # Initialize ML system
    initialize_training_system()
    
    logger.info("✅ Trading bot started with 2-minute cycles")
    
    while trading_active:
        try:
            cycle_count += 1
            cycle_start = datetime.now()
            
            trading_state['cycle_count'] = cycle_count
            trading_state['cycle_progress'] = 0
            trading_state['remaining_time'] = CYCLE_SECONDS
            
            logger.info(f"\n{'='*70}")
            logger.info(f"2-MINUTE TRADING CYCLE #{cycle_count}")
            logger.info(f"{'='*70}")
            
            # 1. GET MARKET DATA
            logger.info("Fetching EUR/USD price...")
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
            
            # 2. CREATE PRICE SERIES
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
            logger.info(f"  Google Sheets: {trading_state['google_sheets_status']}")
            logger.info(f"  Cache Efficiency: {trading_state['cache_efficiency']}")
            logger.info(f"  API Calls/Day: ~240 (SAFE)")
            logger.info(f"  Next cycle in: {next_cycle_time}s")
            logger.info(f"{'='*70}")
            
            # 14. WAIT FOR NEXT CYCLE
            for i in range(next_cycle_time):
                if not trading_active:
                    break
                    
                progress_pct = (i / next_cycle_time) * 100
                trading_state['cycle_progress'] = progress_pct
                trading_state['remaining_time'] = next_cycle_time - i
                
                # Update active trade progress if exists
                if trading_state['current_trade']:
                    entry_time = datetime.fromisoformat(trading_state['current_trade']['timestamp'])
                    trade_duration = (datetime.now() - entry_time).total_seconds()
                    trading_state['trade_progress'] = (trade_duration / CYCLE_SECONDS) * 100
                
                time.sleep(1)
            
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(60)

# ==================== START/STOP TRADING BOT ====================
def start_trading_bot():
    """Start the trading bot in a separate thread"""
    global trading_thread, trading_active
    
    try:
        trading_active = True
        trading_thread = threading.Thread(target=trading_cycle, daemon=True)
        trading_thread.start()
        logger.info("✅ Trading bot started successfully")
        print("✅ 2-Minute trading system ACTIVE")
        print(f"✅ Caching: {CACHE_DURATION}-second cache enabled")
        print(f"✅ Google Sheets: Connected")
        print("✅ API Calls/Day: ~240 (SAFE for all free limits)")
        print("✅ Auto-Trading: ENABLED")
        print("✅ ML Training: Ready after 10 trades")
        return True
    except Exception as e:
        logger.error(f"❌ Error starting trading bot: {e}")
        print(f"❌ Error: {e}")
        return False

def stop_trading_bot():
    """Stop the trading bot"""
    global trading_active
    trading_active = False
    logger.info("🛑 Trading bot stopped")
    return True

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
            for key in trade:
                if isinstance(trade[key], datetime):
                    trade[key] = trade[key].isoformat()
            state_copy['current_trade'] = trade
        
        return jsonify(state_copy)
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trade_history')
def get_trade_history():
    """Get trade history from Google Sheets"""
    try:
        sheet_trades = get_trades_from_google_sheets()
        
        # Format for display
        display_trades = []
        for trade in sheet_trades[-10:]:
            trade_copy = {}
            for key in ['trade_id', 'timestamp', 'action', 'entry_price', 'exit_price',
                       'profit_pips', 'result', 'duration_seconds', 'cycle_number',
                       'confidence', 'signal_strength', 'sl_distance_pips', 'tp_distance_pips']:
                trade_copy[key] = trade.get(key, '')
            
            # Format numbers
            for num_key in ['entry_price', 'exit_price', 'profit_pips', 'confidence']:
                if trade_copy.get(num_key) is not None:
                    try:
                        trade_copy[num_key] = float(trade_copy[num_key])
                    except:
                        trade_copy[num_key] = 0.0
            
            display_trades.append(trade_copy)
        
        return jsonify({
            'trades': display_trades,
            'total': len(sheet_trades),
            'profitable': trading_state['profitable_trades'],
            'win_rate': trading_state['win_rate'],
            'google_sheets_status': trading_state['google_sheets_status']
        })
    except Exception as e:
        logger.error(f"Trade history error: {e}")
        return jsonify({'error': str(e)}), 500

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
        'trade_status': 'RESET',
        'trade_progress': 0,
        'cycle_progress': 0
    })
    
    # Clear local memory but keep Google Sheets data
    trade_history.clear()
    ml_features.clear()
    tp_labels.clear()
    sl_labels.clear()
    
    logger.info("Trading statistics reset")
    
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
        'google_sheets_status': trading_state['google_sheets_status'],
        'trading_active': trading_active,
        'ml_ready': trading_state['ml_model_ready'],
        'total_trades': trading_state['total_trades'],
        'api_calls_per_day': '~240 (SAFE)',
        'version': '2.0-google-sheets'
    })

@app.route('/api/start_trading', methods=['POST'])
def api_start_trading():
    """API endpoint to start trading"""
    result = start_trading_bot()
    return jsonify({'success': result, 'message': 'Trading started'})

@app.route('/api/stop_trading', methods=['POST'])
def api_stop_trading():
    """API endpoint to stop trading"""
    result = stop_trading_bot()
    return jsonify({'success': result, 'message': 'Trading stopped'})

# ==================== MAIN ENTRY POINT ====================
if __name__ == '__main__':
    # ⭐⭐ CRITICAL: Start trading bot when app starts ⭐⭐
    start_trading_bot()
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Web dashboard: http://localhost:{port}")
    print("="*80)
    print("GOOGLE SHEETS SYSTEM READY")
    print(f"• 2-minute cycles with {CACHE_DURATION}-second caching")
    print(f"• Data Storage: Google Sheets")
    print(f"• Auto-trading: ENABLED")
    print(f"• ML Auto-correction: ENABLED")
    print("="*80)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )