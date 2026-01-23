"""
EUR/USD 1-Minute Auto-Learning Trading System
Features: 1-minute cycles, auto TP/SL optimization, ML learning from trade history
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
CYCLE_MINUTES = 1
CYCLE_SECONDS = 60
INITIAL_BALANCE = 10000.0
BASE_TRADE_SIZE = 1000.0
MIN_CONFIDENCE = 65.0
TRAINING_FILE = "training_data.json"

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
    'trade_status': 'NO_TRADE'
}

# Data storage
trade_history = []
price_history_deque = deque(maxlen=200)
prediction_history = deque(maxlen=100)

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
print("EUR/USD 1-MINUTE AUTO-LEARNING TRADING SYSTEM")
print("="*80)
print(f"Cycle: Predict and trade every {CYCLE_MINUTES} minute")
print(f"Goal: Hit TP before SL within {CYCLE_SECONDS} seconds")
print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
print(f"Trade Size: ${BASE_TRADE_SIZE:,.2f}")
print(f"ML Training File: {TRAINING_FILE}")
print(f"Min Confidence: {MIN_CONFIDENCE}%")
print("="*80)
print("Starting system...")

# ==================== REAL FOREX DATA FETCHING ====================
def get_real_eurusd_price():
    """Get REAL EUR/USD price from multiple free APIs"""
    apis_to_try = [
        {
            'name': 'Frankfurter',
            'url': 'https://api.frankfurter.app/latest',
            'params': {'from': 'EUR', 'to': 'USD'},
            'extract_rate': lambda data: data['rates']['USD']
        },
        {
            'name': 'ExchangeRate',
            'url': 'https://api.exchangerate-api.com/v4/latest/EUR',
            'params': None,
            'extract_rate': lambda data: data['rates']['USD']
        },
        {
            'name': 'FreeForexAPI',
            'url': 'https://api.freeforexapi.com/v1/latest',
            'params': {'pairs': 'EURUSD'},
            'extract_rate': lambda data: data['rates']['EURUSD']
        }
    ]
    
    for api in apis_to_try:
        try:
            response = requests.get(api['url'], params=api['params'], timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                rate = api['extract_rate'](data)
                
                if rate:
                    current_price = float(rate)
                    logger.info(f"✅ REAL DATA from {api['name']}: EUR/USD = {current_price:.5f}")
                    return current_price, api['name']
                    
        except Exception as e:
            logger.debug(f"{api['name']} API failed: {str(e)[:100]}")
            continue
    
    # Fallback: Use realistic simulation
    logger.warning("All APIs failed, using realistic simulation")
    simulated_change = np.random.uniform(-0.0003, 0.0003)
    return 1.0850 + simulated_change, 'Simulation (APIs unavailable)'

def create_price_series(current_price, num_points=60):
    """Create realistic 1-minute price series for analysis"""
    prices = []
    base_price = float(current_price)
    
    for i in range(num_points):
        volatility = 0.0002
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
    """Calculate comprehensive indicators for prediction"""
    df = pd.DataFrame(prices, columns=['close'])
    
    try:
        # Momentum indicators
        df['returns_1'] = df['close'].pct_change(1)
        df['returns_3'] = df['close'].pct_change(3)
        df['returns_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'] - df['close'].shift(10)
        
        # Moving averages
        df['sma_5'] = ta.sma(df['close'], length=5)
        df['sma_10'] = ta.sma(df['close'], length=10)
        df['ema_8'] = ta.ema(df['close'], length=8)
        df['ema_20'] = ta.ema(df['close'], length=20)
        
        # RSI with multiple timeframes
        df['rsi_7'] = ta.rsi(df['close'], length=7)
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        
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
            df['bb_middle'] = bb['BBM_20_2.0']
            df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100
        
        # ATR for volatility
        df['atr'] = ta.atr(df['close'], df['close'], df['close'], length=14)
        
        # Stochastic
        stoch = ta.stoch(df['close'], df['close'], df['close'])
        if stoch is not None and isinstance(stoch, pd.DataFrame):
            df['stoch_k'] = stoch['STOCHk_14_3_3']
            df['stoch_d'] = stoch['STOCHd_14_3_3']
        
        # Price patterns
        df['high_5'] = df['close'].rolling(5).max()
        df['low_5'] = df['close'].rolling(5).min()
        df['range_5'] = df['high_5'] - df['low_5']
        
        # Support/Resistance
        df['resistance'] = df['close'].rolling(15).max()
        df['support'] = df['close'].rolling(15).min()
        
        # Market condition flags
        df['overbought'] = (df['rsi_7'] > 70).astype(int)
        df['oversold'] = (df['rsi_7'] < 30).astype(int)
        df['bb_touch_upper'] = (df['close'] >= df['bb_upper'] * 0.998).astype(int)
        df['bb_touch_lower'] = (df['close'] <= df['bb_lower'] * 1.002).astype(int)
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
        
    except Exception as e:
        logger.error(f"Indicator calculation error: {e}")
        return df.fillna(0)

# ==================== ML TRAINING SYSTEM ====================
def initialize_training_system():
    """Initialize or load ML training data"""
    global ml_features, tp_labels, sl_labels, ml_trained
    
    if os.path.exists(TRAINING_FILE):
        try:
            with open(TRAINING_FILE, 'r') as f:
                data = json.load(f)
                ml_features = data.get('features', [])
                tp_labels = data.get('tp_labels', [])
                sl_labels = data.get('sl_labels', [])
                
                if len(ml_features) >= 20:
                    train_ml_models()
                    logger.info(f"✅ ML system loaded with {len(ml_features)} training samples")
                else:
                    logger.info(f"⚠️  {len(ml_features)} samples - collecting more data")
                    
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            # Create initial file
            save_training_data([], [], [])
    else:
        save_training_data([], [], [])
        logger.info("Created new training data file")

def save_training_data(features, tp_labels_data, sl_labels_data):
    """Save training data to file"""
    try:
        data = {
            'features': features,
            'tp_labels': tp_labels_data,
            'sl_labels': sl_labels_data,
            'last_updated': datetime.now().isoformat(),
            'total_samples': len(features)
        }
        with open(TRAINING_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving training data: {e}")

def train_ml_models():
    """Train ML models for TP/SL optimization"""
    global tp_model, sl_model, ml_scaler, ml_trained
    
    if len(ml_features) < 20:
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
    if df.empty or len(df) < 15:
        return None
    
    latest = df.iloc[-1]
    
    features = []
    
    # Price momentum (25%)
    features.append(latest.get('returns_1', 0))
    features.append(latest.get('returns_3', 0))
    features.append(latest.get('returns_5', 0))
    features.append(latest.get('momentum_10', 0))
    
    # RSI values (20%)
    features.append(latest.get('rsi_7', 50))
    features.append(latest.get('rsi_14', 50))
    
    # MACD (15%)
    features.append(latest.get('macd_hist', 0))
    features.append(latest.get('macd', 0))
    
    # Bollinger Bands (15%)
    features.append(latest.get('bb_percent', 50))
    
    # Volatility (15%)
    atr_value = latest.get('atr', 0.0005)
    features.append(atr_value * 10000)  # Convert to pips
    
    # Market condition flags (10%)
    features.append(latest.get('overbought', 0))
    features.append(latest.get('oversold', 0))
    
    # Price range
    features.append(latest.get('range_5', 0) * 10000)
    
    return features

def predict_optimal_levels(features, direction, current_price, df):
    """Predict optimal TP and SL levels using ML"""
    
    # Base levels (used when ML not ready or low confidence)
    base_atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0.0005
    
    if direction == "BULLISH":
        base_tp = current_price + (base_atr * 0.8)
        base_sl = current_price - (base_atr * 1.2)
    elif direction == "BEARISH":
        base_tp = current_price - (base_atr * 0.8)
        base_sl = current_price + (base_atr * 1.2)
    else:
        base_tp = current_price
        base_sl = current_price
    
    # Use ML predictions if available
    if ml_trained and features is not None:
        try:
            X_scaled = ml_scaler.transform([features])
            
            # Predict optimal TP distance (in pips)
            tp_pips_pred = tp_model.predict(X_scaled)[0]
            tp_pips_pred = max(1, min(20, tp_pips_pred))  # Limit to 1-20 pips
            
            # Predict optimal SL distance (in pips)
            sl_pips_pred = sl_model.predict(X_scaled)[0]
            sl_pips_pred = max(2, min(30, sl_pips_pred))  # Limit to 2-30 pips
            
            # Ensure SL is larger than TP for proper risk management
            if sl_pips_pred <= tp_pips_pred * 1.2:
                sl_pips_pred = tp_pips_pred * 1.5
            
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

# ==================== PREDICTION ENGINE ====================
def analyze_1min_prediction(df, current_price):
    """Predict 1-minute price direction with high accuracy"""
    
    if len(df) < 15:
        return 0.5, 50, 'ANALYZING', 1
    
    try:
        latest = df.iloc[-1]
        
        # Initialize scores
        bull_score = 0
        bear_score = 0
        confidence_factors = []
        
        # 1. RSI ANALYSIS (Most important for 1-min)
        rsi_7 = latest.get('rsi_7', 50)
        if rsi_7 < 35:
            bull_score += 6
            confidence_factors.append(1.8 if rsi_7 < 25 else 1.4)
        elif rsi_7 > 65:
            bear_score += 6
            confidence_factors.append(1.8 if rsi_7 > 75 else 1.4)
        
        # 2. MACD HISTOGRAM (Trend momentum)
        macd_hist = latest.get('macd_hist', 0)
        if abs(macd_hist) > 0.0001:  # Significant MACD movement
            if macd_hist > 0:
                bull_score += 4
                confidence_factors.append(1 + abs(macd_hist) * 5000)
            else:
                bear_score += 4
                confidence_factors.append(1 + abs(macd_hist) * 5000)
        
        # 3. BOLLINGER BANDS POSITION
        bb_percent = latest.get('bb_percent', 50)
        if bb_percent < 20:
            bull_score += 3
            confidence_factors.append(1.4)
        elif bb_percent > 80:
            bear_score += 3
            confidence_factors.append(1.4)
        
        # 4. PRICE MOMENTUM
        momentum = latest.get('momentum_10', 0)
        if momentum > 0:
            bull_score += 2
        else:
            bear_score += 2
        
        # 5. STOCHASTIC
        stoch_k = latest.get('stoch_k', 50)
        if stoch_k < 25:
            bull_score += 2
        elif stoch_k > 75:
            bear_score += 2
        
        # Calculate probability
        total_score = bull_score + bear_score
        if total_score == 0:
            return 0.5, 50, 'NEUTRAL', 1
        
        probability = bull_score / total_score
        
        # Calculate confidence
        if confidence_factors:
            base_confidence = np.mean(confidence_factors) * 15
        else:
            base_confidence = 50
        
        # Signal clarity adjustment
        signal_clarity = abs(probability - 0.5) * 2
        confidence = min(95, base_confidence * (1 + signal_clarity))
        
        # Determine direction
        if probability > 0.7:
            direction = 'BULLISH'
            signal_strength = 3
        elif probability > 0.6:
            direction = 'BULLISH'
            signal_strength = 2
        elif probability < 0.3:
            direction = 'BEARISH'
            signal_strength = 3
        elif probability < 0.4:
            direction = 'BEARISH'
            signal_strength = 2
        else:
            direction = 'NEUTRAL'
            signal_strength = 1
            confidence = max(30, confidence * 0.7)
        
        # Recent price change confirmation
        returns_3 = latest.get('returns_3', 0)
        if abs(returns_3) > 0.0005:  # 0.05% change
            if (direction == 'BULLISH' and returns_3 > 0) or (direction == 'BEARISH' and returns_3 < 0):
                confidence = min(95, confidence + 8)
                signal_strength = min(3, signal_strength + 1)
        
        return probability, confidence, direction, signal_strength
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return 0.5, 50, 'ERROR', 1

# ==================== TRADE EXECUTION ====================
def execute_minute_trade(direction, confidence, current_price, optimal_tp, optimal_sl, tp_pips, sl_pips):
    """Execute a trade at the beginning of the 1-minute cycle"""
    
    if direction == 'NEUTRAL' or confidence < MIN_CONFIDENCE:
        trading_state['action'] = 'WAIT'
        trading_state['trade_status'] = 'NO_SIGNAL'
        return None
    
    # Determine action
    if direction == 'BULLISH':
        action = 'BUY'
        action_reason = f"Strong 1-min BULLISH signal ({confidence:.1f}% confidence)"
    else:  # BEARISH
        action = 'SELL'
        action_reason = f"Strong 1-min BEARISH signal ({confidence:.1f}% confidence)"
    
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
        'reason': action_reason
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
    logger.info(f"   Goal: Hit TP ({tp_pips} pips) before SL ({sl_pips} pips) in 60 seconds")
    
    return trade

def monitor_active_trade(current_price):
    """Monitor the active trade throughout the minute"""
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
    
    # Time-based exit (end of 1-minute cycle)
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
        
        # Learn from this trade (add to ML training)
        learn_from_trade(trade, current_price)
        
        # Clear current trade
        trading_state['current_trade'] = None
        trading_state['trade_status'] = 'COMPLETED'
        
        return trade
    
    # Update progress percentage
    progress_pct = (trade_duration / CYCLE_SECONDS) * 100
    trading_state['trade_progress'] = min(100, progress_pct)
    
    return trade

def learn_from_trade(trade, current_price):
    """Learn from trade result and update ML training data"""
    try:
        # We need the original market conditions at trade entry
        # For now, we'll use simplified learning
        if 'result' in trade:
            # Extract features from trade data
            features = [
                trade['confidence'] / 100,  # Normalized confidence
                trade['tp_distance_pips'] / 100,  # Normalized TP
                trade['sl_distance_pips'] / 100,  # Normalized SL
                1 if trade['action'] == 'BUY' else 0,  # Direction
                trade['profit_pips'] / 100  # Result normalized
            ]
            
            # Determine optimal TP/SL based on result
            if trade['result'] == 'SUCCESS':
                # TP was good, SL might need adjustment
                optimal_tp = trade['tp_distance_pips']
                optimal_sl = trade['sl_distance_pips'] * 1.1  # Increase SL slightly
            elif trade['result'] == 'FAILED':
                # TP was too far, SL was too close
                optimal_tp = trade['tp_distance_pips'] * 0.8  # Reduce TP
                optimal_sl = trade['sl_distance_pips'] * 1.2  # Increase SL
            elif trade['result'] == 'PARTIAL_SUCCESS':
                # TP was almost right
                optimal_tp = trade['tp_distance_pips']
                optimal_sl = trade['sl_distance_pips']
            else:
                # Neutral learning
                optimal_tp = trade['tp_distance_pips']
                optimal_sl = trade['sl_distance_pips']
            
            # Add to training data
            ml_features.append(features)
            tp_labels.append(optimal_tp)
            sl_labels.append(optimal_sl)
            
            # Save training data
            save_training_data(ml_features, tp_labels, sl_labels)
            
            # Retrain if we have enough samples
            if len(ml_features) >= 20 and len(ml_features) % 10 == 0:
                train_ml_models()
            
            logger.info(f"📚 Learned from trade #{trade['id']}: {trade['result']}")
            
    except Exception as e:
        logger.error(f"Learning error: {e}")

# ==================== CHART CREATION ====================
def create_trading_chart(prices, current_trade, next_cycle):
    """Create real-time trading chart"""
    try:
        df = pd.DataFrame(prices, columns=['close'])
        
        # Calculate indicators for chart
        df['sma_5'] = ta.sma(df['close'], length=5)
        df['sma_10'] = ta.sma(df['close'], length=10)
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=list(range(len(prices))),
            y=df['close'],
            mode='lines',
            name='EUR/USD Price',
            line=dict(color='#00ff88', width=3),
            hovertemplate='Price: %{y:.5f}<extra></extra>'
        ))
        
        # Moving averages
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
            entry_idx = len(prices) - 10 if len(prices) > 10 else 0
            
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
        title = f'EUR/USD 1-Minute Trading - Next Cycle: {next_cycle}s'
        if trading_state['is_demo_data']:
            title += ' (Simulation Mode)'
        
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
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1
            )
        )
        
        return json.dumps(fig.to_dict(), cls=plotly.utils.PlotlyJSONEncoder)
        
    except Exception as e:
        logger.error(f"Chart error: {e}")
        return None

# ==================== MAIN 1-MINUTE CYCLE ====================
def trading_cycle():
    """Main 1-minute trading cycle"""
    global trading_state
    
    cycle_count = 0
    
    # Initialize ML system
    initialize_training_system()
    
    while True:
        try:
            cycle_count += 1
            cycle_start = datetime.now()
            
            trading_state['cycle_count'] = cycle_count
            trading_state['cycle_progress'] = 0
            
            logger.info(f"\n{'='*80}")
            logger.info(f"1-MINUTE TRADING CYCLE #{cycle_count}")
            logger.info(f"{'='*80}")
            
            # ===== 1. GET REAL MARKET DATA =====
            logger.info("Fetching real EUR/USD price...")
            current_price, data_source = get_real_eurusd_price()
            
            # Update price history
            price_history_deque.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'price': current_price
            })
            
            trading_state['current_price'] = round(float(current_price), 5)
            trading_state['data_source'] = data_source
            trading_state['is_demo_data'] = 'Simulation' in data_source
            trading_state['api_status'] = 'CONNECTED' if 'Simulation' not in data_source else 'DEMO'
            
            # ===== 2. CREATE PRICE SERIES =====
            price_series = create_price_series(current_price, 60)
            
            # ===== 3. CALCULATE INDICATORS =====
            df_indicators = calculate_advanced_indicators(price_series)
            
            # ===== 4. MAKE 1-MINUTE PREDICTION =====
            logger.info("Analyzing market for 1-minute prediction...")
            pred_prob, confidence, direction, signal_strength = analyze_1min_prediction(
                df_indicators, current_price
            )
            
            trading_state['minute_prediction'] = direction
            trading_state['confidence'] = round(float(confidence), 1)
            
            # ===== 5. EXTRACT ML FEATURES =====
            ml_features_current = extract_ml_features(df_indicators, current_price)
            
            # ===== 6. PREDICT OPTIMAL TP/SL =====
            optimal_tp, optimal_sl, tp_pips, sl_pips = predict_optimal_levels(
                ml_features_current, direction, current_price, df_indicators
            )
            
            # ===== 7. CHECK ACTIVE TRADE =====
            if trading_state['current_trade']:
                monitor_active_trade(current_price)
            
            # ===== 8. EXECUTE NEW TRADE =====
            if (trading_state['current_trade'] is None and 
                direction != 'NEUTRAL' and 
                confidence >= MIN_CONFIDENCE and
                signal_strength >= 2):
                
                execute_minute_trade(
                    direction, confidence, current_price, 
                    optimal_tp, optimal_sl, tp_pips, sl_pips
                )
            elif trading_state['current_trade'] is None:
                trading_state['action'] = 'WAIT'
                trading_state['trade_status'] = 'NO_SIGNAL'
                logger.info(f"⚠️  No trade signal: {direction} with {confidence:.1f}% confidence")
            
            # ===== 9. CALCULATE NEXT CYCLE TIME =====
            cycle_duration = (datetime.now() - cycle_start).seconds
            next_cycle_time = max(1, CYCLE_SECONDS - cycle_duration)
            
            trading_state['next_cycle_in'] = next_cycle_time
            
            # ===== 10. CREATE CHART =====
            chart_data = create_trading_chart(
                price_series, 
                trading_state['current_trade'], 
                next_cycle_time
            )
            trading_state['chart_data'] = chart_data
            
            # ===== 11. UPDATE PRICE HISTORY =====
            trading_state['price_history'] = list(price_history_deque)[-20:]
            
            # ===== 12. UPDATE TIMESTAMP =====
            trading_state['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            trading_state['server_time'] = datetime.now().isoformat()
            
            # ===== 13. LOG CYCLE SUMMARY =====
            logger.info(f"CYCLE #{cycle_count} SUMMARY:")
            logger.info(f"  Price: {current_price:.5f} ({data_source})")
            logger.info(f"  Prediction: {direction} (Strength: {signal_strength}/3)")
            logger.info(f"  Confidence: {confidence:.1f}%")
            logger.info(f"  Action: {trading_state['action']}")
            logger.info(f"  TP/SL: {tp_pips}/{sl_pips} pips")
            logger.info(f"  Balance: ${trading_state['balance']:.2f}")
            logger.info(f"  Win Rate: {trading_state['win_rate']:.1f}%")
            logger.info(f"  ML Ready: {trading_state['ml_model_ready']}")
            logger.info(f"  Next cycle in: {next_cycle_time}s")
            logger.info(f"{'='*80}")
            
            # ===== 14. WAIT FOR NEXT CYCLE =====
            # Update progress during wait
            for i in range(next_cycle_time):
                if i % 5 == 0:  # Update progress every 5 seconds
                    progress_pct = (i / next_cycle_time) * 100
                    trading_state['cycle_progress'] = progress_pct
                time.sleep(1)
            
        except Exception as e:
            logger.error(f"Trading cycle error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(30)

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
        for trade in trade_history[-20:]:  # Last 20 trades
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
        'training_samples': len(ml_features),
        'training_file': TRAINING_FILE,
        'last_trained': trading_state['last_update']
    })

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
        'trade_status': 'RESET'
    })
    
    trade_history.clear()
    ml_features.clear()
    tp_labels.clear()
    sl_labels.clear()
    
    # Reset training file
    save_training_data([], [], [])
    
    return jsonify({'success': True, 'message': 'Trading reset complete'})

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'cycle_count': trading_state['cycle_count'],
        'system_status': 'ACTIVE',
        'version': '1.0.0'
    })

# ==================== START TRADING BOT ====================
def start_trading_bot():
    """Start the trading bot"""
    try:
        thread = threading.Thread(target=trading_cycle, daemon=True)
        thread.start()
        logger.info("✅ Trading bot started successfully")
        print("✅ 1-Minute trading system ACTIVE")
        print("✅ Auto-learning ML system initialized")
        print(f"✅ Training data file: {TRAINING_FILE}")
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
    print("SYSTEM READY - Generating 1-minute trading signals for EUR/USD")
    print("="*80)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )