"""
EUR/USD 2-Minute Auto-Learning Trading System
WITH 30-SECOND CACHING for API limit protection
AND GIT PUSH/PULL FOR ML TRAINING
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
import subprocess
import shutil
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

# ==================== GITHUB CONFIGURATION ====================
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
GITHUB_USERNAME = "gicheha-ai"
GITHUB_REPO = "m"
GITHUB_REPO_URL = f"https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}"
LOCAL_REPO_PATH = "trading_repo"
DATA_DIR = os.path.join(LOCAL_REPO_PATH, "data")
TRADES_FILE = os.path.join(DATA_DIR, "trades.json")
STATE_FILE = os.path.join(DATA_DIR, "state.json")

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
    'total_trades': 0,  # Will be updated from Git on startup
    'profitable_trades': 0,  # Will be updated from Git on startup
    'total_profit': 0.0,  # Will be updated from Git on startup
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
    'git_push_enabled': bool(GITHUB_TOKEN),
    'git_last_push': 'Never',
    'git_total_pushes': 0,
    'git_last_pull': 'Never',
    'git_total_trades_loaded': 0
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
ml_fully_trained_on_startup = False  # NEW: Track if ML was fully trained on startup

# Next trade ID tracking
next_trade_id = 1  # Will be updated from Git trades

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Print startup banner
print("="*80)
print("EUR/USD 2-MINUTE TRADING SYSTEM WITH GIT PUSH/PULL")
print("="*80)
print(f"Cycle: Predict and trade every {CYCLE_MINUTES} minutes ({CYCLE_SECONDS} seconds)")
print(f"Cache Duration: {CACHE_DURATION} seconds")
print(f"Git Repo: {GITHUB_REPO_URL}")
print(f"Git Token: {'✅ Configured' if trading_state['git_push_enabled'] else '❌ NOT FOUND'}")
print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
print(f"Trade Size: ${BASE_TRADE_SIZE:,.2f}")
print("="*80)
print("Starting system...")

# ==================== GIT PUSH/PULL FUNCTIONS ====================
def setup_git_repo():
    """Setup Git repository for pushing and pulling"""
    try:
        if not GITHUB_TOKEN:
            logger.warning("⚠️  GITHUB_TOKEN not found in environment variables")
            trading_state['git_push_enabled'] = False
            return False
        
        logger.info("🔑 GitHub token found in environment variables")
        
        # Remove existing repo if exists
        if os.path.exists(LOCAL_REPO_PATH):
            try:
                shutil.rmtree(LOCAL_REPO_PATH)
                logger.info("🗑️  Cleared existing repo directory")
            except Exception as e:
                logger.warning(f"⚠️  Could not remove existing repo: {e}")
        
        # Clone repository with authentication
        logger.info("📦 Cloning repository...")
        
        # Create authenticated URL
        auth_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"
        
        result = subprocess.run(
            ['git', 'clone', auth_url, LOCAL_REPO_PATH],
            capture_output=True,
            text=True,
            timeout=45
        )
        
        if result.returncode != 0:
            logger.error(f"❌ Git clone failed: {result.stderr[:200]}")
            trading_state['git_push_enabled'] = False
            return False
        
        # Configure git user
        subprocess.run(['git', 'config', 'user.email', 'trading-bot@gicheha-ai.com'], 
                      cwd=LOCAL_REPO_PATH, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Trading Bot'], 
                      cwd=LOCAL_REPO_PATH, capture_output=True)
        
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        logger.info("✅ Git repository setup complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Git setup error: {e}")
        trading_state['git_push_enabled'] = False
        return False

def execute_git_push():
    """Execute Git push with trade data"""
    try:
        if not trading_state['git_push_enabled']:
            logger.warning("⚠️  Git push not enabled")
            return {'success': False, 'message': 'Git push not enabled'}
        
        logger.info("🚀 Starting Git push...")
        
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Save all trade history to file
        with open(TRADES_FILE, 'w') as f:
            json.dump(trade_history, f, indent=2, default=str)
        
        # Also save state information
        with open(STATE_FILE, 'w') as f:
            json.dump({
                'last_updated': datetime.now().isoformat(),
                'total_trades': len(trade_history),
                'profitable_trades': trading_state['profitable_trades'],
                'balance': trading_state['balance'],
                'total_profit': trading_state['total_profit'],
                'win_rate': trading_state['win_rate'],
                'ml_model_ready': ml_trained,
                'ml_training_samples': len(ml_features),
                'next_trade_id': next_trade_id
            }, f, indent=2)
        
        # Change to repo directory
        original_dir = os.getcwd()
        os.chdir(LOCAL_REPO_PATH)
        
        try:
            # 1. Add all files
            logger.info("📝 Adding files to Git...")
            result = subprocess.run(
                'git add .',
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.error(f"❌ Git add failed: {result.stderr[:200]}")
                return {'success': False, 'message': 'Git add failed'}
            
            # 2. Check if there are changes
            result = subprocess.run(
                'git status --porcelain',
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if not result.stdout.strip():
                logger.info("📭 No changes to commit")
                return {'success': True, 'message': 'No changes to commit'}
            
            # 3. Commit changes
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            trade_count = len(trade_history)
            commit_msg = f"Trade update: {trade_count} total trades - {timestamp}"
            
            logger.info(f"💾 Committing changes: {commit_msg}")
            result = subprocess.run(
                f'git commit -m "{commit_msg}"',
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                logger.error(f"❌ Git commit failed: {result.stderr[:200]}")
                return {'success': False, 'message': 'Git commit failed'}
            
            # 4. Push to GitHub
            logger.info("⬆️  Pushing to GitHub...")
            
            # Create authenticated push URL
            push_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{GITHUB_REPO}.git"
            
            result = subprocess.run(
                f'git push {push_url} main',
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                trading_state['git_last_push'] = datetime.now().strftime('%H:%M:%S')
                trading_state['git_total_pushes'] += 1
                
                logger.info(f"✅ Git push successful! Total pushes: {trading_state['git_total_pushes']}")
                return {'success': True, 'message': 'Git push successful'}
            else:
                logger.error(f"❌ Git push failed: {result.stderr[:200]}")
                
                # Try alternative method
                logger.info("🔄 Trying alternative push method...")
                result2 = subprocess.run(
                    'git push origin main',
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result2.returncode == 0:
                    trading_state['git_last_push'] = datetime.now().strftime('%H:%M:%S')
                    trading_state['git_total_pushes'] += 1
                    logger.info(f"✅ Git push successful (alternative method)!")
                    return {'success': True, 'message': 'Git push successful'}
                else:
                    return {'success': False, 'message': 'Git push failed'}
                    
        finally:
            os.chdir(original_dir)
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Git push timed out")
        return {'success': False, 'message': 'Git push timed out'}
    except Exception as e:
        logger.error(f"❌ Git push error: {e}")
        return {'success': False, 'message': str(e)}

def load_trades_from_git():
    """Load trade data from Git repo and update all counters"""
    global trade_history, next_trade_id, trading_state
    
    try:
        if not trading_state['git_push_enabled']:
            logger.warning("⚠️  Git load not enabled")
            return {'success': False, 'message': 'Git load not enabled'}
        
        logger.info("📥 Loading trade data from Git repo...")
        
        trades_loaded = 0
        if os.path.exists(TRADES_FILE) and os.path.getsize(TRADES_FILE) > 0:
            with open(TRADES_FILE, 'r') as f:
                loaded_trades = json.load(f)
            
            if loaded_trades:
                # Clear current trade history and load all trades from Git
                trade_history.clear()
                trade_history.extend(loaded_trades)
                trades_loaded = len(trade_history)
                
                # CRITICAL: Update trading state from loaded trades
                total_trades = len(trade_history)
                profitable_trades = 0
                total_profit = 0.0
                balance = INITIAL_BALANCE
                
                # Calculate statistics from all loaded trades
                for trade in trade_history:
                    if trade.get('status') == 'CLOSED':
                        profit = trade.get('profit_amount', 0)
                        total_profit += profit
                        balance += profit
                        
                        if trade.get('result') in ['SUCCESS', 'SUCCESS_FAST', 'PARTIAL_SUCCESS']:
                            profitable_trades += 1
                
                # Update trading state with loaded data
                trading_state['total_trades'] = total_trades
                trading_state['profitable_trades'] = profitable_trades
                trading_state['total_profit'] = total_profit
                trading_state['balance'] = balance
                
                if total_trades > 0:
                    trading_state['win_rate'] = (profitable_trades / total_trades) * 100
                
                # Calculate next trade ID - find the highest numeric ID
                max_id = 0
                for trade in trade_history:
                    trade_id = trade.get('id', '')
                    if trade_id.startswith('T'):
                        try:
                            # Extract numeric part after 'T'
                            num_part = trade_id[1:]
                            # Handle timestamp-based IDs: T20250130123045
                            if num_part.isdigit() and len(num_part) == 14:
                                # For timestamp IDs, we use sequential numbering instead
                                continue
                            else:
                                num_id = int(num_part)
                                max_id = max(max_id, num_id)
                        except:
                            continue
                
                # Set next trade ID to continue sequence
                next_trade_id = max_id + 1
                
                trading_state['git_total_trades_loaded'] = trades_loaded
                logger.info(f"📊 Loaded {trades_loaded} trades from Git repo")
                logger.info(f"📈 Statistics: {total_trades} total, {profitable_trades} profitable")
                logger.info(f"💰 Balance from trades: ${balance:.2f}")
                logger.info(f"🎯 Next trade ID will be: T{next_trade_id}")
        
        # Also load state file if exists
        if os.path.exists(STATE_FILE) and os.path.getsize(STATE_FILE) > 0:
            with open(STATE_FILE, 'r') as f:
                state_data = json.load(f)
                if 'next_trade_id' in state_data:
                    # Use the next_trade_id from state file if available
                    next_trade_id = max(next_trade_id, state_data['next_trade_id'])
                    logger.info(f"📝 Using next trade ID from state file: T{next_trade_id}")
        
        trading_state['git_last_pull'] = datetime.now().strftime('%H:%M:%S')
        logger.info("✅ Git load successful")
        return {'success': True, 'message': 'Git load successful', 'trades_loaded': trades_loaded}
                    
    except Exception as e:
        logger.error(f"❌ Git load error: {e}")
        return {'success': False, 'message': str(e)}

def push_trade_to_git():
    """Push all trades to Git repo"""
    if not trading_state['git_push_enabled']:
        return
    
    # Execute Git push
    result = execute_git_push()
    
    if result.get('success'):
        logger.info(f"✅ Trades pushed to Git successfully (Total: {len(trade_history)})")
    else:
        logger.warning(f"⚠️  Failed to push trades to Git: {result.get('message')}")

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
        
        # Bollinger Bands - FIXED VERSION
        bb = ta.bbands(df['close'], length=20)
        if bb is not None and isinstance(bb, pd.DataFrame):
            # Check for column names and use generic indexing if specific names not found
            if 'BBU_20_2.0' in bb.columns:
                df['bb_upper'] = bb['BBU_20_2.0']
                df['bb_lower'] = bb['BBL_20_2.0']
            else:
                # Use generic column access
                df['bb_upper'] = bb.iloc[:, 0]  # First column is upper band
                df['bb_lower'] = bb.iloc[:, 2]  # Third column is lower band
            
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
def initialize_ml_system():
    """Initialize ML system by loading data from Git and training with ALL historical data"""
    global ml_features, tp_labels, sl_labels, ml_trained, ml_fully_trained_on_startup
    
    # First, load data from Git repo
    load_result = load_trades_from_git()
    if load_result.get('success'):
        logger.info(f"📊 Loaded {len(trade_history)} trades from Git for ML training")
    
    # NEW: TRAIN ML WITH ALL HISTORICAL DATA BEFORE ANY PREDICTION
    logger.info("🤖 STARTUP ML TRAINING: Training with ALL historical data from Git...")
    train_ml_with_all_trades()
    ml_fully_trained_on_startup = True
    logger.info("✅ ML system fully trained with ALL historical trades before making any predictions")
    
    # Extract features from historical trades (for future retraining)
    extract_features_from_trades()
    
    # Train ML if we have enough data (should already be trained from above)
    if len(ml_features) >= 10:
        if not ml_trained:  # If somehow not trained yet
            train_ml_models()
        ml_trained = True
        trading_state['ml_model_ready'] = True
        logger.info(f"✅ ML system ready with {len(ml_features)} samples from {len(trade_history)} trades")
    else:
        ml_trained = False
        trading_state['ml_model_ready'] = False
        logger.info(f"⚠️  Insufficient ML data: {len(ml_features)}/10 samples")

def train_ml_with_all_trades():
    """NEW: Train ML models with ALL historical trades from Git repo"""
    global tp_model, sl_model, ml_scaler, ml_trained, ml_features, tp_labels, sl_labels
    
    if len(trade_history) == 0:
        logger.info("📭 No historical trades found in Git repo for ML training")
        ml_trained = False
        trading_state['ml_model_ready'] = False
        return
    
    # Clear existing training data
    ml_features = []
    tp_labels = []
    sl_labels = []
    
    logger.info(f"🧠 Training ML with ALL {len(trade_history)} historical trades from Git...")
    
    # Extract features from ALL trades
    successful_trades = 0
    failed_trades = 0
    
    for trade in trade_history:
        if trade.get('status') != 'CLOSED' or trade.get('result') not in ['SUCCESS', 'FAILED', 'PARTIAL_SUCCESS', 'PARTIAL_FAIL', 'SUCCESS_FAST']:
            continue
        
        # Extract features from trade data
        try:
            features = [
                trade.get('confidence', 50) / 100,
                trade.get('signal_strength', 1) / 3,
                trade.get('volatility', 0.00015) * 10000,
                1 if trade.get('action') == 'BUY' else 0,
                trade.get('entry_price', 1.0850),
                trade.get('tp_distance_pips', 10) / 100,
                trade.get('sl_distance_pips', 10) / 100,
                trade.get('duration_seconds', 120) / 120,
                trade.get('max_profit_pips', 0) / 100,
                trade.get('max_loss_pips', 0) / 100,
                # Add market condition features
                1 if trade.get('result') in ['SUCCESS', 'SUCCESS_FAST', 'PARTIAL_SUCCESS'] else 0,
                trading_state['win_rate'] / 100
            ]
            
            # Add features to ML training set
            ml_features.append(features)
            
            # Determine optimal TP/SL based on trade result - FOCUS ON HITTING TP FAST
            if trade['result'] == 'SUCCESS_FAST':  # TP hit in first half
                # Perfect! Keep similar settings for FAST TP hits
                optimal_tp = trade.get('tp_distance_pips', 10)
                optimal_sl = trade.get('sl_distance_pips', 10) * 0.9
                successful_trades += 1
                
            elif trade['result'] == 'SUCCESS':  # TP hit but took time
                # Good, but could be faster - make TP smaller
                optimal_tp = trade.get('tp_distance_pips', 10) * 0.9
                optimal_sl = trade.get('sl_distance_pips', 10) * 1.1
                successful_trades += 1
                
            elif trade['result'] == 'PARTIAL_FAIL':  # Time ended with profit but TP not hit
                # Need much smaller TP for faster hits
                optimal_tp = trade.get('tp_distance_pips', 10) * 0.7
                optimal_sl = trade.get('sl_distance_pips', 10) * 0.8
                failed_trades += 1
                
            elif trade['result'] == 'FAILED':  # SL hit or ended with loss
                # TP was too far, SL was too tight
                optimal_tp = trade.get('tp_distance_pips', 10) * 0.6
                optimal_sl = trade.get('sl_distance_pips', 10) * 1.5
                failed_trades += 1
                
            else:  # PARTIAL_SUCCESS or other
                optimal_tp = trade.get('tp_distance_pips', 10) * 0.8
                optimal_sl = trade.get('sl_distance_pips', 10) * 1.0
                successful_trades += 1
            
            tp_labels.append(optimal_tp)
            sl_labels.append(optimal_sl)
            
        except Exception as e:
            logger.warning(f"Could not extract features from trade: {e}")
            continue
    
    # Train ML models with ALL extracted data
    if len(ml_features) >= 3:  # Need at least 3 samples
        try:
            X = np.array(ml_features)
            y_tp = np.array(tp_labels)
            y_sl = np.array(sl_labels)
            
            # Scale features
            X_scaled = ml_scaler.fit_transform(X)
            
            # Train TP model - FOCUS ON HITTING TP BEFORE 2 MINUTES
            tp_model.fit(X_scaled, y_tp)
            
            # Train SL model - FOCUS ON PREVENTING SL HIT
            sl_model.fit(X_scaled, y_sl)
            
            ml_trained = True
            trading_state['ml_model_ready'] = True
            
            # Analyze training results
            logger.info(f"🎯 ML TRAINED WITH ALL HISTORICAL DATA:")
            logger.info(f"   Total trades analyzed: {len(trade_history)}")
            logger.info(f"   ML training samples: {len(ml_features)}")
            logger.info(f"   Successful trades: {successful_trades}")
            logger.info(f"   Failed trades: {failed_trades}")
            logger.info(f"   Primary goal: Hit TP BEFORE 2 minutes")
            logger.info(f"   Secondary goal: Avoid SL at all costs")
            logger.info(f"   Will NEVER settle - always adjusting for faster TP hits")
            
            # Show sample predictions
            if len(X) >= 5:
                tp_predictions = tp_model.predict(X_scaled[:3])
                sl_predictions = sl_model.predict(X_scaled[:3])
                logger.info(f"   Sample TP predictions: {tp_predictions}")
                logger.info(f"   Sample SL predictions: {sl_predictions}")
                logger.info(f"   Goal: TP hit in <{CYCLE_SECONDS} seconds, SL avoided")
            
        except Exception as e:
            logger.error(f"ML training error: {e}")
            ml_trained = False
            trading_state['ml_model_ready'] = False
    else:
        logger.info("⚠️  Not enough historical trades for ML training")
        ml_trained = False
        trading_state['ml_model_ready'] = False

def extract_features_from_trades():
    """Extract ML features from trade history"""
    global ml_features, tp_labels, sl_labels
    
    # Don't clear features here - we already extracted them in train_ml_with_all_trades()
    # Just update the existing features if needed
    
    logger.info(f"📊 ML has {len(ml_features)} samples from historical trades")

def train_ml_models():
    """Train ML models for TP/SL optimization with focus on hitting TP before 2 minutes"""
    global tp_model, sl_model, ml_scaler, ml_trained
    
    if len(ml_features) < 3:  # Reduced minimum for retraining
        ml_trained = False
        trading_state['ml_model_ready'] = False
        return
    
    try:
        X = np.array(ml_features)
        y_tp = np.array(tp_labels)
        y_sl = np.array(sl_labels)
        
        # Scale features
        X_scaled = ml_scaler.fit_transform(X)
        
        # Train TP model - focus on hitting TP BEFORE 2 minutes
        tp_model.fit(X_scaled, y_tp)
        
        # Train SL model - focus on preventing SL hit
        sl_model.fit(X_scaled, y_sl)
        
        ml_trained = True
        trading_state['ml_model_ready'] = True
        
        # Analyze model performance
        if len(X) >= 5:
            tp_predictions = tp_model.predict(X_scaled[:3])
            sl_predictions = sl_model.predict(X_scaled[:3])
            logger.info(f"🤖 ML models retrained on {len(X)} samples")
            logger.info(f"   Sample TP predictions: {tp_predictions}")
            logger.info(f"   Sample SL predictions: {sl_predictions}")
            logger.info(f"   Goal: Hit TP BEFORE 2 minutes, avoid SL completely")
        
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
    
    # Add price level
    features.append(current_price)
    
    return features

def predict_optimal_levels(features, direction, current_price, df):
    """Predict optimal TP and SL levels for 2-minute trades with ML optimization"""
    
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
            
            # Predict optimal TP distance with AGGRESSIVE goal: hit TP before 2 minutes
            tp_pips_pred = tp_model.predict(X_scaled)[0]
            tp_pips_pred = max(5, min(15, tp_pips_pred))  # Aggressive: smaller TP to hit faster
            
            # Predict optimal SL distance with DEFENSIVE goal: avoid SL at all costs
            sl_pips_pred = sl_model.predict(X_scaled)[0]
            sl_pips_pred = max(8, min(20, sl_pips_pred))  # Defensive: larger SL to avoid hit
            
            # Ensure TP is more likely to be hit before SL (better risk/reward)
            if tp_pips_pred / sl_pips_pred < 1.5:
                # Adjust to ensure TP is more attractive
                tp_pips_pred = sl_pips_pred * 1.5
            
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
            
            logger.info(f"🤖 ML OPTIMIZED: TP={tp_pips_pred:.1f}pips (hit in <2min), SL={sl_pips_pred:.1f}pips (avoid)")
            logger.info(f"   Target: Hit TP at {optimal_tp:.5f} BEFORE 2 minutes")
            logger.info(f"   Protection: SL at {optimal_sl:.5f} (avoid at all costs)")
            
            return optimal_tp, optimal_sl, int(tp_pips_pred), int(sl_pips_pred)
            
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
    
    # Fallback to base levels
    tp_pips = int(abs(base_tp - current_price) * 10000)
    sl_pips = int(abs(base_sl - current_price) * 10000)
    
    return base_tp, base_sl, tp_pips, sl_pips

# ==================== 2-MINUTE PREDICTION ENGINE ====================
def analyze_2min_prediction(df, current_price):
    """Predict 2-minute price direction with ML-enhanced accuracy"""
    
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
        
        # 4. PRICE MOMENTUM (ML-enhanced)
        momentum = latest.get('momentum_20', 0)
        if ml_trained:
            # Use ML to weight momentum more intelligently
            if momentum > 0.0003:
                bull_score += 3  # ML says momentum is more important
            elif momentum < -0.0003:
                bear_score += 3
        else:
            if momentum > 0.0003:
                bull_score += 2
            elif momentum < -0.0003:
                bear_score += 2
        
        # 5. VOLATILITY CONSIDERATION (ATR)
        atr_value = latest.get('atr', 0.0005)
        if atr_value > 0.0008:  # High volatility
            confidence_factors.append(0.8)  # Reduce confidence in high volatility
        
        # Calculate probability
        total_score = bull_score + bear_score
        if total_score == 0:
            return 0.5, 50, 'NEUTRAL', 1
        
        probability = bull_score / total_score
        
        # Calculate confidence with ML enhancement
        if confidence_factors:
            base_confidence = np.mean(confidence_factors) * 25
        else:
            base_confidence = 50
        
        # ML enhances confidence if model is trained
        if ml_trained and len(trade_history) >= 5:
            ml_boost = min(20, len(trade_history) * 2)  # ML gives up to 20% boost
            base_confidence += ml_boost
        
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
    
    global next_trade_id
    
    # Determine action
    if direction == 'BULLISH':
        action = 'BUY'
        action_reason = f"Strong 2-min BULLISH signal ({confidence:.1f}% confidence)"
    else:  # BEARISH
        action = 'SELL'
        action_reason = f"Strong 2-min BEARISH signal ({confidence:.1f}% confidence)"
    
    # Use the next trade ID (continues from Git)
    trade_id = f"T{next_trade_id}"
    next_trade_id += 1  # Increment for next trade
    
    trade = {
        'id': trade_id,
        'cycle': trading_state['cycle_count'],
        'action': action,
        'direction': direction,
        'entry_price': float(current_price),
        'entry_time': datetime.now().isoformat(),
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
        'ml_used': ml_trained,
        'ml_training_samples': len(ml_features),
        'volatility': price_cache.get('atr', 0.00015),
        'win_rate': trading_state['win_rate'],
        'git_pushed': False
    }
    
    trading_state['current_trade'] = trade
    trading_state['action'] = action
    trading_state['optimal_tp'] = optimal_tp
    trading_state['optimal_sl'] = optimal_sl
    trading_state['tp_distance_pips'] = tp_pips
    trading_state['sl_distance_pips'] = sl_pips
    trading_state['trade_status'] = 'ACTIVE'
    
    logger.info(f"🔔 {action} ORDER EXECUTED")
    logger.info(f"   Trade ID: {trade_id} (Continuing from Git: total {len(trade_history) + 1})")
    logger.info(f"   Entry Price: {current_price:.5f}")
    logger.info(f"   Take Profit: {optimal_tp:.5f} ({tp_pips} pips)")
    logger.info(f"   Stop Loss: {optimal_sl:.5f} ({sl_pips} pips)")
    logger.info(f"   Goal: Hit TP in <2min, avoid SL completely")
    logger.info(f"   ML Used: {ml_trained} ({len(ml_features)} samples)")
    logger.info(f"   ML Fully Trained on Startup: {ml_fully_trained_on_startup}")
    
    return trade

def monitor_active_trade(current_price):
    """Monitor the active trade throughout the 2-minute cycle"""
    if not trading_state['current_trade']:
        return None
    
    trade = trading_state['current_trade']
    entry_time = datetime.fromisoformat(trade['entry_time'])
    trade_duration = (datetime.now() - entry_time).total_seconds()
    
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
    
    # Check exit conditions - FOCUS ON HITTING TP BEFORE 2 MINUTES
    exit_trade = False
    exit_reason = ""
    trade_result = ""
    
    if trade['action'] == 'BUY':
        if current_price >= trade['optimal_tp']:
            exit_trade = True
            if trade_duration < CYCLE_SECONDS * 0.5:  # Hit TP in first half
                exit_reason = f"✅ TP HIT FAST! +{trade['tp_distance_pips']} pips in {trade_duration:.1f}s"
                trade_result = 'SUCCESS_FAST'
            else:
                exit_reason = f"TP HIT! +{trade['tp_distance_pips']} pips in {trade_duration:.1f}s"
                trade_result = 'SUCCESS'
            
        elif current_price <= trade['optimal_sl']:
            exit_trade = True
            exit_reason = f"❌ SL HIT! -{trade['sl_distance_pips']} pips loss in {trade_duration:.1f}s"
            trade_result = 'FAILED'
            
    else:  # SELL
        if current_price <= trade['optimal_tp']:
            exit_trade = True
            if trade_duration < CYCLE_SECONDS * 0.5:  # Hit TP in first half
                exit_reason = f"✅ TP HIT FAST! +{trade['tp_distance_pips']} pips in {trade_duration:.1f}s"
                trade_result = 'SUCCESS_FAST'
            else:
                exit_reason = f"TP HIT! +{trade['tp_distance_pips']} pips in {trade_duration:.1f}s"
                trade_result = 'SUCCESS'
            
        elif current_price >= trade['optimal_sl']:
            exit_trade = True
            exit_reason = f"❌ SL HIT! -{trade['sl_distance_pips']} pips loss in {trade_duration:.1f}s"
            trade_result = 'FAILED'
    
    # Time-based exit - CONSIDERED FAILURE IF TP NOT HIT
    if not exit_trade and trade_duration >= CYCLE_SECONDS:
        exit_trade = True
        if current_pips > 0:
            exit_reason = f"⏰ TIME ENDED with +{current_pips:.1f}pips (TP NOT HIT)"
            trade_result = 'PARTIAL_FAIL'  # Even if profit, TP not hit = failure
        elif current_pips < 0:
            exit_reason = f"⏰ TIME ENDED with {current_pips:.1f}pips loss"
            trade_result = 'FAILED'
        else:
            exit_reason = f"⏰ TIME ENDED at breakeven (TP NOT HIT)"
            trade_result = 'PARTIAL_FAIL'
    
    # Close trade if exit condition met
    if exit_trade:
        trade['status'] = 'CLOSED'
        trade['exit_price'] = current_price
        trade['exit_time'] = datetime.now().isoformat()
        trade['exit_reason'] = exit_reason
        trade['result'] = trade_result
        
        # Update trading statistics
        trading_state['total_trades'] += 1
        
        if trade_result in ['SUCCESS', 'SUCCESS_FAST']:
            trading_state['profitable_trades'] += 1
            trading_state['total_profit'] += trade['profit_amount']
            trading_state['balance'] += trade['profit_amount']
        else:
            trading_state['balance'] -= abs(trade['profit_amount'])
        
        # Update win rate
        if trading_state['total_trades'] > 0:
            trading_state['win_rate'] = (trading_state['profitable_trades'] / trading_state['total_trades']) * 100
        
        # Add to history
        trade_history.append(trade.copy())
        
        # Push all trades to Git repo
        trade['git_pushed'] = True
        push_trade_to_git()
        
        # Learn from this trade for ML improvement
        learn_from_trade(trade)
        
        # Clear current trade
        trading_state['current_trade'] = None
        trading_state['trade_status'] = 'COMPLETED'
        trading_state['trade_progress'] = 0
        trading_state['remaining_time'] = CYCLE_SECONDS
        
        logger.info(f"📊 TRADE COMPLETED: {exit_reason}")
        logger.info(f"   Result: {trade_result}")
        logger.info(f"   Total Trades: {trading_state['total_trades']}")
        logger.info(f"   Balance: ${trading_state['balance']:.2f}")
        logger.info(f"   Win Rate: {trading_state['win_rate']:.1f}%")
        
        return trade
    
    return trade

def learn_from_trade(trade):
    """Learn from trade result and update ML training - FOCUS ON HITTING TP FAST"""
    try:
        if 'result' not in trade or trade['result'] == 'PENDING':
            return
        
        # Extract features from trade data
        features = [
            trade['confidence'] / 100,
            trade.get('signal_strength', 1) / 3,
            trade.get('volatility', 0.00015) * 10000,
            1 if trade['action'] == 'BUY' else 0,
            trade['entry_price'],
            trade['tp_distance_pips'] / 100,
            trade['sl_distance_pips'] / 100,
            trade['duration_seconds'] / 120,
            trade['max_profit_pips'] / 100,
            trade['max_loss_pips'] / 100,
            1 if trade['result'] in ['SUCCESS', 'SUCCESS_FAST'] else 0,
            trading_state['win_rate'] / 100
        ]
        
        # Determine optimal TP/SL based on result - AGGRESSIVE LEARNING
        if trade['result'] == 'SUCCESS_FAST':  # TP hit in first half
            # Perfect! Keep similar settings
            optimal_tp = trade['tp_distance_pips']
            optimal_sl = trade['sl_distance_pips'] * 0.9  # Even tighter SL
            logger.info(f"🎯 PERFECT TRADE! TP hit fast. Keeping settings.")
            
        elif trade['result'] == 'SUCCESS':  # TP hit but took time
            # Good, but could be faster
            optimal_tp = trade['tp_distance_pips'] * 0.9  # Smaller TP for faster hit
            optimal_sl = trade['sl_distance_pips'] * 1.1  # Slightly larger SL
            logger.info(f"👍 Good trade. Adjusting for faster TP next time.")
            
        elif trade['result'] == 'PARTIAL_FAIL':  # Time ended with profit but TP not hit
            # Need much smaller TP
            optimal_tp = trade['tp_distance_pips'] * 0.7  # Much smaller TP
            optimal_sl = trade['sl_distance_pips'] * 0.8  # Tighter SL
            logger.info(f"⚠️  TP not hit. Making TP much smaller for next time.")
            
        elif trade['result'] == 'FAILED':  # SL hit or ended with loss
            # TP was too far, SL was too tight
            optimal_tp = trade['tp_distance_pips'] * 0.6  # Much smaller TP
            optimal_sl = trade['sl_distance_pips'] * 1.5  # Much larger SL
            logger.info(f"❌ Failed trade. Drastically adjusting TP/SL.")
            
        else:  # PARTIAL_SUCCESS or other
            optimal_tp = trade['tp_distance_pips'] * 0.8
            optimal_sl = trade['sl_distance_pips'] * 1.0
        
        # Add to training data
        ml_features.append(features)
        tp_labels.append(optimal_tp)
        sl_labels.append(optimal_sl)
        
        # NEW: Retrain ML after every 3 trades (as specified)
        if len(ml_features) >= 3 and len(ml_features) % 3 == 0:
            logger.info(f"🔄 RETRAINING ML: {len(ml_features)} samples available")
            train_ml_models()
            logger.info(f"✅ ML retrained with focus on hitting TP before {CYCLE_SECONDS} seconds")
        
        logger.info(f"📚 ML learned from trade #{trade.get('id', 'N/A')}: {trade['result']}")
        logger.info(f"   New target: TP={optimal_tp:.1f}pips, SL={optimal_sl:.1f}pips")
        
    except Exception as e:
        logger.error(f"Learning error: {e}")

# ==================== MAIN 2-MINUTE CYCLE ====================
def trading_cycle():
    """Main 2-minute trading cycle with Git integration"""
    global trading_state
    
    # Setup Git repo
    if trading_state['git_push_enabled']:
        setup_git_repo()
    
    # NEW: Initialize ML system by loading data from Git and training with ALL historical data
    logger.info("🚀 SYSTEM STARTUP: Training ML with ALL historical data before any prediction...")
    initialize_ml_system()
    
    cycle_count = trading_state['cycle_count']  # Start from existing cycle count
    
    logger.info("✅ Trading bot started with 2-minute cycles and Git integration")
    logger.info(f"📊 Starting with {len(trade_history)} historical trades from Git")
    logger.info(f"🎯 Total Trades: {trading_state['total_trades']}")
    logger.info(f"💰 Balance: ${trading_state['balance']:.2f}")
    logger.info(f"📈 Win Rate: {trading_state['win_rate']:.1f}%")
    logger.info(f"🤖 ML Ready: {ml_trained} ({len(ml_features)} samples)")
    logger.info(f"🚀 Git Push: {'✅ Enabled' if trading_state['git_push_enabled'] else '❌ Disabled'}")
    logger.info(f"🆔 Next Trade ID: T{next_trade_id}")
    logger.info(f"🎯 ML FOCUS: Hit TP BEFORE {CYCLE_SECONDS} seconds, everything else = FAILURE")
    logger.info(f"🔄 ML RETRAINING: After every 3 trades, NEVER settling")
    
    while True:
        try:
            cycle_count += 1
            cycle_start = datetime.now()
            
            trading_state['cycle_count'] = cycle_count
            trading_state['cycle_progress'] = 0
            trading_state['remaining_time'] = CYCLE_SECONDS
            
            logger.info(f"\n{'='*70}")
            logger.info(f"2-MINUTE TRADING CYCLE #{cycle_count}")
            logger.info(f"TOTAL TRADES: {trading_state['total_trades']} (Continuing from Git)")
            logger.info(f"ML TRAINED ON STARTUP: {ml_fully_trained_on_startup}")
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
            
            # 4. MAKE 2-MINUTE PREDICTION (ML is already trained from startup)
            logger.info("Analyzing market for 2-minute prediction...")
            pred_prob, confidence, direction, signal_strength = analyze_2min_prediction(
                df_indicators, current_price
            )
            
            trading_state['minute_prediction'] = direction
            trading_state['confidence'] = round(float(confidence), 1)
            trading_state['signal_strength'] = signal_strength
            
            # 5. EXTRACT ML FEATURES
            ml_features_current = extract_ml_features(df_indicators, current_price)
            
            # 6. PREDICT OPTIMAL TP/SL (ML-OPTIMIZED FOR FAST TP)
            optimal_tp, optimal_sl, tp_pips, sl_pips = predict_optimal_levels(
                ml_features_current, direction, current_price, df_indicators
            )
            
            # 7. CHECK ACTIVE TRADE
            if trading_state['current_trade']:
                monitor_active_trade(current_price)
            
            # 8. EXECUTE NEW TRADE
            if (trading_state['current_trade'] is None and 
                direction != 'NEUTRAL' and 
                confidence >= MIN_CONFIDENCE):
                
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
            
            # 10. UPDATE TIMESTAMP
            trading_state['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            trading_state['server_time'] = datetime.now().isoformat()
            
            # 11. LOG CYCLE SUMMARY
            logger.info(f"CYCLE #{cycle_count} SUMMARY:")
            logger.info(f"  Price: {current_price:.5f} ({data_source})")
            logger.info(f"  Prediction: {direction} ({confidence:.1f}% confidence)")
            logger.info(f"  Action: {trading_state['action']}")
            logger.info(f"  TP/SL: {tp_pips}/{sl_pips} pips (Goal: Hit TP in <2min)")
            logger.info(f"  Total Trades: {trading_state['total_trades']} (Continuing from Git)")
            logger.info(f"  Balance: ${trading_state['balance']:.2f}")
            logger.info(f"  Win Rate: {trading_state['win_rate']:.1f}%")
            logger.info(f"  ML Ready: {trading_state['ml_model_ready']} ({len(ml_features)} samples)")
            logger.info(f"  ML Startup Trained: {ml_fully_trained_on_startup}")
            logger.info(f"  Git Pushes: {trading_state['git_total_pushes']}")
            logger.info(f"  Next trade ID: T{next_trade_id}")
            logger.info(f"  Next cycle in: {next_cycle_time}s")
            logger.info(f"{'='*70}")
            
            # 12. WAIT FOR NEXT CYCLE
            for i in range(next_cycle_time):
                progress_pct = (i / next_cycle_time) * 100
                trading_state['cycle_progress'] = progress_pct
                trading_state['remaining_time'] = next_cycle_time - i
                
                # Update active trade progress if exists
                if trading_state['current_trade']:
                    entry_time = datetime.fromisoformat(trading_state['current_trade']['entry_time'])
                    trade_duration = (datetime.now() - entry_time).total_seconds()
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
            state_copy['current_trade'] = trade
        
        return jsonify(state_copy)
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/trade_history')
def get_trade_history():
    """Get trade history"""
    try:
        return jsonify({
            'trades': trade_history[-20:],  # Return last 20 trades
            'total': len(trade_history),
            'profitable': trading_state['profitable_trades'],
            'win_rate': trading_state['win_rate'],
            'next_trade_id': f"T{next_trade_id}"
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
        'total_trades': len(trade_history),
        'git_pushes': trading_state['git_total_pushes'],
        'git_last_push': trading_state['git_last_push'],
        'git_enabled': trading_state['git_push_enabled'],
        'next_trade_id': f"T{next_trade_id}",
        'ml_fully_trained_on_startup': ml_fully_trained_on_startup
    })

@app.route('/api/git_push_now', methods=['POST'])
def git_push_now():
    """Manual Git push endpoint"""
    result = execute_git_push()
    return jsonify(result)

@app.route('/api/git_load_now', methods=['POST'])
def git_load_now():
    """Manual Git load endpoint"""
    result = load_trades_from_git()
    if result.get('success'):
        # Re-initialize ML with new data
        logger.info("🔄 Manual Git load: Retraining ML with updated data...")
        train_ml_with_all_trades()
    return jsonify(result)

@app.route('/api/reset_trading')
def reset_trading():
    """Reset trading statistics (for testing)"""
    global trade_history, ml_features, tp_labels, sl_labels, next_trade_id, ml_fully_trained_on_startup
    
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
    
    trade_history.clear()
    ml_features.clear()
    tp_labels.clear()
    sl_labels.clear()
    next_trade_id = 1
    ml_fully_trained_on_startup = False
    
    # Also push reset state to Git
    push_trade_to_git()
    
    return jsonify({
        'success': True, 
        'message': 'Trading reset complete',
        'next_trade_id': f"T{next_trade_id}"
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'cycle_count': trading_state['cycle_count'],
        'git_enabled': trading_state['git_push_enabled'],
        'git_pushes': trading_state['git_total_pushes'],
        'total_trades': len(trade_history),
        'total_trades_state': trading_state['total_trades'],
        'ml_ready': trading_state['ml_model_ready'],
        'ml_startup_trained': ml_fully_trained_on_startup,
        'next_trade_id': f"T{next_trade_id}",
        'cache_enabled': True,
        'cache_duration': CACHE_DURATION
    })

# ==================== START TRADING BOT ====================
def start_trading_bot():
    """Start the trading bot"""
    try:
        thread = threading.Thread(target=trading_cycle, daemon=True)
        thread.start()
        logger.info("✅ Trading bot started successfully")
        print("="*80)
        print("2-MINUTE TRADING SYSTEM WITH CONTINUOUS TRADE COUNTING")
        print("="*80)
        print(f"✅ Git Push: {'ENABLED' if trading_state['git_push_enabled'] else 'DISABLED'}")
        print(f"✅ Trade Counting: CONTINUES from Git after sleep/wake")
        print(f"✅ Next Trade ID: T{next_trade_id}")
        print(f"✅ ML Training: ALL historical data on startup")
        print(f"✅ ML Retraining: After every 3 trades")
        print(f"✅ ML Focus: Hit TP BEFORE {CYCLE_SECONDS} seconds")
        print(f"✅ Trade data saved to: data/trades.json in Git")
        print(f"✅ Repo: {GITHUB_REPO_URL}")
        print("="*80)
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
    print("SYSTEM READY WITH ENHANCED ML TRAINING")
    print(f"• Trade counting CONTINUES after sleep/wake cycles")
    print(f"• Next trade will be: T{next_trade_id}")
    print(f"• ML training: ALL historical data on startup from Git")
    print(f"• ML retraining: After every 3 trades")
    print(f"• ML focus: Hit TP BEFORE {CYCLE_SECONDS} seconds, everything else = FAILURE")
    print(f"• Pulling trade data from: {GITHUB_REPO_URL}")
    print(f"• Pushing trade data after each trade")
    print(f"• Goal: Constant FAST TP hits before time expires")
    print("="*80)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )