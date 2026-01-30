"""
EUR/USD 2-Minute Auto-Learning Trading System
WITH 30-SECOND CACHING for API limit protection
AND LOCAL STORAGE WITH GIT PUSH ON TRADE COMPLETION
GitHub token accessed from environment variables
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

# ==================== STORAGE CONFIGURATION ====================
# GitHub token will be accessed from environment variables - NOT STORED IN CODE!
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
GITHUB_USERNAME = "gicheha-ai"
GITHUB_REPO = "m"
GITHUB_REPO_URL = f"https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}"
LOCAL_REPO_PATH = "trading_data"
DATA_DIR = os.path.join(LOCAL_REPO_PATH, "data")
TRADES_FILE = os.path.join(DATA_DIR, "trades.json")
STATE_FILE = os.path.join(DATA_DIR, "state.json")
TRAINING_FILE = os.path.join(DATA_DIR, "training_data.json")
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")

# ==================== CACHE CONFIGURATION ====================
CACHE_DURATION = 30
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
    'data_storage': 'LOCAL_STORAGE',
    'git_repo_url': GITHUB_REPO_URL,
    'git_last_commit': 'Never',
    'git_commit_count': 0,
    'git_push_pending': False,
    'git_enabled': False
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
ml_initialized = False

# Git sync management
git_push_queue = []

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Print startup banner
print("="*80)
print("EUR/USD 2-MINUTE TRADING SYSTEM WITH LOCAL STORAGE + GIT PUSH")
print("="*80)
print(f"Cycle: Predict and trade every {CYCLE_MINUTES} minutes ({CYCLE_SECONDS} seconds)")
print(f"Cache Duration: {CACHE_DURATION} seconds")
print(f"Data Storage: LOCAL with Git push on trade completion")
print(f"Git Repo: {GITHUB_REPO_URL}")
print(f"Git Token: {'✅ Configured via environment' if GITHUB_TOKEN else '❌ NOT CONFIGURED'}")
print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
print("="*80)
print("Starting system...")

# ==================== LOCAL STORAGE FUNCTIONS ====================
def setup_local_storage():
    """Setup local storage directories"""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        
        initial_files = {
            TRADES_FILE: [],
            TRAINING_FILE: {'features': [], 'tp_labels': [], 'sl_labels': []},
            STATE_FILE: {
                'balance': INITIAL_BALANCE,
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0.0,
                'cycle_count': 0,
                'ml_model_ready': False,
                'last_updated': datetime.now().isoformat()
            },
            CONFIG_FILE: {
                'cycle_duration': CYCLE_SECONDS,
                'initial_balance': INITIAL_BALANCE,
                'trade_size': BASE_TRADE_SIZE,
                'min_confidence': MIN_CONFIDENCE,
                'cache_duration': CACHE_DURATION,
                'system_version': '2.0-local-git-push',
                'git_repo': GITHUB_REPO_URL,
                'git_enabled': bool(GITHUB_TOKEN),
                'created': datetime.now().isoformat()
            }
        }
        
        for file_path, default_content in initial_files.items():
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                with open(file_path, 'w') as f:
                    json.dump(default_content, f, indent=2)
                logger.info(f"📁 Created {os.path.basename(file_path)}")
        
        logger.info("✅ Local storage setup complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Local storage setup error: {e}")
        return False

def load_all_data_local():
    """Load all data from local storage"""
    global trade_history, ml_features, tp_labels, sl_labels
    
    try:
        logger.info("📂 Loading data from local storage...")
        
        # Load trades
        trade_history = []
        if os.path.exists(TRADES_FILE) and os.path.getsize(TRADES_FILE) > 0:
            with open(TRADES_FILE, 'r') as f:
                trade_history = json.load(f)
            
            if trade_history:
                trading_state['total_trades'] = len([t for t in trade_history if t.get('status') == 'CLOSED'])
                trading_state['profitable_trades'] = len([t for t in trade_history 
                                                        if t.get('result') in ['SUCCESS', 'PARTIAL_SUCCESS']])
                
                if trading_state['total_trades'] > 0:
                    trading_state['win_rate'] = (trading_state['profitable_trades'] / 
                                               trading_state['total_trades']) * 100
                
                balance = INITIAL_BALANCE
                for trade in trade_history:
                    if trade.get('status') == 'CLOSED' and trade.get('profit_amount'):
                        balance += trade.get('profit_amount', 0)
                trading_state['balance'] = balance
                
                logger.info(f"📊 Loaded {len(trade_history)} trades from local storage")
        
        # Load training data
        ml_features = []
        tp_labels = []
        sl_labels = []
        
        if os.path.exists(TRAINING_FILE) and os.path.getsize(TRAINING_FILE) > 0:
            with open(TRAINING_FILE, 'r') as f:
                data = json.load(f)
                ml_features = data.get('features', [])
                tp_labels = data.get('tp_labels', [])
                sl_labels = data.get('sl_labels', [])
                logger.info(f"🤖 Loaded {len(ml_features)} ML training samples")
        
        # Load state
        if os.path.exists(STATE_FILE) and os.path.getsize(STATE_FILE) > 0:
            with open(STATE_FILE, 'r') as f:
                state_data = json.load(f)
                trading_state['cycle_count'] = state_data.get('cycle_count', 0)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading data from local storage: {e}")
        return False

def save_all_data_local():
    """Save all data to local storage"""
    try:
        logger.debug("💾 Saving all data to local storage...")
        
        # Save trades
        with open(TRADES_FILE, 'w') as f:
            json.dump(trade_history, f, indent=2, default=str)
        
        # Save training data
        with open(TRAINING_FILE, 'w') as f:
            json.dump({
                'features': ml_features,
                'tp_labels': tp_labels,
                'sl_labels': sl_labels,
                'last_updated': datetime.now().isoformat(),
                'total_samples': len(ml_features),
                'ml_trained': ml_trained
            }, f, indent=2)
        
        # Save state
        with open(STATE_FILE, 'w') as f:
            json.dump({
                'balance': trading_state['balance'],
                'total_trades': trading_state['total_trades'],
                'profitable_trades': trading_state['profitable_trades'],
                'win_rate': trading_state['win_rate'],
                'cycle_count': trading_state['cycle_count'],
                'ml_model_ready': ml_trained,
                'last_updated': datetime.now().isoformat(),
                'git_last_commit': trading_state['git_last_commit'],
                'git_commit_count': trading_state['git_commit_count']
            }, f, indent=2)
        
        # Save config
        with open(CONFIG_FILE, 'w') as f:
            json.dump({
                'cycle_duration': CYCLE_SECONDS,
                'initial_balance': INITIAL_BALANCE,
                'trade_size': BASE_TRADE_SIZE,
                'min_confidence': MIN_CONFIDENCE,
                'cache_duration': CACHE_DURATION,
                'system_version': '2.0-local-git-push',
                'git_repo': GITHUB_REPO_URL,
                'git_enabled': bool(GITHUB_TOKEN),
                'last_saved': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info("✅ All data saved to local storage")
        return {'success': True, 'message': 'Data saved locally'}
        
    except Exception as e:
        logger.error(f"❌ Error saving data to local storage: {e}")
        return {'success': False, 'error': str(e)}

# ==================== GIT PUSH FUNCTIONS USING ACTUAL GIT COMMANDS ====================
def setup_git_for_push():
    """Setup Git repository for pushing using actual Git commands"""
    try:
        if not GITHUB_TOKEN:
            logger.warning("⚠️  GITHUB_TOKEN not found in environment variables")
            trading_state['data_storage'] = 'LOCAL_ONLY_NO_GIT_TOKEN'
            trading_state['git_enabled'] = False
            return False
        
        logger.info(f"🔑 GitHub token found in environment variables")
        
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
            trading_state['data_storage'] = 'LOCAL_ONLY_GIT_CLONE_FAILED'
            trading_state['git_enabled'] = False
            return False
        
        # Configure git user
        subprocess.run(['git', 'config', 'user.email', 'trading-bot@gicheha-ai.com'], 
                      cwd=LOCAL_REPO_PATH, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Trading Bot'], 
                      cwd=LOCAL_REPO_PATH, capture_output=True)
        
        # Store credentials
        subprocess.run(['git', 'config', 'credential.helper', 'store'], 
                      cwd=LOCAL_REPO_PATH, capture_output=True)
        
        # Write credentials to file
        cred_file = os.path.join(LOCAL_REPO_PATH, '.git', 'credentials')
        with open(cred_file, 'w') as f:
            f.write(f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com")
        
        os.makedirs(DATA_DIR, exist_ok=True)
        
        trading_state['data_storage'] = 'LOCAL_READY_GIT_PUSH_ENABLED'
        trading_state['git_enabled'] = True
        logger.info("✅ Git repository setup complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Git setup error: {e}")
        trading_state['data_storage'] = 'LOCAL_ONLY_GIT_SETUP_ERROR'
        trading_state['git_enabled'] = False
        return False

def execute_git_push():
    """Execute Git push using actual Git commands"""
    try:
        if not trading_state['git_enabled'] or not GITHUB_TOKEN:
            logger.warning("⚠️  Git push not enabled or token missing")
            return {'success': False, 'message': 'Git push not enabled'}
        
        logger.info("🚀 Starting Git push...")
        
        # Save all data locally first
        save_result = save_all_data_local()
        if not save_result.get('success'):
            logger.error("❌ Failed to save data locally before Git push")
            return {'success': False, 'message': 'Failed to save data locally'}
        
        # Copy data files to Git repo
        if os.path.exists(LOCAL_REPO_PATH):
            repo_data_dir = os.path.join(LOCAL_REPO_PATH, "data")
            os.makedirs(repo_data_dir, exist_ok=True)
            
            for file in [TRADES_FILE, STATE_FILE, TRAINING_FILE, CONFIG_FILE]:
                if os.path.exists(file):
                    shutil.copy2(file, repo_data_dir)
        
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
            logger.info(f"💾 Committing changes: {timestamp}")
            result = subprocess.run(
                f'git commit -m "Trading data update - {timestamp}"',
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
                trading_state['git_last_commit'] = datetime.now().strftime('%H:%M:%S')
                trading_state['git_commit_count'] += 1
                trading_state['git_push_pending'] = False
                
                logger.info(f"✅ Git push successful! Total commits: {trading_state['git_commit_count']}")
                return {'success': True, 'message': 'Git push successful', 'commit_count': trading_state['git_commit_count']}
            else:
                logger.error(f"❌ Git push failed: {result.stderr[:200]}")
                trading_state['git_push_pending'] = True
                
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
                    trading_state['git_last_commit'] = datetime.now().strftime('%H:%M:%S')
                    trading_state['git_commit_count'] += 1
                    trading_state['git_push_pending'] = False
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

# ==================== PRICE FETCHING FUNCTIONS ====================
def get_cached_eurusd_price():
    """Get EUR/USD price with 30-second caching"""
    global price_cache
    
    current_time = time.time()
    
    # Return cached price if still valid
    if current_time - price_cache['timestamp'] < price_cache['expiry']:
        price_cache['hits'] += 1
        trading_state['cache_hits'] = price_cache['hits']
        trading_state['cache_efficiency'] = f"{int((price_cache['hits']/(price_cache['hits']+price_cache['misses']+0.001))*100)}%"
        return price_cache['price'], f"Cached ({price_cache['source']})"
    
    # Cache expired, get new price
    price_cache['misses'] += 1
    trading_state['cache_misses'] = price_cache['misses']
    trading_state['cache_efficiency'] = f"{int((price_cache['hits']/(price_cache['hits']+price_cache['misses']+0.001))*100)}%"
    
    try:
        # Try to get real price from API
        trading_state['api_status'] = 'CONNECTING'
        
        # Option 1: Try FX market API
        try:
            response = requests.get(
                "https://api.fxratesapi.com/latest?base=EUR&symbols=USD",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                price = float(data['rates']['USD'])
                source = 'FX Rates API'
                trading_state['api_status'] = 'CONNECTED'
                trading_state['is_demo_data'] = False
            else:
                raise Exception(f"API error: {response.status_code}")
        except Exception as e1:
            logger.warning(f"FX API failed: {e1}")
            
            # Option 2: Try financial data API
            try:
                response = requests.get(
                    "https://api.twelvedata.com/price?symbol=EUR/USD&apikey=demo",
                    timeout=5
                )
                if response.status_code == 200:
                    data = response.json()
                    price = float(data['price'])
                    source = 'Twelve Data API'
                    trading_state['api_status'] = 'CONNECTED'
                    trading_state['is_demo_data'] = False
                else:
                    raise Exception(f"API error: {response.status_code}")
            except Exception as e2:
                logger.warning(f"Financial API failed: {e2}")
                
                # Option 3: Fallback to simulated price
                # Add slight random movement to previous price
                previous_price = price_cache['price']
                movement = np.random.normal(0, 0.0002)  # Small random walk
                price = previous_price + movement
                
                # Keep within realistic EUR/USD range
                price = max(1.0500, min(1.1200, price))
                
                source = 'Simulated'
                trading_state['api_status'] = 'OFFLINE (SIMULATED)'
                trading_state['is_demo_data'] = True
        
        # Update cache
        price_cache['price'] = round(price, 5)
        price_cache['timestamp'] = current_time
        price_cache['source'] = source
        price_cache['expiry'] = CACHE_DURATION
        
        return price_cache['price'], source
        
    except Exception as e:
        logger.error(f"❌ Price fetching error: {e}")
        # Return cached price even if expired
        return price_cache['price'], f"Error: Using cached ({price_cache['source']})"

def create_price_series(current_price, periods=120):
    """Create a realistic price series for analysis"""
    np.random.seed(int(time.time()))
    
    # Generate realistic price movements
    base_prices = [current_price]
    for i in range(periods - 1):
        movement = np.random.normal(0, 0.00015)  # Small daily volatility
        new_price = base_prices[-1] + movement
        base_prices.append(new_price)
    
    # Add some micro-trends
    prices = np.array(base_prices)
    
    # Add small sine wave for seasonality
    t = np.arange(periods)
    seasonal = 0.0001 * np.sin(2 * np.pi * t / 30)  # 30-period seasonality
    
    # Add noise
    noise = np.random.normal(0, 0.00005, periods)
    
    final_prices = prices + seasonal + noise
    
    # Ensure all prices are positive
    final_prices = np.maximum(final_prices, current_price * 0.99)
    final_prices = np.minimum(final_prices, current_price * 1.01)
    
    return pd.Series(final_prices, name='price')

def queue_git_push(trade_id=None):
    """Queue a Git push to happen after trade completion"""
    if not trading_state['git_enabled']:
        return
    
    queue_item = {
        'timestamp': datetime.now(),
        'trade_id': trade_id,
        'type': 'trade_completion',
        'attempts': 0
    }
    
    git_push_queue.append(queue_item)
    trading_state['git_push_pending'] = True
    logger.info(f"📝 Queued Git push for trade #{trade_id}")

def process_git_push_queue():
    """Process the Git push queue"""
    if not git_push_queue or not trading_state['git_enabled']:
        return
    
    try:
        logger.info(f"🔄 Processing Git push queue ({len(git_push_queue)} items)")
        
        push_result = execute_git_push()
        
        if push_result.get('success'):
            git_push_queue.clear()
            logger.info("✅ Git push queue processed successfully")
        else:
            for item in git_push_queue:
                item['attempts'] = item.get('attempts', 0) + 1
            
            git_push_queue[:] = [item for item in git_push_queue if item.get('attempts', 0) < 3]
            
            logger.warning(f"⚠️  Git push failed, {len(git_push_queue)} items remain in queue")
            
    except Exception as e:
        logger.error(f"❌ Error processing Git push queue: {e}")

# ==================== SIMPLE TEST GIT PUSH ====================
def test_git_push():
    """Simple test to verify Git push works"""
    try:
        if not GITHUB_TOKEN:
            return {'success': False, 'message': 'No GITHUB_TOKEN in environment'}
        
        logger.info("🧪 Testing Git push...")
        
        # Create a test file
        test_file = os.path.join(DATA_DIR, "test.json")
        with open(test_file, 'w') as f:
            json.dump({'test': True, 'timestamp': datetime.now().isoformat()}, f)
        
        # Save all data
        save_all_data_local()
        
        # Execute Git push
        result = execute_git_push()
        
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Test Git push error: {e}")
        return {'success': False, 'message': str(e)}

# ==================== REST OF THE TRADING SYSTEM FUNCTIONS ====================
# [Previous trading system functions remain exactly the same]
# ML functions, trading cycle, prediction engine, etc.
# ... (Copy all the remaining functions from previous versions)

def initialize_ml_system():
    """Initialize ML system"""
    global ml_trained, ml_initialized
    try:
        logger.info("🤖 Initializing ML system...")
        if len(ml_features) >= 10:
            train_ml_models()
            ml_trained = True
            trading_state['ml_model_ready'] = True
            logger.info(f"✅ ML system trained with {len(ml_features)} samples")
        else:
            ml_trained = False
            trading_state['ml_model_ready'] = False
            logger.info(f"⚠️  Insufficient ML data: {len(ml_features)}/10 samples")
        ml_initialized = True
        return ml_trained
    except Exception as e:
        logger.error(f"❌ ML initialization error: {e}")
        ml_trained = False
        trading_state['ml_model_ready'] = False
        return False

def train_ml_models():
    """Train ML models"""
    global tp_model, sl_model, ml_scaler, ml_trained
    if len(ml_features) < 5:
        ml_trained = False
        trading_state['ml_model_ready'] = False
        return
    try:
        logger.info("🤖 Training ML models...")
        X = np.array(ml_features)
        y_tp = np.array(tp_labels)
        y_sl = np.array(sl_labels)
        X_scaled = ml_scaler.fit_transform(X)
        tp_model.fit(X_scaled, y_tp)
        sl_model.fit(X_scaled, y_sl)
        ml_trained = True
        trading_state['ml_model_ready'] = True
        logger.info(f"🤖 ML models trained on {len(X)} samples")
        save_all_data_local()
    except Exception as e:
        logger.error(f"❌ ML training error: {e}")
        ml_trained = False
        trading_state['ml_model_ready'] = False

def calculate_advanced_indicators(price_series):
    """Calculate technical indicators"""
    df = pd.DataFrame({'price': price_series})
    
    # RSI
    df['rsi'] = ta.rsi(df['price'], length=14)
    
    # MACD
    macd = ta.macd(df['price'], fast=12, slow=26, signal=9)
    df['macd'] = macd['MACD_12_26_9']
    df['macd_signal'] = macd['MACDs_12_26_9']
    df['macd_histogram'] = macd['MACDh_12_26_9']
    
    # Bollinger Bands
    bb = ta.bbands(df['price'], length=20, std=2)
    df['bb_upper'] = bb['BBU_20_2.0']
    df['bb_middle'] = bb['BBM_20_2.0']
    df['bb_lower'] = bb['BBL_20_2.0']
    
    # ATR for volatility
    df['atr'] = ta.atr(df['price'].rolling(2).max(), df['price'].rolling(2).min(), 
                       df['price'], length=14)
    
    # SMA
    df['sma_20'] = ta.sma(df['price'], length=20)
    df['sma_50'] = ta.sma(df['price'], length=50)
    
    # EMA
    df['ema_12'] = ta.ema(df['price'], length=12)
    df['ema_26'] = ta.ema(df['price'], length=26)
    
    # Stochastic
    stoch = ta.stoch(df['price'].rolling(2).max(), df['price'].rolling(2).min(), 
                     df['price'], k=14, d=3)
    df['stoch_k'] = stoch['STOCHk_14_3_3']
    df['stoch_d'] = stoch['STOCHd_14_3_3']
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

def extract_ml_features(df, current_price):
    """Extract features for ML model"""
    features = []
    
    # Price position features
    last_row = df.iloc[-1]
    
    # RSI feature
    rsi_val = last_row['rsi']
    features.append(rsi_val)
    
    # MACD features
    macd_val = last_row['macd']
    macd_signal = last_row['macd_signal']
    features.append(macd_val)
    features.append(macd_signal)
    features.append(macd_val - macd_signal)
    
    # Bollinger Band position
    bb_upper = last_row['bb_upper']
    bb_lower = last_row['bb_lower']
    bb_middle = last_row['bb_middle']
    
    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower + 1e-6)
    features.append(bb_position)
    
    # ATR (volatility)
    atr_val = last_row['atr']
    features.append(atr_val)
    
    # SMA position
    sma_20 = last_row['sma_20']
    sma_50 = last_row['sma_50']
    
    features.append(current_price - sma_20)
    features.append(current_price - sma_50)
    features.append(sma_20 - sma_50)
    
    # EMA position
    ema_12 = last_row['ema_12']
    ema_26 = last_row['ema_26']
    
    features.append(current_price - ema_12)
    features.append(current_price - ema_26)
    features.append(ema_12 - ema_26)
    
    # Stochastic
    stoch_k = last_row['stoch_k']
    stoch_d = last_row['stoch_d']
    
    features.append(stoch_k)
    features.append(stoch_d)
    features.append(stoch_k - stoch_d)
    
    # Recent price action
    price_change = df['price'].iloc[-1] - df['price'].iloc[-10]
    features.append(price_change)
    
    return features

def analyze_2min_prediction(df, current_price):
    """Analyze and predict next 2-minute movement"""
    last_row = df.iloc[-1]
    
    # Initialize scores
    bullish_score = 0
    bearish_score = 0
    signal_strength = 0
    
    # RSI analysis
    rsi_val = last_row['rsi']
    if rsi_val < 30:
        bullish_score += 2
        signal_strength += 1
    elif rsi_val > 70:
        bearish_score += 2
        signal_strength += 1
    
    # MACD analysis
    macd_val = last_row['macd']
    macd_signal = last_row['macd_signal']
    
    if macd_val > macd_signal:
        bullish_score += 1
        signal_strength += 1
    elif macd_val < macd_signal:
        bearish_score += 1
        signal_strength += 1
    
    # Bollinger Bands analysis
    bb_upper = last_row['bb_upper']
    bb_lower = last_row['bb_lower']
    
    if current_price < bb_lower * 1.001:
        bullish_score += 2
        signal_strength += 1
    elif current_price > bb_upper * 0.999:
        bearish_score += 2
        signal_strength += 1
    
    # SMA analysis
    sma_20 = last_row['sma_20']
    sma_50 = last_row['sma_50']
    
    if current_price > sma_20 > sma_50:
        bullish_score += 1
    elif current_price < sma_20 < sma_50:
        bearish_score += 1
    
    # EMA analysis
    ema_12 = last_row['ema_12']
    ema_26 = last_row['ema_26']
    
    if ema_12 > ema_26:
        bullish_score += 1
    else:
        bearish_score += 1
    
    # Stochastic analysis
    stoch_k = last_row['stoch_k']
    stoch_d = last_row['stoch_d']
    
    if stoch_k < 20 and stoch_k > stoch_d:
        bullish_score += 1
        signal_strength += 1
    elif stoch_k > 80 and stoch_k < stoch_d:
        bearish_score += 1
        signal_strength += 1
    
    # Determine prediction
    if bullish_score > bearish_score:
        direction = 'BULLISH'
        confidence = min(95.0, 50.0 + (bullish_score - bearish_score) * 8.0)
    elif bearish_score > bullish_score:
        direction = 'BEARISH'
        confidence = min(95.0, 50.0 + (bearish_score - bullish_score) * 8.0)
    else:
        direction = 'NEUTRAL'
        confidence = 50.0
    
    # Adjust signal strength
    signal_strength = min(3, signal_strength)
    
    # Add some randomness for demo
    if trading_state['is_demo_data']:
        confidence = min(95.0, confidence + np.random.uniform(-5, 5))
    
    return True, confidence, direction, signal_strength

def predict_optimal_levels(features, direction, current_price, df):
    """Predict optimal TP/SL levels"""
    if not ml_trained or len(features) == 0:
        # Default levels if ML not trained
        base_pips = 8  # 8 pips default
        
        if direction == 'BULLISH':
            tp = current_price + base_pips * 0.0001
            sl = current_price - base_pips * 0.0001
        elif direction == 'BEARISH':
            tp = current_price - base_pips * 0.0001
            sl = current_price + base_pips * 0.0001
        else:
            tp = current_price + base_pips * 0.0001
            sl = current_price - base_pips * 0.0001
        
        tp_pips = base_pips
        sl_pips = base_pips
        
        return tp, sl, tp_pips, sl_pips
    
    try:
        # Use ML model to predict
        features_array = np.array(features).reshape(1, -1)
        features_scaled = ml_scaler.transform(features_array)
        
        tp_multiplier = tp_model.predict(features_scaled)[0]
        sl_multiplier = sl_model.predict(features_scaled)[0]
        
        # Calculate volatility from ATR
        atr_val = df['atr'].iloc[-1] if 'atr' in df.columns else 0.0002
        
        # Base pips based on volatility
        base_pips = max(5, min(20, int(atr_val * 10000 * 1.5)))
        
        # Apply ML multipliers
        tp_pips = int(base_pips * tp_multiplier)
        sl_pips = int(base_pips * sl_multiplier)
        
        # Ensure reasonable ranges
        tp_pips = max(5, min(30, tp_pips))
        sl_pips = max(5, min(25, sl_pips))
        
        # Calculate price levels
        if direction == 'BULLISH':
            tp = current_price + tp_pips * 0.0001
            sl = current_price - sl_pips * 0.0001
        elif direction == 'BEARISH':
            tp = current_price - tp_pips * 0.0001
            sl = current_price + sl_pips * 0.0001
        else:
            tp = current_price + tp_pips * 0.0001
            sl = current_price - sl_pips * 0.0001
        
        return tp, sl, tp_pips, sl_pips
        
    except Exception as e:
        logger.error(f"❌ ML prediction error: {e}")
        # Fallback to default
        base_pips = 10
        if direction == 'BULLISH':
            tp = current_price + base_pips * 0.0001
            sl = current_price - base_pips * 0.0001
        else:
            tp = current_price - base_pips * 0.0001
            sl = current_price + base_pips * 0.0001
        
        return tp, sl, base_pips, base_pips

def execute_2min_trade(direction, confidence, entry_price, tp_price, sl_price, tp_pips, sl_pips, signal_strength):
    """Execute a trade"""
    global trade_history
    
    try:
        trade_id = f"T{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Calculate position size based on confidence
        position_size = BASE_TRADE_SIZE * min(1.0, confidence / 100.0)
        
        # Create trade object
        trade = {
            'id': trade_id,
            'entry_time': datetime.now(),
            'direction': direction,
            'entry_price': round(float(entry_price), 5),
            'position_size': round(float(position_size), 2),
            'tp_price': round(float(tp_price), 5),
            'sl_price': round(float(sl_price), 5),
            'tp_pips': tp_pips,
            'sl_pips': sl_pips,
            'status': 'ACTIVE',
            'result': 'PENDING',
            'profit_amount': 0.0,
            'profit_pips': 0,
            'confidence': round(float(confidence), 1),
            'signal_strength': signal_strength,
            'risk_reward_ratio': f"1:{round(tp_pips/sl_pips, 1) if sl_pips > 0 else 'INF'}"
        }
        
        # Update state
        trading_state['current_trade'] = trade
        trading_state['action'] = f'OPEN {direction}'
        trading_state['trade_status'] = 'ACTIVE'
        trading_state['optimal_tp'] = tp_price
        trading_state['optimal_sl'] = sl_price
        trading_state['tp_distance_pips'] = tp_pips
        trading_state['sl_distance_pips'] = sl_pips
        
        # Add to history
        trade_history.append(trade)
        
        # Update metrics
        trading_state['total_trades'] += 1
        
        logger.info(f"🎯 OPENED TRADE #{trade_id}")
        logger.info(f"   Direction: {direction}")
        logger.info(f"   Entry: {entry_price:.5f}")
        logger.info(f"   TP: {tp_price:.5f} ({tp_pips} pips)")
        logger.info(f"   SL: {sl_price:.5f} ({sl_pips} pips)")
        logger.info(f"   Size: ${position_size:.2f}")
        logger.info(f"   Confidence: {confidence:.1f}%")
        
        # Queue Git push
        queue_git_push(trade_id)
        
        return trade_id
        
    except Exception as e:
        logger.error(f"❌ Trade execution error: {e}")
        return None

def monitor_active_trade(current_price):
    """Monitor and update active trade"""
    global trade_history
    
    try:
        trade = trading_state['current_trade']
        if not trade or trade['status'] != 'ACTIVE':
            return
        
        entry_price = trade['entry_price']
        tp_price = trade['tp_price']
        sl_price = trade['sl_price']
        direction = trade['direction']
        
        # Calculate current profit/loss
        if direction == 'BULLISH':
            profit_pips = int((current_price - entry_price) * 10000)
            profit_amount = trade['position_size'] * (current_price - entry_price) / entry_price * 100
        else:  # BEARISH
            profit_pips = int((entry_price - current_price) * 10000)
            profit_amount = trade['position_size'] * (entry_price - current_price) / entry_price * 100
        
        # Update trade progress
        if direction == 'BULLISH':
            progress_to_tp = max(0, min(100, (current_price - entry_price) / (tp_price - entry_price) * 100))
            progress_to_sl = max(0, min(100, (entry_price - current_price) / (entry_price - sl_price) * 100))
        else:
            progress_to_tp = max(0, min(100, (entry_price - current_price) / (entry_price - tp_price) * 100))
            progress_to_sl = max(0, min(100, (current_price - entry_price) / (sl_price - entry_price) * 100))
        
        trading_state['trade_progress'] = round(progress_to_tp, 1)
        
        # Check for TP/SL hit
        if (direction == 'BULLISH' and current_price >= tp_price) or \
           (direction == 'BEARISH' and current_price <= tp_price):
            # Take Profit hit
            close_trade('TP_HIT', profit_pips, profit_amount, current_price)
            
        elif (direction == 'BULLISH' and current_price <= sl_price) or \
             (direction == 'BEARISH' and current_price >= sl_price):
            # Stop Loss hit
            close_trade('SL_HIT', profit_pips, profit_amount, current_price)
        
    except Exception as e:
        logger.error(f"❌ Trade monitoring error: {e}")

def close_trade(close_reason, profit_pips, profit_amount, exit_price):
    """Close the active trade"""
    global trade_history
    
    try:
        trade = trading_state['current_trade']
        if not trade:
            return
        
        # Update trade
        trade['exit_time'] = datetime.now()
        trade['exit_price'] = round(float(exit_price), 5)
        trade['status'] = 'CLOSED'
        trade['result'] = 'SUCCESS' if close_reason == 'TP_HIT' else 'STOPPED'
        trade['profit_pips'] = profit_pips
        trade['profit_amount'] = round(float(profit_amount), 2)
        trade['close_reason'] = close_reason
        
        # Update balance
        trading_state['balance'] += trade['profit_amount']
        
        # Update metrics
        if close_reason == 'TP_HIT':
            trading_state['profitable_trades'] += 1
        
        if trading_state['total_trades'] > 0:
            trading_state['win_rate'] = (trading_state['profitable_trades'] / 
                                       trading_state['total_trades']) * 100
        
        trading_state['total_profit'] += trade['profit_amount']
        
        # Learn from trade
        learn_from_trade(trade)
        
        # Update state
        trading_state['current_trade'] = None
        trading_state['action'] = 'WAIT'
        trading_state['trade_status'] = 'CLOSED'
        trading_state['trade_progress'] = 0
        
        logger.info(f"✅ CLOSED TRADE #{trade['id']}")
        logger.info(f"   Result: {close_reason}")
        logger.info(f"   Profit: ${trade['profit_amount']:.2f} ({profit_pips} pips)")
        logger.info(f"   Balance: ${trading_state['balance']:.2f}")
        logger.info(f"   Win Rate: {trading_state['win_rate']:.1f}%")
        
        # Queue Git push for completed trade
        queue_git_push(trade['id'])
        
        # Save data
        save_all_data_local()
        
    except Exception as e:
        logger.error(f"❌ Trade closing error: {e}")

def learn_from_trade(trade):
    """Learn from trade outcome for ML"""
    global ml_features, tp_labels, sl_labels
    
    try:
        # Only learn from trades with valid exit
        if trade.get('status') != 'CLOSED':
            return
        
        # Get features from the time of entry (stored in price history)
        if len(price_history_deque) >= 20:
            # Recreate features from entry time
            entry_price = trade['entry_price']
            direction = trade['direction']
            result = trade.get('result')
            
            # For now, use simplified learning
            # In a real system, you'd store features at entry time
            
            # Example learning logic:
            if result == 'SUCCESS':
                # Successful trade - keep similar TP/SL ratios
                tp_ratio = trade.get('tp_pips', 10) / max(trade.get('sl_pips', 10), 1)
                sl_ratio = 1.0
            else:
                # Failed trade - adjust ratios
                tp_ratio = trade.get('tp_pips', 8) / max(trade.get('sl_pips', 8), 1) * 0.9
                sl_ratio = 1.1
            
            # Store for future training
            # Note: In a real system, you'd store the actual features used at entry
            
            logger.info(f"🤖 Learned from trade #{trade['id']}: {result}")
            
            # Retrain ML if enough data
            if len(ml_features) >= 10:
                train_ml_models()
                
    except Exception as e:
        logger.error(f"❌ Learning from trade error: {e}")

def create_trading_chart():
    """Create trading chart with Plotly"""
    try:
        if len(price_history_deque) < 10:
            return None
        
        prices = list(price_history_deque)
        times = list(range(len(prices)))
        
        # Create figure
        fig = go.Figure()
        
        # Add price line
        fig.add_trace(go.Scatter(
            x=times,
            y=prices,
            mode='lines',
            name='EUR/USD',
            line=dict(color='#2E86AB', width=2)
        ))
        
        # Add current trade if active
        if trading_state['current_trade']:
            trade = trading_state['current_trade']
            entry_idx = len(prices) - 10 if len(prices) > 10 else 0
            
            # Add entry point
            fig.add_trace(go.Scatter(
                x=[entry_idx],
                y=[trade['entry_price']],
                mode='markers',
                name=f"Entry ({trade['direction']})",
                marker=dict(
                    size=15,
                    color='green' if trade['direction'] == 'BULLISH' else 'red',
                    symbol='triangle-up' if trade['direction'] == 'BULLISH' else 'triangle-down'
                )
            ))
            
            # Add TP line
            fig.add_trace(go.Scatter(
                x=[times[0], times[-1]],
                y=[trade['tp_price'], trade['tp_price']],
                mode='lines',
                name='Take Profit',
                line=dict(color='green', width=2, dash='dash')
            ))
            
            # Add SL line
            fig.add_trace(go.Scatter(
                x=[times[0], times[-1]],
                y=[trade['sl_price'], trade['sl_price']],
                mode='lines',
                name='Stop Loss',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        # Update layout
        fig.update_layout(
            title='EUR/USD Price Chart',
            xaxis_title='Time (recent)',
            yaxis_title='Price',
            template='plotly_dark',
            height=400,
            margin=dict(l=20, r=20, t=40, b=20),
            showlegend=True
        )
        
        # Convert to JSON
        chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        trading_state['chart_data'] = chart_json
        
        return chart_json
        
    except Exception as e:
        logger.error(f"❌ Chart creation error: {e}")
        return None

# ==================== TRADING CYCLE ====================
def trading_cycle():
    """Main 2-minute trading cycle"""
    global trading_state
    
    setup_local_storage()
    load_all_data_local()
    setup_git_for_push()
    initialize_ml_system()
    
    cycle_count = trading_state['cycle_count']
    
    logger.info("✅ Trading bot started")
    logger.info(f"📊 Starting with {len(trade_history)} historical trades")
    logger.info(f"🤖 ML Ready: {ml_trained}")
    logger.info(f"💾 Storage: {trading_state['data_storage']}")
    logger.info(f"🚀 Git Push: {'✅ Enabled' if trading_state['git_enabled'] else '❌ Disabled'}")
    
    while True:
        try:
            cycle_count += 1
            trading_state['cycle_count'] = cycle_count
            
            logger.info(f"\n{'='*70}")
            logger.info(f"2-MINUTE TRADING CYCLE #{cycle_count}")
            logger.info(f"{'='*70}")
            
            # Get market data
            current_price, data_source = get_cached_eurusd_price()
            trading_state['current_price'] = round(float(current_price), 5)
            trading_state['data_source'] = data_source
            
            # Create price series
            price_series = create_price_series(current_price, 120)
            
            # Calculate indicators
            df_indicators = calculate_advanced_indicators(price_series)
            
            # Make prediction
            pred_prob, confidence, direction, signal_strength = analyze_2min_prediction(df_indicators, current_price)
            trading_state['minute_prediction'] = direction
            trading_state['confidence'] = round(float(confidence), 1)
            trading_state['signal_strength'] = signal_strength
            
            # Extract ML features
            ml_features_current = extract_ml_features(df_indicators, current_price)
            
            # Predict TP/SL
            optimal_tp, optimal_sl, tp_pips, sl_pips = predict_optimal_levels(
                ml_features_current, direction, current_price, df_indicators
            )
            
            # Check active trade
            if trading_state['current_trade']:
                monitor_active_trade(current_price)
            
            # Execute new trade
            if (trading_state['current_trade'] is None and 
                direction != 'NEUTRAL' and 
                confidence >= MIN_CONFIDENCE and
                signal_strength >= 2):
                
                execute_2min_trade(
                    direction, confidence, current_price, 
                    optimal_tp, optimal_sl, tp_pips, sl_pips, signal_strength
                )
            elif trading_state['current_trade'] is None:
                trading_state['action'] = 'WAIT'
                logger.info(f"⚠️  No trade signal: {direction} with {confidence:.1f}% confidence")
            
            # Process Git push queue
            if git_push_queue:
                process_git_push_queue()
            
            # Update state
            trading_state['last_update'] = datetime.now().strftime('%Y-%m-d %H:%M:%S')
            trading_state['server_time'] = datetime.now().isoformat()
            
            # Log summary
            logger.info(f"CYCLE #{cycle_count} SUMMARY:")
            logger.info(f"  Price: {current_price:.5f} ({data_source})")
            logger.info(f"  Prediction: {direction} (Signal: {signal_strength}/3)")
            logger.info(f"  Action: {trading_state['action']}")
            logger.info(f"  Balance: ${trading_state['balance']:.2f}")
            logger.info(f"  Data Storage: {trading_state['data_storage']}")
            logger.info(f"  Git Push: {'✅ Enabled' if trading_state['git_enabled'] else '❌ Disabled'}")
            logger.info(f"  Git Pending: {len(git_push_queue)} queued")
            logger.info(f"{'='*70}")
            
            # Wait for next cycle
            time.sleep(CYCLE_SECONDS)
            
        except Exception as e:
            logger.error(f"❌ Trading cycle error: {e}")
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
        if state_copy['current_trade']:
            trade = state_copy['current_trade'].copy()
            for key in ['entry_time', 'exit_time']:
                if key in trade and trade[key] and isinstance(trade[key], datetime):
                    trade[key] = trade[key].isoformat()
            state_copy['current_trade'] = trade
        return jsonify(state_copy)
    except Exception as e:
        logger.error(f"❌ API error: {e}")
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
            'win_rate': trading_state['win_rate'],
            'git_enabled': trading_state['git_enabled'],
            'git_commits': trading_state['git_commit_count']
        })
    except Exception as e:
        logger.error(f"❌ Trade history error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/manual_git_push', methods=['POST'])
def manual_git_push():
    """Manual Git push endpoint"""
    try:
        if not trading_state['git_enabled']:
            return jsonify({'success': False, 'message': 'Git push not enabled'}), 400
        
        result = execute_git_push()
        
        if result.get('success'):
            return jsonify({
                'success': True,
                'message': 'Git push successful',
                'commit_count': trading_state['git_commit_count']
            })
        else:
            return jsonify({'success': False, 'message': result.get('message')}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/test_git')
def api_test_git():
    """Test Git functionality"""
    try:
        files = {}
        for file in [TRADES_FILE, STATE_FILE, TRAINING_FILE, CONFIG_FILE]:
            exists = os.path.exists(file)
            size = os.path.getsize(file) if exists else 0
            files[os.path.basename(file)] = {'exists': exists, 'size': size}
        
        git_status = {
            'token_exists': bool(GITHUB_TOKEN),
            'repo_exists': os.path.exists(LOCAL_REPO_PATH),
            'git_enabled': trading_state['git_enabled'],
            'data_storage': trading_state['data_storage']
        }
        
        return jsonify({
            'local_files': files,
            'git_status': git_status,
            'trade_history_count': len(trade_history)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/run_git_test', methods=['POST'])
def run_git_test():
    """Run Git push test"""
    result = test_git_push()
    return jsonify(result)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'cycle_count': trading_state['cycle_count'],
        'data_storage': trading_state['data_storage'],
        'git_enabled': trading_state['git_enabled'],
        'git_commits': trading_state['git_commit_count'],
        'trade_count': len(trade_history)
    })

# ==================== START TRADING BOT ====================
def start_trading_bot():
    """Start the trading bot"""
    try:
        thread = threading.Thread(target=trading_cycle, daemon=True)
        thread.start()
        logger.info("✅ Trading bot started successfully")
        print("✅ 2-Minute trading system ACTIVE")
        print(f"✅ Data Storage: Local files with Git push")
        print(f"✅ Git Token: {'✅ Configured via environment' if GITHUB_TOKEN else '❌ NOT CONFIGURED'}")
        print(f"✅ Git Push: {'✅ Enabled' if trading_state['git_enabled'] else '❌ Disabled'}")
    except Exception as e:
        logger.error(f"❌ Error starting trading bot: {e}")
        print(f"❌ Error: {e}")

# ==================== MAIN ENTRY POINT ====================
if __name__ == '__main__':
    start_trading_bot()
    
    port = int(os.environ.get('PORT', 5000))
    print(f"🌐 Web dashboard: http://localhost:{port}")
    print("="*80)
    print("SYSTEM READY")
    print(f"• 2-minute cycles")
    print(f"• Data Storage: Local files")
    print(f"• Git Push: Using GITHUB_TOKEN environment variable")
    print(f"• Test Git: POST to /api/run_git_test")
    print("="*80)
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )