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
# Add this function right after the setup_git_for_push() function (around line 200)

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

# [Add all other trading functions from previous version...]
# get_cached_eurusd_price(), create_price_series(), calculate_advanced_indicators(),
# extract_ml_features(), analyze_2min_prediction(), predict_optimal_levels(),
# predict_with_indicators(), execute_2min_trade(), monitor_active_trade(),
# learn_from_trade(), create_trading_chart()

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
            trading_state['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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