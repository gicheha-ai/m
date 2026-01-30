"""
Simple Git Push Service
Pushes "your service is love" to data/state.json in GitHub repo
"""

import os
import json
import time
import threading
from datetime import datetime
from flask import Flask, jsonify
import subprocess
import shutil
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# GitHub Configuration
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN')
GITHUB_USERNAME = "gicheha-ai"
GITHUB_REPO = "m"
GITHUB_REPO_URL = f"https://github.com/{GITHUB_USERNAME}/{GITHUB_REPO}"
LOCAL_REPO_PATH = "simple_push_repo"
DATA_DIR = os.path.join(LOCAL_REPO_PATH, "data")
STATE_FILE = os.path.join(DATA_DIR, "state.json")

# Status tracking
push_status = {
    'last_push': 'Never',
    'total_pushes': 0,
    'last_error': None,
    'next_push_in': 60,
    'is_running': True,
    'git_enabled': bool(GITHUB_TOKEN)
}

def setup_git_repo():
    """Setup Git repository for pushing"""
    try:
        if not GITHUB_TOKEN:
            logger.warning("⚠️  GITHUB_TOKEN not found in environment variables")
            push_status['git_enabled'] = False
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
            timeout=30
        )
        
        if result.returncode != 0:
            logger.error(f"❌ Git clone failed: {result.stderr[:200]}")
            push_status['git_enabled'] = False
            return False
        
        # Configure git user
        subprocess.run(['git', 'config', 'user.email', 'push-service@gicheha-ai.com'], 
                      cwd=LOCAL_REPO_PATH, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'Push Service'], 
                      cwd=LOCAL_REPO_PATH, capture_output=True)
        
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        logger.info("✅ Git repository setup complete")
        return True
        
    except Exception as e:
        logger.error(f"❌ Git setup error: {e}")
        push_status['git_enabled'] = False
        return False

def execute_git_push():
    """Execute Git push with the message"""
    try:
        if not push_status['git_enabled']:
            logger.warning("⚠️  Git push not enabled")
            return {'success': False, 'message': 'Git push not enabled'}
        
        # Create/update state.json with the message
        message_data = {
            'message': 'your service is love',
            'timestamp': datetime.now().isoformat(),
            'push_number': push_status['total_pushes'] + 1
        }
        
        # Write to file
        with open(STATE_FILE, 'w') as f:
            json.dump(message_data, f, indent=2)
        
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
            
            # 2. Commit changes
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"💾 Committing changes: {timestamp}")
            result = subprocess.run(
                f'git commit -m "Push service update: your service is love - {timestamp}"',
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                # Check if there are actually changes
                result_check = subprocess.run(
                    'git status --porcelain',
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if not result_check.stdout.strip():
                    logger.info("📭 No changes to commit")
                    return {'success': True, 'message': 'No changes to commit'}
                
                logger.error(f"❌ Git commit failed: {result.stderr[:200]}")
                return {'success': False, 'message': 'Git commit failed'}
            
            # 3. Push to GitHub
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
                push_status['last_push'] = datetime.now().strftime('%H:%M:%S')
                push_status['total_pushes'] += 1
                push_status['last_error'] = None
                
                logger.info(f"✅ Git push successful! Total pushes: {push_status['total_pushes']}")
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
                    push_status['last_push'] = datetime.now().strftime('%H:%M:%S')
                    push_status['total_pushes'] += 1
                    push_status['last_error'] = None
                    logger.info(f"✅ Git push successful (alternative method)!")
                    return {'success': True, 'message': 'Git push successful'}
                else:
                    push_status['last_error'] = result.stderr[:200]
                    return {'success': False, 'message': 'Git push failed'}
                    
        finally:
            os.chdir(original_dir)
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Git push timed out")
        push_status['last_error'] = 'Timeout'
        return {'success': False, 'message': 'Git push timed out'}
    except Exception as e:
        logger.error(f"❌ Git push error: {e}")
        push_status['last_error'] = str(e)
        return {'success': False, 'message': str(e)}

def push_cycle():
    """Main push cycle - runs every minute"""
    logger.info("🚀 Starting push service...")
    
    # Setup Git repo
    if not setup_git_repo():
        logger.error("❌ Failed to setup Git repository. Service will not push.")
        push_status['is_running'] = False
        return
    
    push_count = 0
    
    while push_status['is_running']:
        try:
            push_count += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"PUSH CYCLE #{push_count}")
            logger.info(f"{'='*60}")
            
            # Execute push
            result = execute_git_push()
            
            if result.get('success'):
                logger.info(f"✅ Push #{push_count} completed successfully")
                logger.info(f"📊 Total pushes: {push_status['total_pushes']}")
                logger.info(f"🕐 Last push: {push_status['last_push']}")
            else:
                logger.warning(f"⚠️  Push #{push_count} failed: {result.get('message')}")
            
            logger.info(f"{'='*60}")
            logger.info(f"⏰ Next push in 60 seconds...")
            
            # Wait for next cycle
            for i in range(60):
                if not push_status['is_running']:
                    break
                push_status['next_push_in'] = 60 - i
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"❌ Push cycle error: {e}")
            time.sleep(60)

def start_push_service():
    """Start the push service in a background thread"""
    try:
        thread = threading.Thread(target=push_cycle, daemon=True)
        thread.start()
        logger.info("✅ Push service started successfully")
        print("="*60)
        print("SIMPLE GIT PUSH SERVICE")
        print("="*60)
        print(f"Repo: {GITHUB_REPO_URL}")
        print(f"File: data/state.json")
        print(f"Message: 'your service is love'")
        print(f"Interval: Every 60 seconds")
        print(f"Git Token: {'✅ Configured' if push_status['git_enabled'] else '❌ NOT FOUND'}")
        print("="*60)
        print("Service is running...")
        print(f"🌐 Web interface: http://localhost:{os.environ.get('PORT', 5000)}")
        print("="*60)
    except Exception as e:
        logger.error(f"❌ Error starting push service: {e}")
        print(f"❌ Error: {e}")

# ==================== FLASK ROUTES ====================
@app.route('/')
def index():
    """Main page"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Git Push Service</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                min-height: 100vh;
            }
            .container {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }
            h1 {
                text-align: center;
                font-size: 2.5em;
                margin-bottom: 30px;
                text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
            }
            .status-card {
                background: rgba(255, 255, 255, 0.15);
                padding: 20px;
                border-radius: 10px;
                margin: 15px 0;
            }
            .stat {
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            }
            .stat:last-child {
                border-bottom: none;
            }
            .label {
                font-weight: bold;
            }
            .value {
                font-family: monospace;
            }
            .success {
                color: #4ade80;
            }
            .warning {
                color: #fbbf24;
            }
            .error {
                color: #f87171;
            }
            .message-box {
                background: rgba(255, 255, 255, 0.2);
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
                text-align: center;
                font-size: 1.5em;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Git Push Service</h1>
            
            <div class="message-box">
                📝 Pushing: "<span style="color: #ffcc00">your service is love</span>"
            </div>
            
            <div class="status-card">
                <div class="stat">
                    <span class="label">Service Status:</span>
                    <span class="value success">✅ RUNNING</span>
                </div>
                <div class="stat">
                    <span class="label">Git Token:</span>
                    <span class="value """ + ('success">✅ CONFIGURED' if push_status['git_enabled'] else 'warning">⚠️ NOT FOUND') + """</span>
                </div>
                <div class="stat">
                    <span class="label">Last Push:</span>
                    <span class="value">""" + push_status['last_push'] + """</span>
                </div>
                <div class="stat">
                    <span class="label">Total Pushes:</span>
                    <span class="value">""" + str(push_status['total_pushes']) + """</span>
                </div>
                <div class="stat">
                    <span class="label">Next Push In:</span>
                    <span class="value">""" + str(push_status['next_push_in']) + """ seconds</span>
                </div>
            </div>
            
            <div class="status-card">
                <div class="stat">
                    <span class="label">GitHub Repo:</span>
                    <span class="value">""" + GITHUB_REPO_URL + """</span>
                </div>
                <div class="stat">
                    <span class="label">Target File:</span>
                    <span class="value">data/state.json</span>
                </div>
                <div class="stat">
                    <span class="label">Push Interval:</span>
                    <span class="value">Every 60 seconds</span>
                </div>
            </div>
            
            """ + (f'''
            <div class="status-card" style="background: rgba(255, 0, 0, 0.2);">
                <div class="stat">
                    <span class="label">Last Error:</span>
                    <span class="value error">{push_status['last_error'] or "None"}</span>
                </div>
            </div>
            ''' if push_status['last_error'] else '') + """
            
            <div style="text-align: center; margin-top: 30px; font-size: 0.9em; opacity: 0.8;">
                Service automatically pushes "your service is love" to GitHub every minute
            </div>
        </div>
    </body>
    </html>
    """
    return html

@app.route('/api/status')
def get_status():
    """Get service status"""
    return jsonify(push_status)

@app.route('/api/manual_push', methods=['POST'])
def manual_push():
    """Manual push endpoint"""
    result = execute_git_push()
    return jsonify(result)

@app.route('/api/test')
def test_endpoint():
    """Test endpoint"""
    return jsonify({
        'service': 'Git Push Service',
        'message': 'your service is love',
        'status': 'running',
        'timestamp': datetime.now().isoformat(),
        'git_enabled': push_status['git_enabled'],
        'repo': GITHUB_REPO_URL
    })

# ==================== MAIN ENTRY POINT ====================
if __name__ == '__main__':
    start_push_service()
    
    port = int(os.environ.get('PORT', 5000))
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False,
        threaded=True
    )