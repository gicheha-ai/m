// ==================== CONFIGURATION ====================
const REFRESH_INTERVAL = 2000; // 2 seconds
const HISTORY_REFRESH_INTERVAL = 5000; // 5 seconds
const CYCLE_DURATION = 120; // 120 seconds for 2 minutes

// ==================== STATE ====================
let refreshTimer = null;
let historyTimer = null;
let storageTimer = null;
let lastUpdateTime = null;
let isAutoRefreshEnabled = true;

// ==================== INITIALIZATION ====================
document.addEventListener('DOMContentLoaded', function() {
    console.log('EUR/USD 2-Minute Trading System with Git Auto-Sync Initialized');
    
    initializeDashboard();
    setupEventListeners();
    
    fetchTradingState();
    fetchTradeHistory();
    fetchStorageStatus();
    fetchMLStatus();
    
    startAutoRefresh();
});

// ==================== DASHBOARD INITIALIZATION ====================
function initializeDashboard() {
    console.log('Dashboard initialized');
    
    updateSystemStatus('Connecting...', 'warning');
    updateAutoRefreshStatus(true);
    
    updateElementText('remainingTime', `${CYCLE_DURATION}s`);
    updateElementText('cycleDuration', `${CYCLE_DURATION}s`);
    
    updateElementText('dataStorage', `
        <span class="badge bg-info">
            <i class="bi bi-git"></i> Git Auto-Sync
        </span>
    `);
}

// ==================== EVENT LISTENERS ====================
function setupEventListeners() {
    // Sync Now button
    document.getElementById('syncNowBtn').addEventListener('click', function() {
        syncWithGit();
    });
    
    // View Git storage button
    document.getElementById('viewStorageBtn').addEventListener('click', function() {
        window.open('https://github.com/gicheha-ai/m/tree/main/data', '_blank');
        showToast('Opening Git repository...', 'info');
    });
    
    // Refresh button
    document.getElementById('refreshBtn').addEventListener('click', function() {
        console.log('Manual refresh triggered');
        fetchTradingState();
        fetchTradeHistory();
        fetchStorageStatus();
        fetchMLStatus();
        showToast('Data refreshed from Git storage', 'info');
    });
    
    // Reset trading button
    document.getElementById('resetBtn').addEventListener('click', function() {
        if (confirm('Are you sure you want to reset all trading statistics? This will clear trade history and reset balance to $10,000.')) {
            resetTrading();
        }
    });
    
    // Auto-refresh toggle button
    document.getElementById('toggleRefresh').addEventListener('click', function() {
        toggleAutoRefresh();
    });
    
    // Auto-refresh toggle in footer
    document.getElementById('autoRefreshToggle').addEventListener('click', function() {
        toggleAutoRefresh();
    });
}

// ==================== API COMMUNICATION ====================
function fetchTradingState() {
    console.log('Fetching trading state...');
    
    fetch('/api/trading_state')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Trading state received:', data);
            updateDashboard(data);
            updateSystemStatus('Active', 'success');
            lastUpdateTime = new Date();
            
            if (data.git_last_commit || data.git_commit_count) {
                updateGitSyncInfo(data);
            }
        })
        .catch(error => {
            console.error('Error fetching trading state:', error);
            updateSystemStatus('Connection Error', 'danger');
            showToast('Failed to fetch trading data', 'error');
        });
}

function fetchMLStatus() {
    fetch('/api/ml_status')
        .then(response => response.json())
        .then(data => {
            console.log('ML status received:', data);
            updateMLStatusDisplay(data);
        })
        .catch(error => {
            console.error('Error fetching ML status:', error);
        });
}

function fetchTradeHistory() {
    fetch('/api/trade_history')
        .then(response => response.json())
        .then(data => {
            console.log('Trade history received:', data.trades ? data.trades.length : 0, 'trades');
            updateTradeHistory(data);
            
            if (data.ml_samples !== undefined) {
                updateElementText('mlSamplesCount', data.ml_samples);
            }
        })
        .catch(error => {
            console.error('Error fetching trade history:', error);
        });
}

function fetchStorageStatus() {
    fetch('/api/storage_status')
        .then(response => response.json())
        .then(data => {
            console.log('Storage status received:', data);
            updateStorageStatus(data);
        })
        .catch(error => {
            console.error('Error fetching storage status:', error);
        });
}

function syncWithGit() {
    fetch('/api/sync_now', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Synced with Git repository!', 'success');
            fetchTradingState();
            fetchTradeHistory();
            fetchStorageStatus();
            fetchMLStatus();
        } else {
            showToast('Git sync failed', 'error');
        }
    })
    .catch(error => {
        console.error('Error syncing with Git:', error);
        showToast('Error syncing with Git', 'error');
    });
}

function resetTrading() {
    fetch('/api/reset_trading', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showToast('Trading statistics reset successfully!', 'success');
            fetchTradingState();
            fetchTradeHistory();
            fetchStorageStatus();
            fetchMLStatus();
        } else {
            showToast('Failed to reset trading', 'error');
        }
    })
    .catch(error => {
        console.error('Error resetting trading:', error);
        showToast('Error resetting trading', 'error');
    });
}

// ==================== DASHBOARD UPDATES ====================
function updateDashboard(data) {
    if (!data) return;
    
    console.log('Updating dashboard with live data');
    
    // Update cycle information
    updateElementText('cycleCount', data.cycle_count || 0);
    updateElementText('nextCycle', `${data.next_cycle_in || CYCLE_DURATION}s`);
    updateElementText('remainingTime', `${data.remaining_time || CYCLE_DURATION}s`);
    
    // Update progress bars
    const cycleProgress = data.cycle_progress || 0;
    updateProgressBar('cycleProgress', cycleProgress, 'bg-info');
    
    const tradeProgress = data.trade_progress || 0;
    updateElementText('tradeProgress', `${Math.round(tradeProgress)}%`);
    
    // Update time
    updateElementText('lastUpdate', data.last_update || '-');
    updateElementText('serverTime', formatTime(data.server_time));
    
    // Update market data
    updateElementText('currentPrice', (data.current_price || 1.08500).toFixed(5));
    updateElementText('dataSource', data.data_source || '-');
    
    // Update API status
    const apiStatus = document.getElementById('apiStatus');
    apiStatus.textContent = data.api_status || 'UNKNOWN';
    apiStatus.className = `badge ${
        data.api_status === 'CONNECTED' ? 'bg-success' :
        data.api_status === 'CACHED' ? 'bg-warning' :
        data.api_status === 'STALE_CACHE' ? 'bg-warning' :
        data.api_status === 'SIMULATION' ? 'bg-secondary' :
        'bg-danger'
    }`;
    
    // Update data storage status
    const dataStorage = document.getElementById('dataStorage');
    const storageType = data.data_storage || 'GIT_REPO_SYNCING';
    let badgeClass = 'bg-info';
    let icon = 'bi-git';
    let text = 'Git Auto-Sync';
    
    if (storageType.includes('READY')) {
        badgeClass = 'bg-success';
        text = 'Git Auto-Sync Ready';
    } else if (storageType.includes('ERROR')) {
        badgeClass = 'bg-danger';
        text = 'Git Sync Error';
    }
    
    dataStorage.innerHTML = `
        <span class="badge ${badgeClass}">
            <i class="bi ${icon}"></i>
            ${text}
        </span>
    `;
    
    // Update signal strength
    const signalStrength = document.getElementById('signalStrength');
    const strength = data.signal_strength || 0;
    signalStrength.textContent = `${strength}/3`;
    signalStrength.className = `badge ${
        strength === 3 ? 'bg-success' :
        strength === 2 ? 'bg-warning' :
        'bg-secondary'
    }`;
    
    // Update prediction
    const prediction = document.getElementById('prediction');
    const pred = data.minute_prediction || 'ANALYZING';
    prediction.textContent = pred;
    prediction.className = `display-6 fw-bold ${
        pred === 'BULLISH' ? 'buy-signal' :
        pred === 'BEARISH' ? 'sell-signal' :
        'wait-signal'
    }`;
    
    // Update confidence
    const confidence = data.confidence || 0;
    updateProgressBar('confidenceBar', confidence, getConfidenceColor(confidence));
    updateElementText('confidenceText', `${confidence.toFixed(1)}%`);
    
    // Update action
    const action = document.getElementById('action');
    const act = data.action || 'WAIT';
    action.textContent = act;
    action.className = `badge ${
        act === 'BUY' ? 'bg-success fs-5 p-2' :
        act === 'SELL' ? 'bg-danger fs-5 p-2' :
        'bg-secondary fs-5 p-2'
    }`;
    
    // Update statistics
    updateElementText('balance', `$${(data.balance || 10000).toFixed(2)}`);
    updateElementText('totalTrades', data.total_trades || 0);
    updateElementText('winRate', `${(data.win_rate || 0).toFixed(1)}%`);
    updateElementText('tradeStatus', data.trade_status || 'NO_TRADE');
    
    // Update ML status
    const mlReady = document.getElementById('mlReady');
    mlReady.textContent = data.ml_model_ready ? 'Yes' : 'No';
    mlReady.className = `badge ${data.ml_model_ready ? 'bg-success' : 'bg-warning'}`;
    
    // Update TP/SL levels
    updateElementText('optimalTp', data.optimal_tp ? data.optimal_tp.toFixed(5) : '-');
    updateElementText('optimalSl', data.optimal_sl ? data.optimal_sl.toFixed(5) : '-');
    
    const tpPips = data.tp_distance_pips || 0;
    const slPips = data.sl_distance_pips || 0;
    updateElementText('tpPips', `<span class="badge bg-success">${tpPips} pips</span>`);
    updateElementText('slPips', `<span class="badge bg-danger">${slPips} pips</span>`);
    
    // Update risk/reward ratio
    if (tpPips > 0 && slPips > 0) {
        const riskReward = (tpPips / slPips).toFixed(2);
        updateElementText('riskReward', `1:${riskReward}`);
    }
    
    // Update cache efficiency
    if (data.cache_efficiency) {
        updateElementText('cacheEfficiency', data.cache_efficiency);
        updateElementText('cacheInfo', data.cache_efficiency);
    }
    
    // Update volatility
    if (data.volatility) {
        updateElementText('volatility', data.volatility);
    }
    
    // Update active trade
    updateActiveTrade(data.current_trade);
    
    // Update Git repo link
    if (data.git_repo_url) {
        updateElementText('gitRepoLink', `
            <a href="${data.git_repo_url}" target="_blank" class="text-decoration-none text-light">
                <i class="bi bi-github"></i> View on GitHub
            </a>
        `);
    }
    
    // Update chart if data available
    if (data.chart_data) {
        updateChart(data.chart_data);
    }
    
    // Update footer status
    updateFooterStatus();
}

function updateGitSyncInfo(data) {
    const gitSyncInfo = document.getElementById('gitSyncInfo');
    if (gitSyncInfo && (data.git_last_commit || data.git_commit_count)) {
        const lastCommit = data.git_last_commit || 'Never';
        const commitCount = data.git_commit_count || 0;
        
        gitSyncInfo.innerHTML = `
            <div class="small text-success">
                <i class="bi bi-git"></i> Last Commit: ${lastCommit}
                <br>
                <i class="bi bi-hash"></i> Total Commits: ${commitCount}
            </div>
        `;
    }
}

function updateMLStatusDisplay(mlData) {
    const mlStatusContainer = document.getElementById('mlStatusContainer');
    if (mlStatusContainer) {
        const isTrained = mlData.ml_model_ready || false;
        const samples = mlData.training_samples || 0;
        
        mlStatusContainer.innerHTML = `
            <span class="badge ${isTrained ? 'bg-success' : 'bg-warning'}">
                <i class="bi bi-cpu"></i> ${isTrained ? 'ML Trained' : 'ML Learning'}
            </span>
        `;
    }
    
    updateElementText('mlSamplesCount', mlData.training_samples || 0);
    
    const mlFooterStatus = document.getElementById('mlFooterStatus');
    const mlStatusBadge = document.getElementById('mlStatusBadge');
    if (mlFooterStatus && mlStatusBadge) {
        const isTrained = mlData.ml_model_ready || false;
        mlFooterStatus.textContent = isTrained ? 'ML Active' : 'ML Learning';
        mlFooterStatus.className = `badge ${isTrained ? 'bg-success' : 'bg-warning'}`;
        mlStatusBadge.textContent = isTrained ? 'Trained' : 'Learning';
        mlStatusBadge.className = `badge ${isTrained ? 'bg-success' : 'bg-warning'}`;
    }
}

function updateActiveTrade(trade) {
    const activeTradeDiv = document.getElementById('activeTrade');
    
    if (!trade) {
        activeTradeDiv.innerHTML = `
            <div class="text-center text-muted py-4">
                <i class="bi bi-hourglass-split display-4 d-block mb-3"></i>
                <span class="fs-5">No active trade</span>
                <p class="small mt-2">Waiting for next trading signal...</p>
            </div>
        `;
        return;
    }
    
    const profitPips = trade.profit_pips || 0;
    const duration = trade.duration_seconds || 0;
    const timeRemaining = Math.max(0, CYCLE_DURATION - duration);
    const profitPercent = trade.profit_amount ? ((trade.profit_amount / trade.trade_size) * 100).toFixed(3) : '0.000';
    const mlUsed = trade.ml_used || false;
    
    const progressPercent = Math.min(100, (duration / CYCLE_DURATION) * 100);
    
    activeTradeDiv.innerHTML = `
        <div class="text-center">
            <h5 class="${trade.action === 'BUY' ? 'text-success' : 'text-danger'}">
                <i class="bi ${trade.action === 'BUY' ? 'bi-arrow-up-circle' : 'bi-arrow-down-circle'}"></i>
                ${trade.action} #${trade.id}
            </h5>
            
            ${mlUsed ? `
            <div class="small text-info mb-2">
                <i class="bi bi-cpu"></i> ML-Optimized Trade
            </div>
            ` : ''}
            
            <div class="row text-start mt-3">
                <div class="col-6"><small>Entry Price:</small></div>
                <div class="col-6 text-end"><strong>${trade.entry_price.toFixed(5)}</strong></div>
                
                <div class="col-6"><small>Current P/L:</small></div>
                <div class="col-6 text-end ${profitPips >= 0 ? 'text-success' : 'text-danger'}">
                    <strong>${profitPips.toFixed(1)} pips</strong>
                </div>
                
                <div class="col-6"><small>P/L %:</small></div>
                <div class="col-6 text-end ${profitPips >= 0 ? 'text-success' : 'text-danger'}">
                    <strong>${profitPercent}%</strong>
                </div>
                
                <div class="col-6"><small>Duration:</small></div>
                <div class="col-6 text-end"><strong>${Math.round(duration)}s</strong></div>
                
                <div class="col-6"><small>Time Left:</small></div>
                <div class="col-6 text-end"><strong>${timeRemaining}s</strong></div>
                
                <div class="col-6"><small>Status:</small></div>
                <div class="col-6 text-end">
                    <span class="badge ${trade.status === 'OPEN' ? 'bg-warning' : 'bg-secondary'}">
                        ${trade.status || 'OPEN'}
                    </span>
                </div>
            </div>
            
            <div class="mt-4">
                <div class="progress" style="height: 12px;">
                    <div class="progress-bar progress-bar-striped progress-bar-animated ${
                        profitPips >= 0 ? 'bg-success' : 'bg-danger'
                    }" style="width: ${progressPercent}%"></div>
                </div>
                <small class="text-muted d-block mt-1">Trade progress: ${Math.round(progressPercent)}% (${CYCLE_DURATION}s total)</small>
            </div>
            
            <div class="mt-3">
                <small class="text-muted">
                    <i class="bi bi-info-circle"></i>
                    Target: TP at ${trade.optimal_tp.toFixed(5)} (${trade.tp_distance_pips} pips) | 
                    Max Loss: SL at ${trade.optimal_sl.toFixed(5)} (${trade.sl_distance_pips} pips)
                </small>
            </div>
            
            ${trade.data_stored_in ? `
            <div class="small text-info mt-2">
                <i class="bi bi-git"></i> Auto-saved to Git
            </div>
            ` : ''}
        </div>
    `;
}

function updateTradeHistory(data) {
    const tbody = document.getElementById('tradeHistory');
    
    if (!data || !data.trades || data.trades.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="11" class="text-center py-4">
                    <i class="bi bi-inbox display-6 d-block text-muted mb-2"></i>
                    <span class="text-muted">No trades yet</span>
                    ${data && data.data_source === 'Git Repository (Auto-Sync)' ? 
                        '<div class="small text-info mt-2"><i class="bi bi-git"></i> All trades auto-saved to Git repository</div>' : 
                        ''}
                </td>
            </tr>
        `;
        
        updateElementText('totalHistoryTrades', '0');
        updateElementText('totalWins', '0');
        return;
    }
    
    const totalTrades = data.total || data.trades.length;
    const profitableTrades = data.profitable || data.trades.filter(t => t.result === 'SUCCESS').length;
    
    updateElementText('totalHistoryTrades', totalTrades);
    updateElementText('totalWins', profitableTrades);
    
    let html = '';
    data.trades.slice().reverse().forEach(trade => {
        const entryTime = trade.entry_time ? new Date(trade.entry_time) : new Date();
        const exitTime = trade.exit_time ? new Date(trade.exit_time) : null;
        const duration = exitTime ? Math.round((exitTime - entryTime) / 1000) : 0;
        const profitPips = trade.profit_pips || 0;
        const confidence = trade.confidence || 0;
        const mlUsed = trade.ml_used || false;
        
        const timeStr = entryTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        let resultClass = 'bg-secondary';
        let resultText = trade.result || 'PENDING';
        
        if (trade.result === 'SUCCESS') {
            resultClass = 'bg-success';
        } else if (trade.result === 'FAILED') {
            resultClass = 'bg-danger';
        } else if (trade.result === 'PARTIAL_SUCCESS') {
            resultClass = 'bg-info';
        } else if (trade.result === 'PARTIAL_FAIL') {
            resultClass = 'bg-warning';
        } else if (trade.result === 'BREAKEVEN') {
            resultClass = 'bg-secondary';
        }
        
        html += `
            <tr>
                <td>
                    <strong>#${trade.id}</strong>
                    ${mlUsed ? '<br><small class="text-info"><i class="bi bi-cpu"></i> ML</small>' : ''}
                </td>
                <td>${timeStr}</td>
                <td>
                    <span class="badge ${trade.action === 'BUY' ? 'bg-success' : 'bg-danger'}">
                        ${trade.action}
                    </span>
                </td>
                <td>${trade.entry_price.toFixed(5)}</td>
                <td>
                    <small class="d-block">TP: ${trade.optimal_tp ? trade.optimal_tp.toFixed(5) : '-'}</small>
                    <small>SL: ${trade.optimal_sl ? trade.optimal_sl.toFixed(5) : '-'}</small>
                </td>
                <td>${trade.exit_price ? trade.exit_price.toFixed(5) : '-'}</td>
                <td class="${profitPips >= 0 ? 'text-success' : 'text-danger'} fw-bold">
                    ${profitPips.toFixed(1)}
                </td>
                <td>${duration}s</td>
                <td>
                    <span class="badge ${resultClass}">
                        ${resultText}
                    </span>
                </td>
                <td>
                    <div class="progress" style="height: 6px; width: 60px;">
                        <div class="progress-bar ${getConfidenceColor(confidence)}" 
                             style="width: ${confidence}%" title="${confidence}% confidence">
                        </div>
                    </div>
                    <small>${confidence.toFixed(0)}%</small>
                </td>
                <td>
                    ${trade.signal_strength ? 
                        `<span class="badge ${trade.signal_strength === 3 ? 'bg-success' : trade.signal_strength === 2 ? 'bg-warning' : 'bg-secondary'}">
                            ${trade.signal_strength}/3
                        </span>` : 
                        '-'}
                </td>
            </tr>
        `;
    });
    
    tbody.innerHTML = html;
}

function updateStorageStatus(data) {
    if (!data) return;
    
    const storageInfo = document.getElementById('storageInfo');
    if (storageInfo) {
        let filesHtml = '';
        if (data.files) {
            for (const [fileName, fileInfo] of Object.entries(data.files)) {
                if (fileInfo.exists) {
                    filesHtml += `
                        <div class="small ${fileInfo.size_bytes > 0 ? 'text-success' : 'text-muted'}">
                            <i class="bi ${fileInfo.size_bytes > 0 ? 'bi-file-earmark-check' : 'bi-file-earmark'}"></i> 
                            ${fileName}: ${fileInfo.size_human}
                        </div>
                    `;
                }
            }
        }
        
        const mlTrained = data.ml_trained || false;
        const trainingSamples = data.training_samples || 0;
        
        storageInfo.innerHTML = `
            <div class="text-center">
                <i class="bi bi-git display-6 ${data.git_commits > 0 ? 'text-success' : 'text-info'} mb-2"></i>
                <h6 class="mb-2">Git Auto-Sync Storage</h6>
                <div class="small ${data.git_commits > 0 ? 'text-success' : 'text-muted'} mb-2">
                    <i class="bi bi-hash"></i> Git Commits: ${data.git_commits || 0}
                </div>
                <div class="small ${data.last_commit && data.last_commit !== 'Never' ? 'text-success' : 'text-muted'} mb-2">
                    <i class="bi bi-clock"></i> Last Commit: ${data.last_commit || 'Never'}
                </div>
                <div class="small ${data.trade_count > 0 ? 'text-success' : 'text-muted'} mb-2">
                    <i class="bi bi-database"></i> Trade Count: ${data.trade_count || 0}
                </div>
                <div class="small ${mlTrained ? 'text-success' : trainingSamples > 0 ? 'text-warning' : 'text-muted'} mb-2">
                    <i class="bi bi-cpu"></i> ML Samples: ${trainingSamples} ${mlTrained ? '(Trained)' : '(Learning)'}
                </div>
                ${filesHtml ? `
                <div class="mt-2 pt-2 border-top border-secondary">
                    <div class="small mb-1">Files:</div>
                    ${filesHtml}
                </div>
                ` : ''}
            </div>
        `;
    }
}

// ==================== AUTO-REFRESH SYSTEM ====================
function startAutoRefresh() {
    if (refreshTimer) {
        clearInterval(refreshTimer);
    }
    
    refreshTimer = setInterval(() => {
        if (isAutoRefreshEnabled) {
            fetchTradingState();
        }
    }, REFRESH_INTERVAL);
    
    if (historyTimer) {
        clearInterval(historyTimer);
    }
    
    historyTimer = setInterval(() => {
        if (isAutoRefreshEnabled) {
            fetchTradeHistory();
            fetchStorageStatus();
            fetchMLStatus();
        }
    }, HISTORY_REFRESH_INTERVAL);
    
    if (storageTimer) {
        clearInterval(storageTimer);
    }
    
    storageTimer = setInterval(() => {
        if (isAutoRefreshEnabled) {
            fetchStorageStatus();
        }
    }, 10000);
    
    updateAutoRefreshStatus(true);
    console.log('Auto-refresh started with Git sync monitoring');
}

function stopAutoRefresh() {
    if (refreshTimer) {
        clearInterval(refreshTimer);
        refreshTimer = null;
    }
    
    if (historyTimer) {
        clearInterval(historyTimer);
        historyTimer = null;
    }
    
    if (storageTimer) {
        clearInterval(storageTimer);
        storageTimer = null;
    }
    
    updateAutoRefreshStatus(false);
    console.log('Auto-refresh stopped');
}

function toggleAutoRefresh() {
    isAutoRefreshEnabled = !isAutoRefreshEnabled;
    
    if (isAutoRefreshEnabled) {
        startAutoRefresh();
        showToast('Auto-refresh enabled', 'info');
    } else {
        stopAutoRefresh();
        showToast('Auto-refresh disabled', 'warning');
    }
    
    const toggleBtn = document.getElementById('toggleRefresh');
    if (toggleBtn) {
        const statusSpan = toggleBtn.querySelector('span');
        if (statusSpan) {
            statusSpan.textContent = isAutoRefreshEnabled ? 'Enabled' : 'Disabled';
            statusSpan.className = `badge ${isAutoRefreshEnabled ? 'bg-success' : 'bg-secondary'}`;
        }
    }
}

// ==================== HELPER FUNCTIONS ====================
function updateElementText(elementId, text) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = text;
    }
}

function updateProgressBar(elementId, value, colorClass = 'bg-primary') {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.width = `${Math.min(100, value)}%`;
        element.textContent = `${Math.round(value)}%`;
        element.className = `progress-bar ${colorClass}`;
        if (value > 0 && value < 100) {
            element.classList.add('progress-bar-striped', 'progress-bar-animated');
        }
    }
}

function getConfidenceColor(confidence) {
    if (confidence >= 80) return 'bg-success';
    if (confidence >= 60) return 'bg-info';
    if (confidence >= 40) return 'bg-warning';
    return 'bg-danger';
}

function formatTime(dateString) {
    if (!dateString) return '-';
    try {
        const date = new Date(dateString);
        return date.toLocaleTimeString([], { 
            hour: '2-digit', 
            minute: '2-digit',
            second: '2-digit'
        });
    } catch (e) {
        return dateString;
    }
}

function updateSystemStatus(status, type = 'info') {
    const elements = [
        document.getElementById('systemStatus'),
        document.getElementById('systemStatusFooter')
    ];
    
    elements.forEach(element => {
        if (element) {
            element.textContent = status;
            element.className = `badge bg-${type}`;
        }
    });
}

function updateAutoRefreshStatus(enabled) {
    const elements = [
        document.getElementById('autoRefreshStatus'),
        document.getElementById('autoRefreshToggle')
    ];
    
    elements.forEach(element => {
        if (element) {
            element.textContent = enabled ? 'Enabled' : 'Disabled';
            element.className = `badge ${enabled ? 'bg-success' : 'bg-secondary'}`;
        }
    });
}

function updateFooterStatus() {
    const now = new Date();
    document.getElementById('lastUpdate').textContent = now.toLocaleTimeString();
}

function showToast(message, type = 'info') {
    let toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toastContainer';
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header bg-${type} text-white">
                <strong class="me-auto">
                    <i class="bi ${
                        type === 'success' ? 'bi-check-circle' :
                        type === 'error' ? 'bi-exclamation-circle' :
                        type === 'warning' ? 'bi-exclamation-triangle' :
                        'bi-info-circle'
                    }"></i>
                    Trading System
                </strong>
                <small>Just now</small>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body bg-dark text-white">
                ${message}
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    const toastEl = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastEl, {
        delay: 3000
    });
    toast.show();
    
    toastEl.addEventListener('hidden.bs.toast', function () {
        toastEl.remove();
    });
}

// ==================== CLEANUP ====================
window.addEventListener('beforeunload', function() {
    stopAutoRefresh();
});

// Export functions for debugging
window.fetchTradingState = fetchTradingState;
window.fetchTradeHistory = fetchTradeHistory;
window.fetchStorageStatus = fetchStorageStatus;
window.fetchMLStatus = fetchMLStatus;
window.syncWithGit = syncWithGit;
window.resetTrading = resetTrading;
window.startAutoRefresh = startAutoRefresh;
window.stopAutoRefresh = stopAutoRefresh;
window.toggleAutoRefresh = toggleAutoRefresh;
window.showToast = showToast;

console.log('Trading System loaded successfully');