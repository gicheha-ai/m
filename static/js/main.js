/**
 * EUR/USD Trading System - Main JavaScript Module
 * Fixed for 2-Minute Cycles with Git Repository Storage
 */

// Configuration
const API_BASE_URL = '';
const REFRESH_INTERVAL = 2000; // 2 seconds
const HISTORY_REFRESH_INTERVAL = 5000; // 5 seconds
const CYCLE_DURATION = 120; // ⭐ FIXED: 120 seconds for 2 minutes

// State
let refreshTimer = null;
let historyTimer = null;
let lastUpdateTime = null;
let isAutoRefreshEnabled = true;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('EUR/USD 2-Minute Trading System Initialized (Git Storage)');
    
    // Initialize systems
    initializeDashboard();
    
    // Load initial data
    fetchTradingState();
    fetchTradeHistory();
    fetchStorageStatus(); // ⭐ NEW: Check Git storage status
    
    // Set up event listeners
    setupEventListeners();
    
    // Start auto-refresh
    startAutoRefresh();
    
    // Initialize chart system if available
    if (typeof initializeChartSystem === 'function') {
        initializeChartSystem();
    }
});

// Initialize dashboard elements
function initializeDashboard() {
    console.log('Dashboard initialized (2-Minute Cycles with Git Storage)');
    
    // Set initial values
    updateSystemStatus('Initializing...', 'warning');
    updateAutoRefreshStatus(true);
    
    // Set cycle duration displays
    updateElementText('remainingTime', `${CYCLE_DURATION}s`);
    updateElementText('cycleDuration', `${CYCLE_DURATION}s`);
    
    // Update data storage info
    updateElementText('dataStorage', `
        <span class="badge bg-info">
            <i class="bi bi-git"></i> Git Repository
        </span>
    `);
}

// Set up all event listeners
function setupEventListeners() {
    // Refresh button
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            fetchTradingState();
            fetchTradeHistory();
            fetchStorageStatus(); // ⭐ NEW: Refresh storage status
            showToast('Data refreshed manually', 'info');
        });
    }
    
    // Reset trading button
    const resetBtn = document.getElementById('resetBtn');
    if (resetBtn) {
        resetBtn.addEventListener('click', function() {
            if (confirm('Are you sure you want to reset all trading statistics? This will clear trade history and reset balance to $10,000.')) {
                resetTrading();
            }
        });
    }
    
    // Toggle auto-refresh
    const autoRefreshToggle = document.getElementById('autoRefreshToggle');
    if (autoRefreshToggle) {
        autoRefreshToggle.addEventListener('click', function() {
            toggleAutoRefresh();
        });
    }
    
    // Toggle chart fullscreen
    const toggleChartBtn = document.getElementById('toggleChart');
    if (toggleChartBtn) {
        toggleChartBtn.addEventListener('click', function() {
            // Chart fullscreen handled by charts.js
        });
    }
    
    // ⭐ NEW: View Git storage button
    const viewStorageBtn = document.getElementById('viewStorageBtn');
    if (viewStorageBtn) {
        viewStorageBtn.addEventListener('click', function() {
            window.open('https://github.com/gicheha-ai/m/tree/main/data', '_blank');
            showToast('Opening Git repository...', 'info');
        });
    }
}

// Fetch current trading state from API
function fetchTradingState() {
    fetch('/api/trading_state')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            updateDashboard(data);
            updateSystemStatus('Active', 'success');
            lastUpdateTime = new Date();
        })
        .catch(error => {
            console.error('Error fetching trading state:', error);
            updateSystemStatus('Connection Error', 'danger');
            showToast('Failed to fetch trading data', 'error');
        });
}

// Fetch trade history from API
function fetchTradeHistory() {
    fetch('/api/trade_history')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            updateTradeHistory(data);
        })
        .catch(error => {
            console.error('Error fetching trade history:', error);
        });
}

// ⭐ NEW: Fetch storage status from API
function fetchStorageStatus() {
    fetch('/api/storage_status')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            updateStorageStatus(data);
        })
        .catch(error => {
            console.error('Error fetching storage status:', error);
        });
}

// Update dashboard with new data
function updateDashboard(data) {
    if (!data) return;
    
    // Update cycle information
    updateElementText('cycleCount', data.cycle_count || 0);
    updateElementText('nextCycle', `${data.next_cycle_in || CYCLE_DURATION}s`);
    
    // Update progress bars (scaled to 120 seconds)
    const cycleProgress = data.cycle_progress || 0;
    updateProgressBar('cycleProgress', cycleProgress, 'bg-info');
    
    const tradeProgress = data.trade_progress || 0;
    updateElementText('tradeProgress', `${Math.round(tradeProgress)}%`);
    
    // Update time
    updateElementText('lastUpdate', data.last_update || '-');
    updateElementText('serverTime', formatTime(data.server_time));
    updateElementText('remainingTime', `${data.remaining_time || CYCLE_DURATION}s`);
    
    // Update market data
    updateElementText('currentPrice', (data.current_price || 1.08500).toFixed(5));
    updateElementText('dataSource', data.data_source || '-');
    
    // Update API status
    const apiStatus = document.getElementById('apiStatus');
    if (apiStatus) {
        apiStatus.textContent = data.api_status || 'UNKNOWN';
        apiStatus.className = `badge ${
            data.api_status === 'CONNECTED' ? 'bg-success' :
            data.api_status === 'DEMO' ? 'bg-warning' :
            'bg-danger'
        }`;
    }
    
    // ⭐ NEW: Update data storage status
    const dataStorage = document.getElementById('dataStorage');
    if (dataStorage) {
        const storageType = data.data_storage || 'GIT_REPO';
        dataStorage.innerHTML = `
            <span class="badge ${storageType.includes('READY') ? 'bg-success' : 'bg-warning'}">
                <i class="bi ${storageType.includes('GIT') ? 'bi-git' : 'bi-hdd'}"></i>
                ${storageType.includes('GIT') ? 'Git Repository' : storageType}
            </span>
        `;
    }
    
    // Update signal strength
    const signalStrength = document.getElementById('signalStrength');
    if (signalStrength) {
        const strength = data.signal_strength || 0;
        signalStrength.textContent = `${strength}/3`;
        signalStrength.className = `badge ${
            strength === 3 ? 'bg-success' :
            strength === 2 ? 'bg-warning' :
            'bg-secondary'
        }`;
    }
    
    // Update prediction
    const prediction = document.getElementById('prediction');
    if (prediction) {
        prediction.textContent = data.minute_prediction || 'ANALYZING';
        prediction.className = `display-6 fw-bold ${
            data.minute_prediction === 'BULLISH' ? 'buy-signal' :
            data.minute_prediction === 'BEARISH' ? 'sell-signal' :
            'wait-signal'
        }`;
    }
    
    // Update confidence
    const confidence = data.confidence || 0;
    updateProgressBar('confidenceBar', confidence, getConfidenceColor(confidence));
    updateElementText('confidenceText', `${confidence.toFixed(1)}%`);
    
    // Update action
    const action = document.getElementById('action');
    if (action) {
        action.textContent = data.action || 'WAIT';
        action.className = `badge ${
            data.action === 'BUY' ? 'bg-success fs-5 p-2' :
            data.action === 'SELL' ? 'bg-danger fs-5 p-2' :
            'bg-secondary fs-5 p-2'
        }`;
    }
    
    // Update statistics
    updateElementText('balance', `$${(data.balance || 10000).toFixed(2)}`);
    updateElementText('totalTrades', data.total_trades || 0);
    updateElementText('winRate', `${(data.win_rate || 0).toFixed(1)}%`);
    
    // Update ML status
    const mlReady = document.getElementById('mlReady');
    if (mlReady) {
        mlReady.textContent = data.ml_model_ready ? 'Yes' : 'No';
        mlReady.className = `badge ${data.ml_model_ready ? 'bg-success' : 'bg-warning'}`;
    }
    
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
    
    // Update active trade
    updateActiveTrade(data.current_trade);
    
    // Update chart if available
    if (typeof updateChartFromState === 'function' && data.chart_data) {
        updateChartFromState(data);
    }
    
    // Update price history if available
    if (data.price_history && Array.isArray(data.price_history)) {
        updatePriceHistory(data.price_history);
    }
    
    // Update trade status
    updateElementText('tradeStatus', data.trade_status || 'NO_TRADE');
    
    // ⭐ NEW: Update Git repo info
    if (data.git_repo_url) {
        updateElementText('gitRepoLink', `
            <a href="${data.git_repo_url}" target="_blank" class="text-decoration-none">
                <span class="badge bg-dark">
                    <i class="bi bi-github"></i> View on GitHub
                </span>
            </a>
        `);
    }
}

// ⭐ NEW: Update storage status display
function updateStorageStatus(data) {
    if (!data) return;
    
    const storageInfo = document.getElementById('storageInfo');
    if (storageInfo) {
        let filesHtml = '';
        if (data.files) {
            for (const [fileName, fileInfo] of Object.entries(data.files)) {
                if (fileInfo.exists) {
                    filesHtml += `
                        <div class="small text-muted">
                            <i class="bi bi-file-earmark-text"></i> ${fileName}: ${fileInfo.size_human}
                        </div>
                    `;
                }
            }
        }
        
        storageInfo.innerHTML = `
            <div class="card bg-dark border-secondary">
                <div class="card-header py-2 bg-transparent border-secondary">
                    <h6 class="mb-0">
                        <i class="bi bi-database"></i> Git Storage Status
                    </h6>
                </div>
                <div class="card-body py-2">
                    <div class="row">
                        <div class="col-6">
                            <div class="small">Storage Type:</div>
                            <div class="fw-bold">${data.data_storage || 'Git Repository'}</div>
                        </div>
                        <div class="col-6">
                            <div class="small">Trade Count:</div>
                            <div class="fw-bold">${data.trade_count || 0}</div>
                        </div>
                    </div>
                    <div class="row mt-2">
                        <div class="col-6">
                            <div class="small">Training Samples:</div>
                            <div class="fw-bold">${data.training_samples || 0}</div>
                        </div>
                        <div class="col-6">
                            <div class="small">Data Directory:</div>
                            <div class="fw-bold">${data.data_directory || 'data/'}</div>
                        </div>
                    </div>
                    ${filesHtml ? `
                    <div class="mt-2 pt-2 border-top border-secondary">
                        <div class="small mb-1">Files:</div>
                        ${filesHtml}
                    </div>
                    ` : ''}
                </div>
            </div>
        `;
    }
}

// Update active trade display (adjusted for 120 seconds)
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
    
    // Calculate progress percentage (based on 120 seconds)
    const progressPercent = Math.min(100, (duration / CYCLE_DURATION) * 100);
    
    // ⭐ NEW: Show Git storage info in active trade
    const gitStorageInfo = trade.data_stored_in ? `
        <div class="small text-info mt-2">
            <i class="bi bi-git"></i> Stored in: ${trade.data_stored_in}
        </div>
    ` : '';
    
    activeTradeDiv.innerHTML = `
        <div class="text-center">
            <h5 class="${trade.action === 'BUY' ? 'text-success' : 'text-danger'}">
                <i class="bi ${trade.action === 'BUY' ? 'bi-arrow-up-circle' : 'bi-arrow-down-circle'}"></i>
                ${trade.action} #${trade.id}
            </h5>
            
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
            
            ${gitStorageInfo}
        </div>
    `;
}

// Update trade history table with Git storage info
function updateTradeHistory(data) {
    const tbody = document.getElementById('tradeHistory');
    
    if (!data || !data.trades || data.trades.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="11" class="text-center py-4">
                    <i class="bi bi-inbox display-6 d-block text-muted mb-2"></i>
                    <span class="text-muted">No trades yet</span>
                    ${data && data.data_source === 'Git Repository' ? 
                        '<div class="small text-info mt-2"><i class="bi bi-git"></i> All trades stored in Git repository</div>' : 
                        ''}
                </td>
            </tr>
        `;
        
        updateElementText('totalHistoryTrades', '0');
        updateElementText('totalWins', '0');
        return;
    }
    
    // Update summary counts
    const totalTrades = data.total || data.trades.length;
    const profitableTrades = data.profitable || data.trades.filter(t => t.result === 'SUCCESS').length;
    
    updateElementText('totalHistoryTrades', totalTrades);
    updateElementText('totalWins', profitableTrades);
    
    // Update Git storage info
    const gitRepoInfo = document.getElementById('gitRepoInfo');
    if (gitRepoInfo && data.git_repo) {
        gitRepoInfo.innerHTML = `
            <div class="small text-info">
                <i class="bi bi-git"></i> 
                <a href="${data.git_repo}" target="_blank" class="text-decoration-none text-info">
                    All trades stored in Git repository
                </a>
                <span class="text-muted"> | File: ${data.storage_file || 'data/trades.json'}</span>
            </div>
        `;
    }
    
    // Build table rows
    let html = '';
    data.trades.slice().reverse().forEach(trade => {
        const entryTime = trade.entry_time ? new Date(trade.entry_time) : new Date();
        const exitTime = trade.exit_time ? new Date(trade.exit_time) : null;
        const duration = exitTime ? Math.round((exitTime - entryTime) / 1000) : 0;
        const profitPips = trade.profit_pips || 0;
        const confidence = trade.confidence || 0;
        
        // Format time
        const timeStr = entryTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        // Determine result badge class
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
        
        // ⭐ NEW: Git icon for stored trades
        const gitIcon = trade.data_stored_in ? 
            '<i class="bi bi-check-circle text-success" title="Stored in Git"></i>' : 
            '';
        
        html += `
            <tr>
                <td><strong>#${trade.id}</strong> ${gitIcon}</td>
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

// Reset trading statistics (now includes Git storage)
function resetTrading() {
    fetch('/api/reset_trading', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.success) {
            showToast('Trading statistics reset successfully!', 'success');
            fetchTradingState();
            fetchTradeHistory();
            fetchStorageStatus(); // ⭐ NEW: Refresh storage status
        } else {
            showToast('Failed to reset trading', 'error');
        }
    })
    .catch(error => {
        console.error('Error resetting trading:', error);
        showToast('Error resetting trading', 'error');
    });
}

// Start auto-refresh
function startAutoRefresh() {
    if (refreshTimer) {
        clearInterval(refreshTimer);
    }
    
    refreshTimer = setInterval(() => {
        if (isAutoRefreshEnabled) {
            fetchTradingState();
        }
    }, REFRESH_INTERVAL);
    
    // Separate timer for history (less frequent)
    if (historyTimer) {
        clearInterval(historyTimer);
    }
    
    historyTimer = setInterval(() => {
        if (isAutoRefreshEnabled) {
            fetchTradeHistory();
        }
    }, HISTORY_REFRESH_INTERVAL);
    
    // ⭐ NEW: Separate timer for storage status (every 10 seconds)
    const storageTimer = setInterval(() => {
        if (isAutoRefreshEnabled) {
            fetchStorageStatus();
        }
    }, 10000);
    
    updateAutoRefreshStatus(true);
    console.log('Auto-refresh started (Git Storage)');
}

// Stop auto-refresh
function stopAutoRefresh() {
    if (refreshTimer) {
        clearInterval(refreshTimer);
        refreshTimer = null;
    }
    
    if (historyTimer) {
        clearInterval(historyTimer);
        historyTimer = null;
    }
    
    updateAutoRefreshStatus(false);
    console.log('Auto-refresh stopped');
}

// Toggle auto-refresh
function toggleAutoRefresh() {
    isAutoRefreshEnabled = !isAutoRefreshEnabled;
    
    if (isAutoRefreshEnabled) {
        startAutoRefresh();
        showToast('Auto-refresh enabled', 'info');
    } else {
        stopAutoRefresh();
        showToast('Auto-refresh disabled', 'warning');
    }
}

// Helper function to update element text
function updateElementText(elementId, text) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = text;
    }
}

// Helper function to update progress bar
function updateProgressBar(elementId, value, colorClass = 'bg-primary') {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.width = `${Math.min(100, value)}%`;
        element.textContent = `${Math.round(value)}%`;
        
        // Update color class
        element.className = `progress-bar ${colorClass}`;
        if (value > 0 && value < 100) {
            element.classList.add('progress-bar-striped', 'progress-bar-animated');
        }
    }
}

// Get color based on confidence level
function getConfidenceColor(confidence) {
    if (confidence >= 80) return 'bg-success';
    if (confidence >= 60) return 'bg-info';
    if (confidence >= 40) return 'bg-warning';
    return 'bg-danger';
}

// Format time string
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

// Update system status
function updateSystemStatus(status, type = 'info') {
    const element = document.getElementById('systemStatus');
    if (element) {
        element.textContent = status;
        element.className = `badge bg-${type}`;
    }
}

// Update auto-refresh status
function updateAutoRefreshStatus(enabled) {
    const element = document.getElementById('autoRefreshStatus');
    if (element) {
        element.textContent = enabled ? 'Enabled' : 'Disabled';
        element.className = `badge ${enabled ? 'bg-success' : 'bg-secondary'}`;
    }
}

// ⭐ NEW: Update Git storage button status
function updateGitStorageStatus(ready) {
    const element = document.getElementById('gitStorageStatus');
    if (element) {
        element.textContent = ready ? 'Connected' : 'Not Ready';
        element.className = `badge ${ready ? 'bg-success' : 'bg-warning'}`;
    }
}

// Show toast notification with Git storage info
function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toastContainer';
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    // Add Git storage icon for storage-related messages
    const gitIcon = message.includes('Git') || message.includes('storage') || message.includes('repository') ? 
        '<i class="bi bi-git me-1"></i>' : '';
    
    // Create toast element
    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header bg-${type} text-white">
                <strong class="me-auto">
                    ${gitIcon}
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
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;
    
    toastContainer.insertAdjacentHTML('beforeend', toastHtml);
    
    // Initialize and show toast
    const toastEl = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastEl, {
        delay: 3000
    });
    toast.show();
    
    // Remove toast after it's hidden
    toastEl.addEventListener('hidden.bs.toast', function () {
        toastEl.remove();
    });
}

// Clean up on page unload
window.addEventListener('beforeunload', function() {
    stopAutoRefresh();
    
    if (typeof stopChartUpdates === 'function') {
        stopChartUpdates();
    }
});

// Export functions for debugging
window.fetchTradingState = fetchTradingState;
window.fetchTradeHistory = fetchTradeHistory;
window.fetchStorageStatus = fetchStorageStatus; // ⭐ NEW
window.resetTrading = resetTrading;
window.startAutoRefresh = startAutoRefresh;
window.stopAutoRefresh = stopAutoRefresh;
window.toggleAutoRefresh = toggleAutoRefresh;

console.log('Main.js loaded successfully (2-Minute Cycle with Git Storage)');