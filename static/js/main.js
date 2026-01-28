/**
 * EUR/USD Trading System - Main JavaScript Module (Google Sheets Edition)
 * Fixed for 2-Minute Cycles with Google Sheets Data Storage
 */

// Configuration
const API_BASE_URL = '';
const REFRESH_INTERVAL = 2000; // 2 seconds
const HISTORY_REFRESH_INTERVAL = 5000; // 5 seconds
const STORAGE_CHECK_INTERVAL = 30000; // 30 seconds
const CYCLE_DURATION = 120; // ⭐ FIXED: 120 seconds for 2 minutes

// State
let refreshTimer = null;
let historyTimer = null;
let storageTimer = null;
let lastUpdateTime = null;
let isAutoRefreshEnabled = true;
let googleSheetsStatus = 'checking';
let lastSyncTime = null;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('EUR/USD 2-Minute Trading System (Google Sheets Edition) Initialized');
    
    // Initialize systems
    initializeDashboard();
    
    // Load initial data
    fetchTradingState();
    fetchTradeHistory();
    
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
    console.log('Dashboard initialized (2-Minute Cycles, Google Sheets)');
    
    // Set initial values
    updateSystemStatus('Initializing...', 'warning');
    updateAutoRefreshStatus(true);
    updateStorageStatus('checking', 'Checking Google Sheets...');
    
    // Set cycle duration displays
    updateElementText('remainingTime', `${CYCLE_DURATION}s`);
    
    // Update last sync time
    updateElementText('lastSync', 'Never');
    lastSyncTime = new Date();
}

// Set up all event listeners
function setupEventListeners() {
    // Refresh button
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            fetchTradingState();
            fetchTradeHistory();
            showToast('Data refreshed manually', 'info');
        });
    }
    
    // Reset trading button
    const resetBtn = document.getElementById('resetBtn');
    if (resetBtn) {
        resetBtn.addEventListener('click', function() {
            if (confirm('Reset trading statistics? This will clear local stats but preserve Google Sheets data.')) {
                resetTrading();
            }
        });
    }
    
    // Check Google Sheets button
    const checkSheetsBtn = document.getElementById('checkSheetsBtn');
    if (checkSheetsBtn) {
        checkSheetsBtn.addEventListener('click', function() {
            checkGoogleSheetsConnection();
        });
    }
    
    // Sync button
    const syncBtn = document.getElementById('syncBtn');
    if (syncBtn) {
        syncBtn.addEventListener('click', function() {
            fetchTradeHistory();
            showToast('Syncing with Google Sheets...', 'info');
        });
    }
    
    // Export CSV button
    const exportBtn = document.getElementById('exportBtn');
    if (exportBtn) {
        exportBtn.addEventListener('click', function() {
            exportToCSV();
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
            
            // Update storage status from API response
            if (data.google_sheets_status) {
                updateStorageStatusFromAPI(data.google_sheets_status);
            }
        })
        .catch(error => {
            console.error('Error fetching trading state:', error);
            updateSystemStatus('Connection Error', 'danger');
            updateStorageStatus('error', 'Connection Error');
            showToast('Failed to fetch trading data', 'error');
        });
}

// Fetch trade history from API (Google Sheets)
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
            lastSyncTime = new Date();
            updateElementText('lastSync', formatTime(lastSyncTime));
            
            // Update storage status
            if (data.google_sheets_status) {
                updateStorageStatusFromAPI(data.google_sheets_status);
            }
        })
        .catch(error => {
            console.error('Error fetching trade history:', error);
            updateStorageStatus('error', 'Failed to sync');
        });
}

// Check Google Sheets connection
function checkGoogleSheetsConnection() {
    showToast('Checking Google Sheets connection...', 'info');
    
    fetch('/api/health')
        .then(response => response.json())
        .then(data => {
            const status = data.google_sheets_status || 'UNKNOWN';
            updateStorageStatusFromAPI(status);
            showToast(`Google Sheets: ${status}`, 
                     status.includes('CONNECTED') ? 'success' : 'warning');
        })
        .catch(error => {
            console.error('Error checking Google Sheets:', error);
            updateStorageStatus('error', 'Check failed');
            showToast('Failed to check Google Sheets connection', 'danger');
        });
}

// Export trade history to CSV
function exportToCSV() {
    fetch('/api/trade_history')
        .then(response => response.json())
        .then(data => {
            if (!data.trades || data.trades.length === 0) {
                showToast('No trade data to export', 'warning');
                return;
            }
            
            // Convert to CSV
            const headers = ['Trade ID', 'Timestamp', 'Action', 'Entry Price', 'Exit Price', 
                           'Profit (Pips)', 'Result', 'Duration (s)', 'Confidence', 'Signal Strength'];
            
            const csvRows = [
                headers.join(','),
                ...data.trades.map(trade => [
                    trade.trade_id,
                    trade.timestamp,
                    trade.action,
                    trade.entry_price,
                    trade.exit_price || '',
                    trade.profit_pips,
                    trade.result,
                    trade.duration_seconds,
                    trade.confidence,
                    trade.signal_strength
                ].join(','))
            ];
            
            const csvString = csvRows.join('\n');
            const blob = new Blob([csvString], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            
            // Create download link
            const a = document.createElement('a');
            a.href = url;
            a.download = `eurusd_trades_${new Date().toISOString().split('T')[0]}.csv`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            showToast(`Exported ${data.trades.length} trades to CSV`, 'success');
        })
        .catch(error => {
            console.error('Export error:', error);
            showToast('Failed to export data', 'danger');
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
    
    // Update storage status
    const storageStatus = document.getElementById('storageStatus');
    if (storageStatus) {
        const status = data.google_sheets_status || 'UNKNOWN';
        let badgeClass = 'bg-warning';
        let displayText = 'Checking...';
        
        if (status.includes('CONNECTED')) {
            badgeClass = 'bg-success';
            displayText = 'Google Sheets ✓';
        } else if (status.includes('ERROR')) {
            badgeClass = 'bg-danger';
            displayText = 'Storage Error';
        }
        
        storageStatus.textContent = displayText;
        storageStatus.className = `badge ${badgeClass}`;
    }
    
    // Update cache efficiency
    if (data.cache_efficiency) {
        updateElementText('cacheEfficiency', data.cache_efficiency);
    }
    
    // Update TP/SL levels
    updateElementText('optimalTp', data.optimal_tp ? data.optimal_tp.toFixed(5) : '-');
    updateElementText('optimalSl', data.optimal_sl ? data.optimal_sl.toFixed(5) : '-');
    
    const tpPips = data.tp_distance_pips || 0;
    const slPips = data.sl_distance_pips || 0;
    updateElementText('tpPips', `<span class="badge bg-success">${tpPips} pips</span>`);
    updateElementText('slPips', `<span class="badge bg-danger">${slPips} pips</span>`);
    
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
    
    activeTradeDiv.innerHTML = `
        <div class="text-center">
            <h5 class="${trade.action === 'BUY' ? 'text-success' : 'text-danger'}">
                <i class="bi ${trade.action === 'BUY' ? 'bi-arrow-up-circle' : 'bi-arrow-down-circle'}"></i>
                ${trade.action} #${trade.trade_id || trade.id}
            </h5>
            
            <div class="row text-start mt-3">
                <div class="col-6"><small>Entry Price:</small></div>
                <div class="col-6 text-end"><strong>${trade.entry_price ? trade.entry_price.toFixed(5) : '0.00000'}</strong></div>
                
                <div class="col-6"><small>Current P/L:</small></div>
                <div class="col-6 text-end ${profitPips >= 0 ? 'text-success' : 'text-danger'}">
                    <strong>${profitPips.toFixed(1)} pips</strong>
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
                    <i class="bi bi-cloud-check"></i>
                    Data saved to Google Sheets
                </small>
            </div>
        </div>
    `;
}

// Update trade history table (from Google Sheets)
function updateTradeHistory(data) {
    const tbody = document.getElementById('tradeHistory');
    
    if (!data || !data.trades || data.trades.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="10" class="text-center py-4">
                    <i class="bi bi-cloud-arrow-down display-6 d-block text-muted mb-2"></i>
                    <span class="text-muted">No trades in Google Sheets yet</span>
                    <div class="spinner-border spinner-border-sm text-primary mt-2" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </td>
            </tr>
        `;
        
        updateElementText('totalHistoryTrades', '0');
        updateElementText('totalWins', '0');
        
        // Update storage badge
        updateStorageBadge(data.google_sheets_status);
        return;
    }
    
    // Update summary counts
    const totalTrades = data.total || data.trades.length;
    const profitableTrades = data.profitable || data.trades.filter(t => 
        t.result && (t.result === 'SUCCESS' || t.result === 'PARTIAL_SUCCESS' || t.result === 'WIN')
    ).length;
    
    updateElementText('totalHistoryTrades', totalTrades);
    updateElementText('totalWins', profitableTrades);
    
    // Update storage badge
    updateStorageBadge(data.google_sheets_status);
    
    // Build table rows
    let html = '';
    data.trades.slice().reverse().forEach(trade => {
        const timestamp = trade.timestamp || trade.entry_time || new Date().toISOString();
        const entryTime = new Date(timestamp);
        const exitTime = trade.exit_time ? new Date(trade.exit_time) : null;
        const duration = trade.duration_seconds || 
                        (exitTime ? Math.round((exitTime - entryTime) / 1000) : 0);
        const profitPips = trade.profit_pips || 0;
        const confidence = trade.confidence || 0;
        
        // Format time
        const timeStr = entryTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        
        // Determine result badge class
        let resultClass = 'bg-secondary';
        let resultText = trade.result || 'PENDING';
        
        if (trade.result === 'SUCCESS' || trade.result === 'WIN') {
            resultClass = 'bg-success';
        } else if (trade.result === 'FAILED' || trade.result === 'LOSE') {
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
                <td><strong>#${trade.trade_id || trade.id}</strong></td>
                <td>${timeStr}</td>
                <td>
                    <span class="badge ${trade.action === 'BUY' ? 'bg-success' : 'bg-danger'}">
                        ${trade.action}
                    </span>
                </td>
                <td>${trade.entry_price ? trade.entry_price.toFixed(5) : '0.00000'}</td>
                <td>
                    <small class="d-block">TP: ${trade.tp_distance_pips || '0'} pips</small>
                    <small>SL: ${trade.sl_distance_pips || '0'} pips</small>
                </td>
                <td>${trade.exit_price ? trade.exit_price.toFixed(5) : '-'}</td>
                <td class="${profitPips >= 0 ? 'text-success' : 'text-danger'} fw-bold">
                    ${profitPips.toFixed(1)}
                </td>
                <td>${duration.toFixed(1)}s</td>
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
            </tr>
        `;
    });
    
    tbody.innerHTML = html;
}

// Update storage status badge
function updateStorageBadge(status) {
    const badge = document.getElementById('storageBadge');
    if (!badge) return;
    
    let badgeClass = 'bg-warning';
    let badgeIcon = 'bi-cloud';
    let badgeText = 'Sheets';
    
    if (status && status.includes('CONNECTED')) {
        badgeClass = 'bg-success';
        badgeIcon = 'bi-cloud-check';
        badgeText = 'Sheets ✓';
    } else if (status && (status.includes('ERROR') || status.includes('FAILED'))) {
        badgeClass = 'bg-danger';
        badgeIcon = 'bi-cloud-slash';
        badgeText = 'Sheets ✗';
    }
    
    badge.className = `badge ${badgeClass}`;
    badge.innerHTML = `<i class="bi ${badgeIcon}"></i> ${badgeText}`;
}

// Update storage status from API response
function updateStorageStatusFromAPI(status) {
    googleSheetsStatus = status;
    
    // Update sheets status badge
    const sheetsStatus = document.getElementById('sheetsStatus');
    if (sheetsStatus) {
        let badgeClass = 'bg-warning';
        let statusText = 'Sheets: Checking...';
        
        if (status.includes('CONNECTED')) {
            badgeClass = 'bg-success';
            statusText = 'Sheets: Connected';
        } else if (status.includes('ERROR')) {
            badgeClass = 'bg-danger';
            statusText = 'Sheets: Error';
        }
        
        sheetsStatus.className = `badge ${badgeClass}`;
        sheetsStatus.textContent = statusText;
    }
    
    // Update storage indicator in footer
    const storageIndicator = document.getElementById('storageIndicator');
    if (storageIndicator) {
        if (status.includes('CONNECTED')) {
            storageIndicator.className = 'badge bg-success';
            storageIndicator.textContent = 'Google Sheets ✓';
        } else if (status.includes('ERROR')) {
            storageIndicator.className = 'badge bg-danger';
            storageIndicator.textContent = 'Storage Error';
        } else {
            storageIndicator.className = 'badge bg-warning';
            storageIndicator.textContent = 'Storage: Connecting...';
        }
    }
}

// Update storage status
function updateStorageStatus(status, message = '') {
    googleSheetsStatus = status;
    
    const indicator = document.getElementById('storageStatus');
    if (indicator) {
        let badgeClass = 'bg-warning';
        let displayText = 'Checking...';
        
        if (status === 'connected') {
            badgeClass = 'bg-success';
            displayText = 'Google Sheets ✓';
        } else if (status === 'error') {
            badgeClass = 'bg-danger';
            displayText = 'Storage Error';
        }
        
        indicator.className = `badge ${badgeClass}`;
        indicator.textContent = displayText;
    }
}

// Reset trading statistics (preserves Google Sheets data)
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
            showToast('Trading statistics reset (Google Sheets preserved)', 'success');
            fetchTradingState();
            fetchTradeHistory();
        } else {
            showToast('Failed to reset trading', 'error');
        }
    })
    .catch(error => {
        console.error('Error resetting trading:', error);
        showToast('Error resetting trading', 'error');
    });
}

// Start auto-refresh with storage checks
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
    
    // Storage check timer
    if (storageTimer) {
        clearInterval(storageTimer);
    }
    
    storageTimer = setInterval(() => {
        if (isAutoRefreshEnabled) {
            checkGoogleSheetsConnection();
        }
    }, STORAGE_CHECK_INTERVAL);
    
    updateAutoRefreshStatus(true);
    console.log('Auto-refresh started with Google Sheets monitoring');
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
    
    if (storageTimer) {
        clearInterval(storageTimer);
        storageTimer = null;
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
        if (typeof dateString === 'string') {
            const date = new Date(dateString);
            return date.toLocaleTimeString([], { 
                hour: '2-digit', 
                minute: '2-digit',
                second: '2-digit'
            });
        } else if (dateString instanceof Date) {
            return dateString.toLocaleTimeString([], { 
                hour: '2-digit', 
                minute: '2-digit',
                second: '2-digit'
            });
        }
        return '-';
    } catch (e) {
        return '-';
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

// Show toast notification
function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toastContainer');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toastContainer';
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
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
window.resetTrading = resetTrading;
window.checkGoogleSheetsConnection = checkGoogleSheetsConnection;
window.exportToCSV = exportToCSV;
window.startAutoRefresh = startAutoRefresh;
window.stopAutoRefresh = stopAutoRefresh;
window.toggleAutoRefresh = toggleAutoRefresh;

console.log('Main.js loaded successfully (2-Minute Cycle, Google Sheets Edition)');