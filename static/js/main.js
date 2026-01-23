/**
 * EUR/USD Trading System - Main JavaScript Module
 * Updated for 2-Minute Cycles
 */

// Configuration
const API_BASE_URL = '';
const REFRESH_INTERVAL = 2000; // 2 seconds
const HISTORY_REFRESH_INTERVAL = 5000; // 5 seconds
const CYCLE_DURATION = 120; // ⭐ Changed from 60 to 120 seconds

// State
let refreshTimer = null;
let historyTimer = null;
let lastUpdateTime = null;
let isAutoRefreshEnabled = true;
let cycleDuration = CYCLE_DURATION;

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('EUR/USD 2-Minute Trading System Initialized');  // ⭐ 2-Minute
    
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
    
    // Set cycle info
    updateCycleInfo();
});

// Initialize dashboard elements
function initializeDashboard() {
    console.log('Dashboard initialized (2-Minute Cycles)');  // ⭐ 2-Minute
    
    // Set initial values
    updateSystemStatus('Initializing...', 'warning');
    updateAutoRefreshStatus(true);
    updateElementText('cycleDuration', `${cycleDuration}s`);
    updateElementText('cycleTime', `${cycleDuration}s`);
    updateElementText('chartTimeframe', `${cycleDuration}s`);
    updateElementText('cycleInfo', `${cycleDuration}s Cycles`);
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
            if (confirm('Are you sure you want to reset all trading statistics? This will clear trade history and reset balance to $10,000.')) {
                resetTrading();
            }
        });
    }
    
    // API Status button
    const apiStatusBtn = document.getElementById('apiStatusBtn');
    if (apiStatusBtn) {
        apiStatusBtn.addEventListener('click', function() {
            checkApiStatus();
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

// Update dashboard with new data
function updateDashboard(data) {
    if (!data) return;
    
    // Update cycle information
    updateElementText('cycleCount', data.cycle_count || 0);
    updateElementText('nextCycle', `${data.next_cycle_in || cycleDuration}s`);
    
    // Update progress bar (scaled to 120 seconds)
    const cycleProgress = data.cycle_progress || 0;
    updateProgressBar('cycleProgress', cycleProgress, 'bg-info');
    
    // Update time
    updateElementText('lastUpdate', data.last_update || '-');
    updateElementText('serverTime', formatTime(data.server_time));
    
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
    
    // Update API calls info
    const apiCalls = document.getElementById('apiCalls');
    if (apiCalls) {
        const isSafe = data.api_status === 'CONNECTED' || data.api_status === 'DEMO';
        apiCalls.textContent = isSafe ? '720 (SAFE)' : 'Limit Exceeded';
        apiCalls.className = `badge ${isSafe ? 'bg-success' : 'bg-danger'}`;
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
    updateElementText('predictionAccuracy', `${(data.prediction_accuracy || 0).toFixed(1)}%`);
    updateElementText('cycleDurationStat', `${cycleDuration}s`);
    
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
    if (typeof updateChartFromState === 'function') {
        updateChartFromState(data);
    }
    
    // Update price history if available
    if (data.price_history && Array.isArray(data.price_history)) {
        updatePriceHistory(data.price_history);
    }
    
    // Update trade status
    updateElementText('tradeStatus', data.trade_status || 'NO_TRADE');
    
    // Update volatility if available
    if (data.volatility) {
        updateElementText('volatility', data.volatility);
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
                <p class="small mt-2">Waiting for next 2-minute trading signal...</p>
            </div>
        `;
        return;
    }
    
    const profitPips = trade.profit_pips || 0;
    const duration = trade.duration_seconds || 0;
    const timeRemaining = Math.max(0, cycleDuration - duration);
    const profitPercent = trade.profit_amount ? ((trade.profit_amount / trade.trade_size) * 100).toFixed(3) : '0.000';
    
    // Calculate progress percentage (based on 120 seconds)
    const progressPercent = Math.min(100, (duration / cycleDuration) * 100);
    
    activeTradeDiv.innerHTML = `
        <div class="text-center">
            <h5 class="${trade.action === 'BUY' ? 'text-success' : 'text-danger'}">
                <i class="bi ${trade.action === 'BUY' ? 'bi-arrow-up-circle' : 'bi-arrow-down-circle'}"></i>
                ${trade.action} #${trade.id} (2-Minute)
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
                
                <div class="col-6"><small>Cycle:</small></div>
                <div class="col-6 text-end"><strong>${cycleDuration}s</strong></div>
                
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
                <small class="text-muted d-block mt-1">Trade progress: ${Math.round(progressPercent)}% (${cycleDuration}s cycle)</small>
            </div>
            
            <div class="mt-3">
                <small class="text-muted">
                    <i class="bi bi-info-circle"></i>
                    Target: TP at ${trade.optimal_tp.toFixed(5)} (${trade.tp_distance_pips} pips) | 
                    Max Loss: SL at ${trade.optimal_sl.toFixed(5)} (${trade.sl_distance_pips} pips)
                </small>
            </div>
        </div>
    `;
}

// Update trade history table
function updateTradeHistory(data) {
    const tbody = document.getElementById('tradeHistory');
    
    if (!data || !data.trades || data.trades.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="10" class="text-center py-4">
                    <i class="bi bi-inbox display-6 d-block text-muted mb-2"></i>
                    <span class="text-muted">No trades yet</span>
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
    
    // Build table rows
    let html = '';
    data.trades.slice(0, 20).reverse().forEach(trade => { // Show latest 20 trades
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
        
        html += `
            <tr>
                <td><strong>#${trade.id}</strong></td>
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
                <td>${duration}s/${cycleDuration}s</td>  <!-- ⭐ Added cycle duration -->
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

// Check API status
function checkApiStatus() {
    fetch('/api/api_status')
        .then(response => response.json())
        .then(data => {
            showToast(`API Status: ${data.api_limits.status}`, 'info');
            console.log('API Status:', data);
        })
        .catch(error => {
            console.error('Error checking API status:', error);
        });
}

// Update cycle information
function updateCycleInfo() {
    updateElementText('cycleDuration', `${cycleDuration}s`);
    updateElementText('cycleTime', `${cycleDuration}s`);
    updateElementText('chartTimeframe', `${cycleDuration}s`);
    updateElementText('cycleInfo', `${cycleDuration}s Cycles`);
}

// Export functions for debugging
window.fetchTradingState = fetchTradingState;
window.fetchTradeHistory = fetchTradeHistory;
window.resetTrading = resetTrading;
window.startAutoRefresh = startAutoRefresh;
window.stopAutoRefresh = stopAutoRefresh;
window.toggleAutoRefresh = toggleAutoRefresh;
window.checkApiStatus = checkApiStatus;

console.log('Main.js loaded successfully (2-Minute Cycle)');