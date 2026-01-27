// main.js - Real-time Trading Dashboard
document.addEventListener('DOMContentLoaded', function() {
    let tradingState = {};
    let tradeHistory = [];
    let eventSource = null;
    let updateInterval = null;
    let autoRefresh = true;
    
    // Initialize
    function init() {
        console.log('Initializing real-time dashboard...');
        
        // Load initial data
        fetchAllData();
        
        // Start real-time updates
        startRealTimeUpdates();
        
        // Setup event listeners
        setupEventListeners();
        
        // Initialize charts
        initializeCharts();
        
        showMessage('Dashboard connected', 'success');
    }
    
    // Fetch all initial data
    function fetchAllData() {
        fetchTradingState();
        fetchTradeHistory();
        fetchMLStatus();
        fetchCacheStatus();
    }
    
    // Start Server-Sent Events for real-time updates
    function startRealTimeUpdates() {
        if (eventSource) {
            eventSource.close();
        }
        
        eventSource = new EventSource('/api/events');
        
        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                handleRealTimeUpdate(data);
            } catch (e) {
                console.error('SSE parse error:', e);
            }
        };
        
        eventSource.onerror = function(error) {
            console.error('SSE error:', error);
            eventSource.close();
            
            // Fallback to polling
            setTimeout(() => {
                startRealTimeUpdates();
            }, 5000);
        };
        
        // Also set up polling as fallback
        if (updateInterval) {
            clearInterval(updateInterval);
        }
        
        updateInterval = setInterval(() => {
            if (autoRefresh) {
                fetchTradingState();
            }
        }, 3000);
    }
    
    // Handle real-time updates
    function handleRealTimeUpdate(data) {
        switch (data.type) {
            case 'state_update':
                updateTradingDisplay(data.data);
                break;
            case 'new_trade':
                fetchTradeHistory();
                showMessage('New trade completed!', 'success');
                break;
            case 'queue_update':
                if (data.data && data.data.message) {
                    showMessage(data.data.message, 'info');
                }
                fetchAllData();
                break;
        }
    }
    
    // Fetch trading state
    function fetchTradingState() {
        fetch('/api/trading_state')
            .then(response => response.json())
            .then(data => {
                tradingState = data;
                updateTradingDisplay();
                updateCharts();
            })
            .catch(error => {
                console.error('State fetch error:', error);
            });
    }
    
    // Update display
    function updateTradingDisplay(updateData = null) {
        const data = updateData || tradingState;
        
        // Basic info
        updateElement('currentPrice', data.current_price?.toFixed(5) || '1.08500');
        updateElement('prediction', data.minute_prediction || 'ANALYZING');
        updateElement('action', data.action || 'WAIT');
        updateElement('confidenceText', data.confidence?.toFixed(1) + '%' || '0%');
        updateElement('cycleCount', data.cycle_count || 0);
        updateElement('balance', '$' + (data.balance?.toFixed(2) || '10000.00'));
        updateElement('totalTrades', data.total_trades || 0);
        updateElement('profitableTrades', data.profitable_trades || 0);
        updateElement('winRate', data.win_rate?.toFixed(1) + '%' || '0%');
        updateElement('nextCycle', data.next_cycle_in + 's' || '120s');
        updateElement('apiStatus', data.api_status || 'CONNECTING');
        updateElement('dataSource', data.data_source || '-');
        updateElement('lastUpdate', data.last_update || '-');
        updateElement('serverTime', formatTime(data.server_time));
        updateElement('remainingTime', data.remaining_time + 's' || '120s');
        
        // Confidence bar
        const confidenceBar = document.getElementById('confidenceBar');
        if (confidenceBar) {
            confidenceBar.style.width = (data.confidence || 0) + '%';
            confidenceBar.textContent = (data.confidence || 0).toFixed(1) + '%';
            confidenceBar.className = 'progress-bar ' + getConfidenceClass(data.confidence);
        }
        
        // Signal strength
        updateElement('signalStrength', '★'.repeat(data.signal_strength || 0));
        
        // Progress bars
        updateProgressBar('cycleProgress', data.cycle_progress || 0);
        updateProgressBar('tradeProgress', data.trade_progress || 0);
        
        // TP/SL info
        updateElement('optimalTp', data.optimal_tp?.toFixed(5) || '-');
        updateElement('optimalSl', data.optimal_sl?.toFixed(5) || '-');
        updateElement('tpPips', `<span class="badge bg-success">${data.tp_distance_pips || 0} pips</span>`);
        updateElement('slPips', `<span class="badge bg-danger">${data.sl_distance_pips || 0} pips</span>`);
        
        // Trade status
        updateTradeStatus(data);
        
        // Update styling
        updatePredictionStyle(data.minute_prediction);
        updateActionStyle(data.action);
    }
    
    // Update trade status display
    function updateTradeStatus(data) {
        const activeTradeDiv = document.getElementById('activeTrade');
        
        if (data.current_trade) {
            const trade = data.current_trade;
            const duration = Math.floor((new Date() - new Date(trade.entry_time)) / 1000);
            
            activeTradeDiv.innerHTML = `
                <div class="text-center">
                    <h5 class="text-${trade.action === 'BUY' ? 'success' : 'danger'}">
                        <i class="bi bi-${trade.action === 'BUY' ? 'arrow-up' : 'arrow-down'}-circle"></i>
                        ${trade.action} #${trade.id}
                    </h5>
                    <div class="my-3">
                        <div class="row">
                            <div class="col-6">
                                <small class="text-muted d-block">Entry</small>
                                <strong>${trade.entry_price?.toFixed(5) || '-'}</strong>
                            </div>
                            <div class="col-6">
                                <small class="text-muted d-block">Current</small>
                                <strong>${data.current_price?.toFixed(5) || '-'}</strong>
                            </div>
                        </div>
                    </div>
                    <div class="mt-3">
                        <small class="text-muted">Duration: ${duration}s / ${data.cycle_duration || 120}s</small>
                        <div class="progress mt-1" style="height: 6px;">
                            <div class="progress-bar ${trade.action === 'BUY' ? 'bg-success' : 'bg-danger'}" 
                                 style="width: ${data.trade_progress || 0}%"></div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            activeTradeDiv.innerHTML = `
                <div class="text-center text-muted py-4">
                    <i class="bi bi-hourglass-split display-4 d-block mb-3"></i>
                    <span class="fs-5">No active trade</span>
                    <p class="small mt-2">Waiting for next trading signal...</p>
                </div>
            `;
        }
    }
    
    // Fetch trade history
    function fetchTradeHistory() {
        fetch('/api/trade_history')
            .then(response => response.json())
            .then(data => {
                tradeHistory = data.trades || [];
                updateHistoryTable();
                updateElement('totalHistoryTrades', data.total || 0);
                updateElement('totalWins', data.profitable || 0);
            })
            .catch(error => {
                console.error('History fetch error:', error);
            });
    }
    
    // Update history table
    function updateHistoryTable() {
        const tbody = document.getElementById('tradeHistory');
        if (!tbody) return;
        
        if (tradeHistory.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="10" class="text-center py-4">
                        <i class="bi bi-inbox display-6 d-block text-muted mb-2"></i>
                        <span class="text-muted">No trades yet</span>
                    </td>
                </tr>
            `;
            return;
        }
        
        let html = '';
        tradeHistory.slice().reverse().forEach(trade => {
            const entryTime = trade.entry_time ? formatTime(trade.entry_time) : '-';
            const exitTime = trade.exit_time ? formatTime(trade.exit_time) : '-';
            const profit = trade.profit_pips || 0;
            const profitClass = profit >= 0 ? 'profit-positive' : 'profit-negative';
            const resultClass = `result-${trade.result?.toLowerCase().replace('_', '-') || 'pending'}`;
            
            html += `
                <tr class="trade-${trade.result === 'SUCCESS' ? 'success' : trade.result === 'FAILED' ? 'failed' : 'partial'}">
                    <td>${trade.id || '--'}</td>
                    <td>${entryTime}</td>
                    <td><span class="action-${trade.action?.toLowerCase() || 'none'}">${trade.action || '--'}</span></td>
                    <td>${trade.entry_price?.toFixed(5) || '--'}</td>
                    <td>${trade.tp_distance_pips || '--'}/${trade.sl_distance_pips || '--'}</td>
                    <td>${trade.exit_price?.toFixed(5) || '--'}</td>
                    <td class="${profitClass}">${profit >= 0 ? '+' : ''}${profit.toFixed(1)}</td>
                    <td>${trade.duration_seconds?.toFixed(0) || '--'}s</td>
                    <td><span class="${resultClass}">${trade.result || '--'}</span></td>
                    <td>${trade.confidence?.toFixed(1) || '--'}%</td>
                </tr>
            `;
        });
        
        tbody.innerHTML = html;
    }
    
    // Fetch ML status
    function fetchMLStatus() {
        fetch('/api/ml_status')
            .then(response => response.json())
            .then(data => {
                updateElement('trainingSamples', data.training_samples || '0');
                updateElement('mlCorrectionsApplied', data.ml_corrections_applied || '0');
                updateElement('mlTrainingStatus', data.ml_training_status || 'Not trained yet');
                
                // Update ML ready badge
                const mlReady = document.getElementById('mlReady');
                if (mlReady) {
                    mlReady.textContent = data.ml_model_ready ? 'YES' : 'NO';
                    mlReady.className = `badge ${data.ml_model_ready ? 'bg-success' : 'bg-danger'}`;
                }
                
                // Show ML messages
                if (data.ml_data_load_status) {
                    if (data.ml_data_load_status.includes('✅') || data.ml_data_load_status.includes('Saved')) {
                        showMLDataMessage(data.ml_data_load_status, 'success');
                    }
                }
            })
            .catch(error => {
                console.error('ML status error:', error);
            });
    }
    
    // Fetch cache status
    function fetchCacheStatus() {
        fetch('/api/cache_status')
            .then(response => response.json())
            .then(data => {
                updateElement('cacheHits', data.cache_hits || '0');
                updateElement('cacheMisses', data.cache_misses || '0');
                updateElement('cacheEfficiency', data.cache_efficiency || '0%');
                updateElement('apiCallsToday', data.api_calls_today || '~240 (SAFE)');
            })
            .catch(error => {
                console.error('Cache status error:', error);
            });
    }
    
    // Setup event listeners
    function setupEventListeners() {
        // Refresh button
        document.getElementById('refreshBtn')?.addEventListener('click', fetchAllData);
        
        // Reset trading
        document.getElementById('resetTrading')?.addEventListener('click', function() {
            if (confirm('Reset all trading statistics?')) {
                fetch('/api/reset_trading', { method: 'POST' })
                    .then(response => response.json())
                    .then(data => {
                        showMessage('Trading reset successfully', 'success');
                        fetchAllData();
                    });
            }
        });
        
        // Force ML training
        document.getElementById('forceMLTraining')?.addEventListener('click', function() {
            fetch('/api/force_ml_training', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showMLTrainingMessage('✅ ' + data.message, 'success');
                        fetchMLStatus();
                    } else {
                        showMLTrainingMessage('❌ ' + data.message, 'error');
                    }
                });
        });
        
        // View ML data
        document.getElementById('viewMLData')?.addEventListener('click', function() {
            showMessage('ML data loaded from data.txt', 'info');
        });
        
        // Auto-refresh toggle
        const toggle = document.getElementById('autoRefreshToggle');
        if (toggle) {
            toggle.checked = autoRefresh;
            toggle.addEventListener('change', function() {
                autoRefresh = this.checked;
                showMessage(`Auto-refresh ${autoRefresh ? 'enabled' : 'disabled'}`, 'info');
            });
        }
        
        // Manual trade buttons
        document.getElementById('manualBuy')?.addEventListener('click', function() {
            showMessage('Manual BUY not implemented in demo', 'warning');
        });
        
        document.getElementById('manualSell')?.addEventListener('click', function() {
            showMessage('Manual SELL not implemented in demo', 'warning');
        });
    }
    
    // Chart functions
    function initializeCharts() {
        console.log('Charts initialized');
    }
    
    function updateCharts() {
        if (tradingState.chart_data) {
            try {
                const chartData = JSON.parse(tradingState.chart_data);
                if (typeof Plotly !== 'undefined') {
                    Plotly.react('chart', chartData.data, chartData.layout);
                    updateElement('chartStatus', 'Live');
                    document.getElementById('chartStatus')?.classList.add('bg-success');
                }
            } catch (e) {
                console.error('Chart update error:', e);
            }
        }
    }
    
    // Utility functions
    function updateElement(id, content) {
        const element = document.getElementById(id);
        if (element) {
            if (typeof content === 'string' && content.includes('<')) {
                element.innerHTML = content;
            } else {
                element.textContent = content;
            }
        }
    }
    
    function updateProgressBar(id, value) {
        const bar = document.getElementById(id);
        if (bar) {
            bar.style.width = value + '%';
            bar.textContent = Math.round(value) + '%';
        }
    }
    
    function updatePredictionStyle(prediction) {
        const element = document.getElementById('prediction');
        if (element) {
            element.className = 'display-6 fw-bold ' + 
                (prediction === 'BULLISH' ? 'text-success' : 
                 prediction === 'BEARISH' ? 'text-danger' : 'text-warning');
        }
    }
    
    function updateActionStyle(action) {
        const element = document.getElementById('action');
        if (element) {
            element.className = `badge fs-5 p-2 ${action === 'BUY' ? 'bg-success' : action === 'SELL' ? 'bg-danger' : 'bg-secondary'}`;
        }
    }
    
    function getConfidenceClass(confidence) {
        if (confidence >= 80) return 'bg-success';
        if (confidence >= 60) return 'bg-warning';
        return 'bg-danger';
    }
    
    function formatTime(timestamp) {
        if (!timestamp) return '-';
        try {
            const date = new Date(timestamp);
            return date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        } catch (e) {
            return timestamp;
        }
    }
    
    // Message functions
    function showMessage(message, type = 'info') {
        console.log(`${type}: ${message}`);
        
        const messageDiv = document.getElementById('systemMessage');
        const messageText = document.getElementById('messageText');
        
        if (messageDiv && messageText) {
            messageText.textContent = message;
            messageDiv.className = `alert alert-${type} d-flex align-items-center justify-content-between`;
            messageDiv.style.display = 'flex';
            
            // Auto-hide success/info messages
            if (type === 'success' || type === 'info') {
                setTimeout(() => {
                    messageDiv.style.display = 'none';
                }, 3000);
            }
        }
    }
    
    function showMLDataMessage(message, type = 'info') {
        const element = document.getElementById('mlDataMessage');
        if (element) {
            element.textContent = message;
            element.className = `alert alert-${type} alert-sm mb-2 py-1 text-center`;
            element.style.display = 'block';
            
            if (type === 'success') {
                setTimeout(() => {
                    element.style.display = 'none';
                }, 3000);
            }
        }
    }
    
    function showMLTrainingMessage(message, type = 'info') {
        const element = document.getElementById('mlTrainingMessage');
        if (element) {
            element.textContent = message;
            element.className = `alert alert-${type} alert-sm mb-2 py-1 text-center`;
            element.style.display = 'block';
            
            if (type === 'success') {
                setTimeout(() => {
                    element.style.display = 'none';
                }, 3000);
            }
        }
    }
    
    // Initialize
    init();
});