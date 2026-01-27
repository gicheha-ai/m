// main.js - Advanced Real-time Trading Dashboard
// Compatible with advanced app.py for Render deployment

document.addEventListener('DOMContentLoaded', function() {
    // Global state
    let tradingState = {};
    let tradeHistory = [];
    let mlStatus = {};
    let cacheStatus = {};
    let advancedMetrics = {};
    let eventSource = null;
    let updateInterval = null;
    let autoRefresh = true;
    let lastUpdateTime = null;
    
    // Chart reference
    let currentChart = null;
    
    // DOM Elements cache
    const elements = {
        // Price & Prediction
        currentPrice: document.getElementById('currentPrice'),
        minutePrediction: document.getElementById('minutePrediction'),
        action: document.getElementById('action'),
        confidence: document.getElementById('confidence'),
        confidenceBar: document.getElementById('confidenceBar'),
        confidenceText: document.getElementById('confidenceText'),
        signalStrength: document.getElementById('signalStrength'),
        
        // Cycle & Time
        cycleCount: document.getElementById('cycleCount'),
        nextCycleIn: document.getElementById('nextCycleIn'),
        remainingTime: document.getElementById('remainingTime'),
        lastUpdate: document.getElementById('lastUpdate'),
        serverTime: document.getElementById('serverTime'),
        
        // Trading Stats
        balance: document.getElementById('balance'),
        totalTrades: document.getElementById('totalTrades'),
        profitableTrades: document.getElementById('profitableTrades'),
        winRate: document.getElementById('winRate'),
        totalPips: document.getElementById('totalPips'),
        dailyProfit: document.getElementById('dailyProfit'),
        
        // Market Data
        dataSource: document.getElementById('dataSource'),
        apiStatus: document.getElementById('apiStatus'),
        volatility: document.getElementById('volatility'),
        riskRewardRatio: document.getElementById('riskRewardRatio'),
        
        // Progress Bars
        cycleProgress: document.getElementById('cycleProgress'),
        tradeProgress: document.getElementById('tradeProgress'),
        
        // Active Trade
        activeTrade: document.getElementById('activeTrade'),
        optimalTp: document.getElementById('optimalTp'),
        optimalSl: document.getElementById('optimalSl'),
        tpDistancePips: document.getElementById('tpDistancePips'),
        slDistancePips: document.getElementById('slDistancePips'),
        tradeDuration: document.getElementById('tradeDuration'),
        currentProfit: document.getElementById('currentProfit'),
        
        // ML Status
        mlReady: document.getElementById('mlReady'),
        trainingSamples: document.getElementById('trainingSamples'),
        mlTrainingStatus: document.getElementById('mlTrainingStatus'),
        mlCorrectionsApplied: document.getElementById('mlCorrectionsApplied'),
        mlDataLoadStatus: document.getElementById('mlDataLoadStatus'),
        
        // Cache Status
        cacheHits: document.getElementById('cacheHits'),
        cacheMisses: document.getElementById('cacheMisses'),
        cacheEfficiency: document.getElementById('cacheEfficiency'),
        apiCallsToday: document.getElementById('apiCallsToday'),
        
        // Advanced Metrics
        profitFactor: document.getElementById('profitFactor'),
        expectancy: document.getElementById('expectancy'),
        maxDrawdown: document.getElementById('maxDrawdown'),
        avgWinPips: document.getElementById('avgWinPips'),
        avgLossPips: document.getElementById('avgLossPips'),
        bestTrade: document.getElementById('bestTrade'),
        worstTrade: document.getElementById('worstTrade'),
        consecutiveWins: document.getElementById('consecutiveWins'),
        consecutiveLosses: document.getElementById('consecutiveLosses'),
        
        // History & Tables
        tradeHistoryBody: document.getElementById('tradeHistoryBody'),
        totalHistoryTrades: document.getElementById('totalHistoryTrades'),
        totalWins: document.getElementById('totalWins'),
        
        // System Status
        systemStatus: document.getElementById('systemStatus'),
        systemMessage: document.getElementById('systemMessage'),
        messageText: document.getElementById('messageText'),
        
        // ML Messages
        mlDataMessage: document.getElementById('mlDataMessage'),
        mlTrainingMessage: document.getElementById('mlTrainingMessage'),
        mlCorrectionsMessage: document.getElementById('mlCorrectionsMessage'),
        
        // Chart
        chartContainer: document.getElementById('chart')
    };

    // Initialize dashboard
    function init() {
        console.log('🚀 Advanced Trading Dashboard Initializing...');
        
        // Load initial data
        fetchAllData();
        
        // Start real-time updates
        startRealTimeUpdates();
        
        // Setup event listeners
        setupEventListeners();
        
        // Initialize charts
        initializeChart();
        
        // Show welcome message
        showMessage('Advanced Trading System Connected', 'success');
        
        console.log('✅ Dashboard initialized successfully');
    }
    
    // ==================== DATA FETCHING ====================
    function fetchAllData() {
        fetchTradingState();
        fetchTradeHistory();
        fetchMLStatus();
        fetchCacheStatus();
        fetchAdvancedMetrics();
    }
    
    function fetchTradingState() {
        fetch('/api/trading_state')
            .then(response => {
                if (!response.ok) throw new Error('Network error');
                return response.json();
            })
            .then(data => {
                tradingState = data;
                updateTradingDisplay();
                updateChart();
                updateActiveTradeDisplay();
                updateProgressBars();
                
                // Update last update time
                lastUpdateTime = new Date();
            })
            .catch(error => {
                console.error('Error fetching trading state:', error);
                showMessage('Connection issue - retrying...', 'warning');
            });
    }
    
    function fetchTradeHistory() {
        fetch('/api/trade_history')
            .then(response => response.json())
            .then(data => {
                tradeHistory = data.trades || [];
                updateHistoryTable();
                updateElement('totalHistoryTrades', data.total || 0);
                updateElement('totalWins', data.profitable || 0);
                
                // Update advanced stats if available
                if (data.advanced_stats) {
                    updateAdvancedStatsDisplay(data.advanced_stats);
                }
            })
            .catch(error => {
                console.error('Error fetching trade history:', error);
            });
    }
    
    function fetchMLStatus() {
        fetch('/api/ml_status')
            .then(response => response.json())
            .then(data => {
                mlStatus = data;
                updateMLStatusDisplay();
                
                // Show ML status messages
                if (data.ml_data_load_status) {
                    handleMLStatusMessage(data.ml_data_load_status, 'mlDataMessage');
                }
                if (data.ml_training_status) {
                    handleMLStatusMessage(data.ml_training_status, 'mlTrainingMessage');
                }
            })
            .catch(error => {
                console.error('Error fetching ML status:', error);
            });
    }
    
    function fetchCacheStatus() {
        fetch('/api/cache_status')
            .then(response => response.json())
            .then(data => {
                cacheStatus = data;
                updateCacheStatusDisplay();
            })
            .catch(error => {
                console.error('Error fetching cache status:', error);
            });
    }
    
    function fetchAdvancedMetrics() {
        fetch('/api/advanced_metrics')
            .then(response => response.json())
            .then(data => {
                advancedMetrics = data;
                updateAdvancedMetricsDisplay();
            })
            .catch(error => {
                console.error('Error fetching advanced metrics:', error);
            });
    }
    
    // ==================== REAL-TIME UPDATES ====================
    function startRealTimeUpdates() {
        // Close existing connection if any
        if (eventSource) {
            eventSource.close();
        }
        
        // Start Server-Sent Events
        eventSource = new EventSource('/api/events');
        
        eventSource.onopen = function() {
            console.log('🔗 SSE connection established');
            showMessage('Real-time updates connected', 'success');
        };
        
        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                handleRealTimeUpdate(data);
            } catch (e) {
                console.error('Error parsing SSE data:', e);
            }
        };
        
        eventSource.onerror = function(error) {
            console.error('SSE connection error:', error);
            eventSource.close();
            
            // Fallback to polling
            showMessage('Real-time connection lost, using polling', 'warning');
            startPollingFallback();
        };
        
        // Also start polling as backup
        startPollingFallback();
    }
    
    function handleRealTimeUpdate(data) {
        switch (data.type) {
            case 'update':
                // Update trading display with new data
                updateTradingDisplay(data.data);
                break;
                
            case 'trade':
                if (data.status === 'active') {
                    // Active trade update
                    fetchTradingState();
                }
                break;
                
            case 'notification':
                // Show notification message
                if (data.data && data.data.message) {
                    showNotification(data.data.message, data.data.success ? 'success' : 'error');
                }
                break;
                
            case 'ml_data_saved':
                // ML data saved notification
                showMLMessage('✅ Trade data saved to data.txt', 'success');
                fetchMLStatus();
                break;
                
            case 'trade_executed':
                // New trade executed
                showNotification('🔔 ' + data.message, 'info');
                fetchAllData();
                break;
                
            case 'trade_closed':
                // Trade closed
                showNotification('💰 ' + data.message, data.trade.result === 'SUCCESS' ? 'success' : 'error');
                fetchAllData();
                break;
        }
    }
    
    function startPollingFallback() {
        if (updateInterval) {
            clearInterval(updateInterval);
        }
        
        updateInterval = setInterval(() => {
            if (autoRefresh) {
                fetchTradingState();
                
                // Periodically refresh other data
                const now = new Date();
                if (!lastUpdateTime || (now - lastUpdateTime) > 10000) {
                    fetchAllData();
                }
            }
        }, 3000); // Poll every 3 seconds
        
        console.log('🔄 Polling fallback started (3s interval)');
    }
    
    // ==================== DISPLAY UPDATES ====================
    function updateTradingDisplay(updateData = null) {
        const data = updateData || tradingState;
        
        // Update basic elements
        updateElement('currentPrice', formatPrice(data.current_price));
        updateElement('minutePrediction', data.minute_prediction || 'ANALYZING');
        updateElement('action', data.action || 'WAIT');
        updateElement('confidence', data.confidence?.toFixed(1) + '%' || '0%');
        updateElement('cycleCount', data.cycle_count || 0);
        updateElement('balance', formatCurrency(data.balance));
        updateElement('totalTrades', data.total_trades || 0);
        updateElement('profitableTrades', data.profitable_trades || 0);
        updateElement('winRate', data.win_rate?.toFixed(1) + '%' || '0%');
        updateElement('totalPips', data.total_pips?.toFixed(1) || '0.0');
        updateElement('dailyProfit', formatCurrency(data.daily_profit || 0));
        updateElement('nextCycleIn', data.next_cycle_in + 's' || '120s');
        updateElement('remainingTime', data.remaining_time + 's' || '120s');
        updateElement('dataSource', data.data_source || '-');
        updateElement('apiStatus', data.api_status || 'CONNECTING');
        updateElement('volatility', data.volatility || 'MEDIUM');
        updateElement('riskRewardRatio', data.risk_reward_ratio || '1:2');
        updateElement('lastUpdate', formatTime(data.last_update));
        updateElement('serverTime', formatTime(data.server_time));
        updateElement('systemStatus', data.system_status || 'RUNNING');
        
        // Update confidence bar
        const confidence = data.confidence || 0;
        if (elements.confidenceBar) {
            elements.confidenceBar.style.width = confidence + '%';
            elements.confidenceBar.textContent = confidence.toFixed(1) + '%';
            elements.confidenceBar.className = 'progress-bar ' + getConfidenceClass(confidence);
        }
        
        // Update signal strength
        const signalStrength = data.signal_strength || 0;
        updateElement('signalStrength', '★'.repeat(signalStrength) + '☆'.repeat(3 - signalStrength));
        
        // Update styling based on prediction and action
        updatePredictionStyle(data.minute_prediction);
        updateActionStyle(data.action);
        
        // Update progress bars
        updateProgressBar('cycleProgress', data.cycle_progress || 0);
        updateProgressBar('tradeProgress', data.trade_progress || 0);
    }
    
    function updateActiveTradeDisplay() {
        const trade = tradingState.current_trade;
        
        if (!trade) {
            if (elements.activeTrade) {
                elements.activeTrade.innerHTML = `
                    <div class="text-center text-muted py-4">
                        <i class="bi bi-hourglass-split display-4 d-block mb-3"></i>
                        <span class="fs-5">No active trade</span>
                        <p class="small mt-2">Waiting for trading signal...</p>
                    </div>
                `;
            }
            return;
        }
        
        // Calculate current profit
        const currentPrice = tradingState.current_price || trade.entry_price;
        let currentPips = 0;
        let profitAmount = 0;
        
        if (trade.action === 'BUY') {
            currentPips = (currentPrice - trade.entry_price) * 10000;
        } else {
            currentPips = (trade.entry_price - currentPrice) * 10000;
        }
        
        profitAmount = (currentPips / 10000) * (trade.trade_size || 1000);
        
        // Format duration
        const duration = trade.duration_seconds || 0;
        const durationStr = `${Math.floor(duration)}s`;
        
        // Update trade info
        updateElement('optimalTp', formatPrice(trade.optimal_tp));
        updateElement('optimalSl', formatPrice(trade.optimal_sl));
        updateElement('tpDistancePips', `${trade.tp_distance_pips || 0} pips`);
        updateElement('slDistancePips', `${trade.sl_distance_pips || 0} pips`);
        updateElement('tradeDuration', durationStr);
        updateElement('currentProfit', `${currentPips.toFixed(1)} pips (${formatCurrency(profitAmount)})`);
        
        // Update active trade card
        if (elements.activeTrade) {
            const profitClass = currentPips >= 0 ? 'text-success' : 'text-danger';
            const actionClass = trade.action === 'BUY' ? 'bg-success' : 'bg-danger';
            
            elements.activeTrade.innerHTML = `
                <div class="trade-active-card">
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h5 class="mb-0">
                            <span class="badge ${actionClass} me-2">${trade.action}</span>
                            Trade #${trade.id}
                        </h5>
                        <span class="badge bg-warning">${trade.signal_strength || 1}/3 Signal</span>
                    </div>
                    
                    <div class="row mb-3">
                        <div class="col-6">
                            <small class="text-muted d-block">Entry Price</small>
                            <strong>${formatPrice(trade.entry_price)}</strong>
                        </div>
                        <div class="col-6">
                            <small class="text-muted d-block">Current Price</small>
                            <strong>${formatPrice(currentPrice)}</strong>
                        </div>
                    </div>
                    
                    <div class="row mb-3">
                        <div class="col-6">
                            <small class="text-muted d-block">Take Profit</small>
                            <strong class="text-success">${formatPrice(trade.optimal_tp)}</strong>
                            <small class="d-block">${trade.tp_distance_pips || 0} pips</small>
                        </div>
                        <div class="col-6">
                            <small class="text-muted d-block">Stop Loss</small>
                            <strong class="text-danger">${formatPrice(trade.optimal_sl)}</strong>
                            <small class="d-block">${trade.sl_distance_pips || 0} pips</small>
                        </div>
                    </div>
                    
                    <div class="trade-metrics">
                        <div class="d-flex justify-content-between mb-2">
                            <span>Current P/L:</span>
                            <span class="${profitClass} fw-bold">${currentPips.toFixed(1)} pips</span>
                        </div>
                        <div class="d-flex justify-content-between mb-2">
                            <span>Duration:</span>
                            <span>${durationStr} / 120s</span>
                        </div>
                        <div class="d-flex justify-content-between">
                            <span>Confidence:</span>
                            <span>${trade.confidence?.toFixed(1) || '0'}%</span>
                        </div>
                    </div>
                    
                    ${trade.ml_enhanced ? `
                        <div class="mt-3 text-center">
                            <span class="badge bg-info">
                                <i class="bi bi-cpu me-1"></i> ML Enhanced
                            </span>
                        </div>
                    ` : ''}
                </div>
            `;
        }
    }
    
    function updateMLStatusDisplay() {
        updateElement('trainingSamples', mlStatus.training_samples || '0');
        updateElement('mlCorrectionsApplied', mlStatus.ml_corrections_applied || '0');
        updateElement('mlTrainingStatus', mlStatus.ml_training_status || 'Collecting data...');
        updateElement('mlDataLoadStatus', mlStatus.ml_data_load_status || 'Loading...');
        
        // Update ML ready badge
        if (elements.mlReady) {
            elements.mlReady.textContent = mlStatus.ml_model_ready ? 'YES' : 'NO';
            elements.mlReady.className = `badge ${mlStatus.ml_model_ready ? 'bg-success' : 'bg-danger'}`;
        }
    }
    
    function updateCacheStatusDisplay() {
        updateElement('cacheHits', cacheStatus.cache_hits || '0');
        updateElement('cacheMisses', cacheStatus.cache_misses || '0');
        updateElement('cacheEfficiency', cacheStatus.cache_efficiency || '0%');
        updateElement('apiCallsToday', cacheStatus.api_calls_today || '~320');
    }
    
    function updateAdvancedMetricsDisplay() {
        updateElement('profitFactor', advancedMetrics.profit_factor?.toFixed(2) || '0.00');
        updateElement('expectancy', advancedMetrics.expectancy?.toFixed(1) || '0.0');
        updateElement('maxDrawdown', advancedMetrics.max_drawdown?.toFixed(1) + '%' || '0.0%');
        updateElement('avgWinPips', advancedMetrics.avg_win_pips?.toFixed(1) || '0.0');
        updateElement('avgLossPips', advancedMetrics.avg_loss_pips?.toFixed(1) || '0.0');
        updateElement('bestTrade', advancedMetrics.best_trade_pips?.toFixed(1) || '0.0');
        updateElement('worstTrade', advancedMetrics.worst_trade_pips?.toFixed(1) || '0.0');
        updateElement('consecutiveWins', advancedMetrics.consecutive_wins || '0');
        updateElement('consecutiveLosses', advancedMetrics.consecutive_losses || '0');
    }
    
    function updateAdvancedStatsDisplay(stats) {
        // Optional: Update additional stats from trade history
    }
    
    function updateHistoryTable() {
        const tbody = elements.tradeHistoryBody;
        if (!tbody) return;
        
        if (tradeHistory.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="11" class="text-center py-5">
                        <i class="bi bi-inbox display-4 d-block text-muted mb-3"></i>
                        <span class="text-muted">No trades yet</span>
                        <p class="small mt-2">Trades will appear here after execution</p>
                    </td>
                </tr>
            `;
            return;
        }
        
        let html = '';
        tradeHistory.slice().reverse().forEach(trade => {
            const entryTime = trade.entry_time ? formatTime(trade.entry_time) : '-';
            const exitTime = trade.exit_time ? formatTime(trade.exit_time) : '-';
            const profitPips = trade.profit_pips || 0;
            const profitAmount = trade.profit_amount || 0;
            
            // Determine row and text classes
            const rowClass = getTradeRowClass(trade.result);
            const profitClass = profitPips >= 0 ? 'profit-positive' : 'profit-negative';
            const resultClass = `trade-result-${trade.result?.toLowerCase().replace('_', '-') || 'pending'}`;
            
            html += `
                <tr class="${rowClass}">
                    <td class="fw-bold">#${trade.id}</td>
                    <td>${entryTime}</td>
                    <td><span class="trade-action-${trade.action?.toLowerCase()}">${trade.action}</span></td>
                    <td>${formatPrice(trade.entry_price)}</td>
                    <td>${formatPrice(trade.exit_price) || '-'}</td>
                    <td><span class="${profitClass}">${profitPips >= 0 ? '+' : ''}${profitPips.toFixed(1)}</span></td>
                    <td>${formatCurrency(profitAmount)}</td>
                    <td>${trade.duration_seconds?.toFixed(0) || '0'}s</td>
                    <td><span class="${resultClass}">${trade.result || '-'}</span></td>
                    <td>${trade.confidence?.toFixed(1) || '0'}%</td>
                    <td>${trade.tp_distance_pips || '0'}/${trade.sl_distance_pips || '0'}</td>
                </tr>
            `;
        });
        
        tbody.innerHTML = html;
    }
    
    // ==================== CHART FUNCTIONS ====================
    function initializeChart() {
        console.log('📊 Chart system initialized');
        
        // Create empty chart initially
        if (typeof Plotly !== 'undefined' && elements.chartContainer) {
            const layout = getChartLayout();
            const data = [{
                x: [],
                y: [],
                type: 'scatter',
                mode: 'lines',
                name: 'EUR/USD',
                line: { color: '#00ff88', width: 2 }
            }];
            
            currentChart = Plotly.newPlot(elements.chartContainer, data, layout, {
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            });
        }
    }
    
    function updateChart() {
        if (!tradingState.chart_data || !elements.chartContainer || typeof Plotly === 'undefined') {
            return;
        }
        
        try {
            const chartData = JSON.parse(tradingState.chart_data);
            Plotly.react(elements.chartContainer, chartData.data, chartData.layout);
            
            // Update chart status
            updateElement('chartStatus', 'Live');
            document.getElementById('chartStatus')?.classList.add('bg-success');
            
        } catch (error) {
            console.error('Chart update error:', error);
        }
    }
    
    function getChartLayout() {
        return {
            title: {
                text: 'EUR/USD Live Trading Chart',
                font: { size: 16, color: '#ffffff' }
            },
            xaxis: {
                title: 'Time',
                gridcolor: 'rgba(255,255,255,0.1)',
                color: '#ffffff'
            },
            yaxis: {
                title: 'Price',
                tickformat: '.5f',
                gridcolor: 'rgba(255,255,255,0.1)',
                color: '#ffffff'
            },
            plot_bgcolor: 'rgba(0,0,0,0)',
            paper_bgcolor: 'rgba(0,0,0,0)',
            template: 'plotly_dark',
            showlegend: true,
            legend: {
                x: 0.02,
                y: 1,
                bgcolor: 'rgba(0,0,0,0.5)',
                font: { color: '#ffffff' }
            },
            margin: { l: 50, r: 30, t: 40, b: 40 }
        };
    }
    
    // ==================== EVENT HANDLERS ====================
    function setupEventListeners() {
        // Refresh button
        document.getElementById('refreshBtn')?.addEventListener('click', () => {
            fetchAllData();
            showMessage('Dashboard refreshed', 'info');
        });
        
        // Reset trading
        document.getElementById('resetTradingBtn')?.addEventListener('click', resetTrading);
        
        // Force ML training
        document.getElementById('forceMLTrainingBtn')?.addEventListener('click', forceMLTraining);
        
        // View ML data
        document.getElementById('viewMLDataBtn')?.addEventListener('click', viewMLData);
        
        // Toggle fullscreen chart
        document.getElementById('toggleChartBtn')?.addEventListener('click', toggleChartFullscreen);
        
        // Export chart data
        document.getElementById('exportChartBtn')?.addEventListener('click', exportChartData);
        
        // Auto-refresh toggle
        const autoRefreshToggle = document.getElementById('autoRefreshToggle');
        if (autoRefreshToggle) {
            autoRefreshToggle.checked = autoRefresh;
            autoRefreshToggle.addEventListener('change', function() {
                autoRefresh = this.checked;
                showMessage(`Auto-refresh ${autoRefresh ? 'enabled' : 'disabled'}`, 'info');
            });
        }
        
        // Manual trade buttons (demo)
        document.getElementById('manualBuyBtn')?.addEventListener('click', () => manualTrade('BUY'));
        document.getElementById('manualSellBtn')?.addEventListener('click', () => manualTrade('SELL'));
        
        // Close message button
        document.querySelector('.btn-close')?.addEventListener('click', hideMessage);
        
        // Health check button
        document.getElementById('healthCheckBtn')?.addEventListener('click', checkHealth);
    }
    
    function resetTrading() {
        if (!confirm('Are you sure you want to reset ALL trading data? This cannot be undone.')) {
            return;
        }
        
        showMessage('Resetting trading system...', 'warning');
        
        fetch('/api/reset_trading', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showMessage('✅ Trading system reset successfully', 'success');
                    fetchAllData();
                } else {
                    showMessage('❌ Reset failed: ' + data.message, 'error');
                }
            })
            .catch(error => {
                showMessage('❌ Reset failed: Network error', 'error');
            });
    }
    
    function forceMLTraining() {
        showMessage('Forcing ML training...', 'info');
        
        fetch('/api/force_ml_training', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showMLTrainingMessage('✅ ' + data.message, 'success');
                    showMLCorrectionsMessage(`✅ Applied ${data.corrections || 0} ML corrections`, 'success');
                    fetchMLStatus();
                } else {
                    showMLTrainingMessage('❌ ' + data.message, 'error');
                }
            })
            .catch(error => {
                showMLTrainingMessage('❌ ML training failed', 'error');
            });
    }
    
    function viewMLData() {
        fetch('/api/view_ml_data')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showMessage('Error: ' + data.error, 'error');
                    return;
                }
                
                // Create modal to show ML data
                createMLDataModal(data);
                showMessage('ML data loaded successfully', 'success');
            })
            .catch(error => {
                showMessage('Failed to load ML data', 'error');
            });
    }
    
    function createMLDataModal(data) {
        const modal = document.createElement('div');
        modal.className = 'modal fade show d-block';
        modal.style.backgroundColor = 'rgba(0,0,0,0.7)';
        modal.innerHTML = `
            <div class="modal-dialog modal-lg modal-dialog-scrollable">
                <div class="modal-content bg-dark text-light">
                    <div class="modal-header border-secondary">
                        <h5 class="modal-title">
                            <i class="bi bi-file-earmark-text me-2"></i>
                            ML Training Data (${data.file})
                        </h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <div class="row mb-4">
                            <div class="col-md-6">
                                <div class="card bg-secondary mb-3">
                                    <div class="card-body">
                                        <h6 class="card-title">ML Status</h6>
                                        <p class="mb-1">Ready: <span class="badge ${data.ml_status.ready ? 'bg-success' : 'bg-danger'}">${data.ml_status.ready ? 'YES' : 'NO'}</span></p>
                                        <p class="mb-1">Samples: <strong>${data.ml_status.samples}</strong></p>
                                        <p class="mb-0">Trained on: <strong>${data.ml_status.trained_on}</strong> samples</p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card bg-secondary">
                                    <div class="card-body">
                                        <h6 class="card-title">File Info</h6>
                                        <p class="mb-1">Total lines: <strong>${data.total_lines}</strong></p>
                                        <p class="mb-0">Preview: First ${data.preview.length} trades</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <h6>Data Preview:</h6>
                        <div class="bg-black p-3 rounded" style="max-height: 300px; overflow-y: auto;">
                            <pre class="text-info mb-0">${JSON.stringify(data.preview, null, 2)}</pre>
                        </div>
                    </div>
                    <div class="modal-footer border-secondary">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <button type="button" class="btn btn-primary" onclick="downloadMLData()">
                            <i class="bi bi-download me-1"></i> Download data.txt
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Close modal on background click or close button
        modal.addEventListener('click', function(e) {
            if (e.target === modal || e.target.classList.contains('btn-close')) {
                document.body.removeChild(modal);
            }
        });
        
        // Download function
        window.downloadMLData = function() {
            const dataStr = JSON.stringify(data, null, 2);
            const blob = new Blob([dataStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'ml_data_export.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            showMessage('ML data exported', 'success');
        };
    }
    
    function toggleChartFullscreen() {
        const chartDiv = elements.chartContainer;
        const btn = document.getElementById('toggleChartBtn');
        
        if (!document.fullscreenElement) {
            if (chartDiv.requestFullscreen) {
                chartDiv.requestFullscreen();
                btn.innerHTML = '<i class="bi bi-fullscreen-exit"></i>';
                btn.title = 'Exit Fullscreen';
            }
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
                btn.innerHTML = '<i class="bi bi-arrows-angle-expand"></i>';
                btn.title = 'Fullscreen';
            }
        }
    }
    
    function exportChartData() {
        showMessage('Exporting chart data...', 'info');
        // Implementation would depend on what data you want to export
    }
    
    function manualTrade(action) {
        showMessage(`Manual ${action} trade executed (demo mode)`, 'warning');
        // In a real system, this would call an API endpoint
    }
    
    function checkHealth() {
        fetch('/api/health')
            .then(response => response.json())
            .then(data => {
                showMessage(`System health: ${data.status.toUpperCase()}`, 'success');
                console.log('System health:', data);
            })
            .catch(error => {
                showMessage('Health check failed', 'error');
            });
    }
    
    // ==================== UTILITY FUNCTIONS ====================
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
            const width = Math.min(100, Math.max(0, value));
            bar.style.width = width + '%';
            bar.textContent = Math.round(width) + '%';
            
            // Add warning class if trade is near end
            if (id === 'tradeProgress' && width > 80) {
                bar.classList.add('progress-bar-warning');
            } else {
                bar.classList.remove('progress-bar-warning');
            }
        }
    }
    
    function updatePredictionStyle(prediction) {
        const element = elements.minutePrediction;
        if (element) {
            element.className = 'prediction-display';
            if (prediction === 'BULLISH') {
                element.classList.add('prediction-bullish');
            } else if (prediction === 'BEARISH') {
                element.classList.add('prediction-bearish');
            } else {
                element.classList.add('prediction-neutral');
            }
        }
    }
    
    function updateActionStyle(action) {
        const element = elements.action;
        if (element) {
            element.className = 'action-badge';
            if (action === 'BUY') {
                element.classList.add('action-buy');
            } else if (action === 'SELL') {
                element.classList.add('action-sell');
            } else {
                element.classList.add('action-wait');
            }
        }
    }
    
    function getConfidenceClass(confidence) {
        if (confidence >= 80) return 'bg-success';
        if (confidence >= 60) return 'bg-warning';
        return 'bg-danger';
    }
    
    function getTradeRowClass(result) {
        switch (result) {
            case 'SUCCESS': return 'trade-row-success';
            case 'FAILED': return 'trade-row-failed';
            case 'PARTIAL_SUCCESS': return 'trade-row-partial-success';
            case 'PARTIAL_FAIL': return 'trade-row-partial-fail';
            default: return '';
        }
    }
    
    function formatPrice(price) {
        if (price == null) return '-';
        return parseFloat(price).toFixed(5);
    }
    
    function formatCurrency(amount) {
        if (amount == null) return '$0.00';
        return '$' + parseFloat(amount).toFixed(2);
    }
    
    function formatTime(timestamp) {
        if (!timestamp) return '-';
        try {
            const date = new Date(timestamp);
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        } catch (e) {
            return timestamp;
        }
    }
    
    // ==================== MESSAGE FUNCTIONS ====================
    function showMessage(message, type = 'info') {
        console.log(`${type.toUpperCase()}: ${message}`);
        
        const messageDiv = elements.systemMessage;
        const messageText = elements.messageText;
        
        if (messageDiv && messageText) {
            messageText.textContent = message;
            messageDiv.className = `alert alert-${type} alert-dismissible fade show`;
            messageDiv.style.display = 'block';
            
            // Auto-hide success/info messages
            if (type === 'success' || type === 'info') {
                setTimeout(() => {
                    if (messageDiv.style.display === 'block') {
                        messageDiv.style.display = 'none';
                    }
                }, 5000);
            }
        }
    }
    
    function hideMessage() {
        if (elements.systemMessage) {
            elements.systemMessage.style.display = 'none';
        }
    }
    
    function showNotification(message, type = 'info') {
        // Create toast notification
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        const toastContainer = document.getElementById('toastContainer') || createToastContainer();
        toastContainer.appendChild(toast);
        
        // Initialize and show toast
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        // Remove toast after it's hidden
        toast.addEventListener('hidden.bs.toast', function() {
            toast.remove();
        });
    }
    
    function createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(container);
        return container;
    }
    
    function handleMLStatusMessage(message, elementId) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        if (message.includes('✅') || message.includes('Saved') || message.includes('Loaded')) {
            element.textContent = message;
            element.className = 'alert alert-success alert-sm py-1 mb-2 text-center';
            element.style.display = 'block';
            
            setTimeout(() => {
                element.style.display = 'none';
            }, 5000);
        } else if (message.includes('❌') || message.includes('Error') || message.includes('Failed')) {
            element.textContent = message;
            element.className = 'alert alert-danger alert-sm py-1 mb-2 text-center';
            element.style.display = 'block';
        } else {
            element.textContent = message;
            element.className = 'alert alert-info alert-sm py-1 mb-2 text-center';
            element.style.display = 'block';
        }
    }
    
    function showMLMessage(message, type = 'info') {
        const element = elements.mlDataMessage;
        if (element) {
            element.textContent = message;
            element.className = `alert alert-${type} alert-sm py-1 mb-2 text-center`;
            element.style.display = 'block';
            
            if (type === 'success') {
                setTimeout(() => {
                    element.style.display = 'none';
                }, 5000);
            }
        }
    }
    
    function showMLTrainingMessage(message, type = 'info') {
        const element = elements.mlTrainingMessage;
        if (element) {
            element.textContent = message;
            element.className = `alert alert-${type} alert-sm py-1 mb-2 text-center`;
            element.style.display = 'block';
            
            if (type === 'success') {
                setTimeout(() => {
                    element.style.display = 'none';
                }, 5000);
            }
        }
    }
    
    function showMLCorrectionsMessage(message, type = 'info') {
        const element = elements.mlCorrectionsMessage;
        if (element) {
            element.textContent = message;
            element.className = `alert alert-${type} alert-sm py-1 mb-2 text-center`;
            element.style.display = 'block';
            
            if (type === 'success') {
                setTimeout(() => {
                    element.style.display = 'none';
                }, 5000);
            }
        }
    }
    
    // ==================== INITIALIZATION ====================
    init();
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', function() {
        if (eventSource) {
            eventSource.close();
        }
        if (updateInterval) {
            clearInterval(updateInterval);
        }
    });
});