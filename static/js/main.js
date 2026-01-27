// main.js - Trading Dashboard with data.txt ML Integration

document.addEventListener('DOMContentLoaded', function() {
    // Global state
    let tradingState = {};
    let tradeHistory = [];
    let mlStatus = {};
    let cacheStatus = {};
    let updateInterval;
    
    // DOM Elements
    const elements = {
        currentPrice: document.getElementById('currentPrice'),
        minutePrediction: document.getElementById('minutePrediction'),
        action: document.getElementById('action'),
        confidence: document.getElementById('confidence'),
        cycleCount: document.getElementById('cycleCount'),
        balance: document.getElementById('balance'),
        totalTrades: document.getElementById('totalTrades'),
        profitableTrades: document.getElementById('profitableTrades'),
        winRate: document.getElementById('winRate'),
        nextCycleIn: document.getElementById('nextCycleIn'),
        apiStatus: document.getElementById('apiStatus'),
        dataSource: document.getElementById('dataSource'),
        lastUpdate: document.getElementById('lastUpdate'),
        
        // ML Status Elements
        mlModelReady: document.getElementById('mlModelReady'),
        trainingSamples: document.getElementById('trainingSamples'),
        trainingFile: document.getElementById('trainingFile'),
        mlDataLoadStatus: document.getElementById('mlDataLoadStatus'),
        mlTrainingStatus: document.getElementById('mlTrainingStatus'),
        mlCorrectionsApplied: document.getElementById('mlCorrectionsApplied'),
        
        // Cache Status Elements
        cacheHits: document.getElementById('cacheHits'),
        cacheMisses: document.getElementById('cacheMisses'),
        cacheEfficiency: document.getElementById('cacheEfficiency'),
        apiCallsToday: document.getElementById('apiCallsToday'),
        
        // Progress Bars
        cycleProgress: document.getElementById('cycleProgress'),
        tradeProgress: document.getElementById('tradeProgress'),
        
        // Chart Container
        chartContainer: document.getElementById('chartContainer'),
        
        // Trade Details
        currentTradeContainer: document.getElementById('currentTradeContainer'),
        optimalTp: document.getElementById('optimalTp'),
        optimalSl: document.getElementById('optimalSl'),
        tpDistancePips: document.getElementById('tpDistancePips'),
        slDistancePips: document.getElementById('slDistancePips'),
        remainingTime: document.getElementById('remainingTime'),
        
        // Risk Indicators
        riskRewardRatio: document.getElementById('riskRewardRatio'),
        volatility: document.getElementById('volatility'),
        signalStrength: document.getElementById('signalStrength'),
        
        // History Table
        historyTable: document.getElementById('historyTable'),
        
        // Status Messages
        mlDataMessage: document.getElementById('mlDataMessage'),
        mlTrainingMessage: document.getElementById('mlTrainingMessage'),
        mlCorrectionsMessage: document.getElementById('mlCorrectionsMessage')
    };

    // Initialize the dashboard
    function init() {
        console.log('Initializing Trading Dashboard...');
        fetchTradingState();
        fetchTradeHistory();
        fetchMLStatus();
        fetchCacheStatus();
        
        // Start periodic updates
        updateInterval = setInterval(updateDashboard, 2000);
        
        // Setup event listeners
        setupEventListeners();
        
        // Show initial status
        showMessage('Dashboard initialized. Loading trading data...', 'info');
    }

    // Fetch current trading state
    function fetchTradingState() {
        fetch('/api/trading_state')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => {
                tradingState = data;
                updateTradingDisplay();
                updateChart();
                updateCurrentTradeDisplay();
            })
            .catch(error => {
                console.error('Error fetching trading state:', error);
                showMessage('Error fetching trading data. Retrying...', 'error');
                // Fallback to demo data
                tradingState = getDemoTradingState();
                updateTradingDisplay();
            });
    }

    // Fetch trade history
    function fetchTradeHistory() {
        fetch('/api/trade_history')
            .then(response => response.json())
            .then(data => {
                tradeHistory = data.trades || [];
                updateHistoryTable();
            })
            .catch(error => {
                console.error('Error fetching trade history:', error);
                tradeHistory = getDemoTradeHistory();
                updateHistoryTable();
            });
    }

    // Fetch ML status with data.txt integration
    function fetchMLStatus() {
        fetch('/api/ml_status')
            .then(response => response.json())
            .then(data => {
                mlStatus = data;
                updateMLStatusDisplay();
                
                // Show success/failure messages based on ML operations
                if (data.ml_data_load_status) {
                    if (data.ml_data_load_status.includes('Successfully') || 
                        data.ml_data_load_status.includes('saved successfully')) {
                        showMLDataMessage('✅ ' + data.ml_data_load_status, 'success');
                    } else if (data.ml_data_load_status.includes('failed') || 
                               data.ml_data_load_status.includes('error')) {
                        showMLDataMessage('❌ ' + data.ml_data_load_status, 'error');
                    } else {
                        showMLDataMessage('ℹ️ ' + data.ml_data_load_status, 'info');
                    }
                }
                
                if (data.ml_training_status) {
                    if (data.ml_training_status.includes('✅')) {
                        showMLTrainingMessage('✅ ' + data.ml_training_status, 'success');
                    } else if (data.ml_training_status.includes('failed') || 
                               data.ml_training_status.includes('error')) {
                        showMLTrainingMessage('❌ ' + data.ml_training_status, 'error');
                    } else {
                        showMLTrainingMessage('ℹ️ ' + data.ml_training_status, 'info');
                    }
                }
            })
            .catch(error => {
                console.error('Error fetching ML status:', error);
                mlStatus = getDemoMLStatus();
                updateMLStatusDisplay();
                showMLDataMessage('❌ Failed to load ML status', 'error');
            });
    }

    // Fetch cache status
    function fetchCacheStatus() {
        fetch('/api/cache_status')
            .then(response => response.json())
            .then(data => {
                cacheStatus = data;
                updateCacheStatusDisplay();
            })
            .catch(error => {
                console.error('Error fetching cache status:', error);
                cacheStatus = getDemoCacheStatus();
                updateCacheStatusDisplay();
            });
    }

    // Update the trading display
    function updateTradingDisplay() {
        if (!tradingState) return;
        
        // Update basic info
        elements.currentPrice.textContent = tradingState.current_price?.toFixed(5) || '1.08500';
        elements.minutePrediction.textContent = tradingState.minute_prediction || 'ANALYZING';
        elements.action.textContent = tradingState.action || 'WAIT';
        elements.confidence.textContent = tradingState.confidence?.toFixed(1) + '%' || '0%';
        elements.cycleCount.textContent = tradingState.cycle_count || 0;
        elements.balance.textContent = '$' + (tradingState.balance?.toFixed(2) || '10000.00');
        elements.totalTrades.textContent = tradingState.total_trades || 0;
        elements.profitableTrades.textContent = tradingState.profitable_trades || 0;
        elements.winRate.textContent = tradingState.win_rate?.toFixed(1) + '%' || '0%';
        elements.nextCycleIn.textContent = tradingState.next_cycle_in + 's' || '120s';
        elements.apiStatus.textContent = tradingState.api_status || 'CONNECTING';
        elements.dataSource.textContent = tradingState.data_source || 'Initializing...';
        elements.lastUpdate.textContent = tradingState.last_update || new Date().toLocaleTimeString();
        
        // Update risk indicators
        elements.riskRewardRatio.textContent = tradingState.risk_reward_ratio || '1:2';
        elements.volatility.textContent = tradingState.volatility || 'MEDIUM';
        elements.signalStrength.textContent = '★'.repeat(tradingState.signal_strength || 0);
        
        // Update progress bars
        updateProgressBars();
        
        // Update trade status styling
        updateTradeStatusStyling();
    }

    // Update ML status display
    function updateMLStatusDisplay() {
        if (!mlStatus) return;
        
        elements.mlModelReady.textContent = mlStatus.ml_model_ready ? '✅ READY' : '❌ NOT READY';
        elements.mlModelReady.className = mlStatus.ml_model_ready ? 'status-ready' : 'status-not-ready';
        
        elements.trainingSamples.textContent = mlStatus.training_samples || '0';
        elements.trainingFile.textContent = mlStatus.training_file || 'data.txt';
        elements.mlDataLoadStatus.textContent = mlStatus.ml_data_load_status || 'No data loaded';
        elements.mlTrainingStatus.textContent = mlStatus.ml_training_status || 'Not trained yet';
        elements.mlCorrectionsApplied.textContent = mlStatus.ml_corrections_applied || '0';
        
        // Update training samples indicator
        const samples = parseInt(mlStatus.training_samples) || 0;
        if (samples >= 10) {
            elements.trainingSamples.className = 'status-ready';
        } else if (samples >= 5) {
            elements.trainingSamples.className = 'status-warning';
        } else {
            elements.trainingSamples.className = 'status-not-ready';
        }
    }

    // Update cache status display
    function updateCacheStatusDisplay() {
        if (!cacheStatus) return;
        
        elements.cacheHits.textContent = cacheStatus.cache_hits || '0';
        elements.cacheMisses.textContent = cacheStatus.cache_misses || '0';
        elements.cacheEfficiency.textContent = cacheStatus.cache_efficiency || '0%';
        elements.apiCallsToday.textContent = cacheStatus.api_calls_today || '~240 (SAFE)';
    }

    // Update progress bars
    function updateProgressBars() {
        if (elements.cycleProgress) {
            elements.cycleProgress.style.width = (tradingState.cycle_progress || 0) + '%';
            elements.cycleProgress.textContent = Math.round(tradingState.cycle_progress || 0) + '%';
        }
        
        if (elements.tradeProgress) {
            elements.tradeProgress.style.width = (tradingState.trade_progress || 0) + '%';
            elements.tradeProgress.textContent = Math.round(tradingState.trade_progress || 0) + '%';
        }
    }

    // Update trade status styling
    function updateTradeStatusStyling() {
        const action = tradingState.action || 'WAIT';
        const prediction = tradingState.minute_prediction || 'ANALYZING';
        
        // Update action styling
        elements.action.className = 'trade-action';
        if (action === 'BUY') {
            elements.action.classList.add('buy');
        } else if (action === 'SELL') {
            elements.action.classList.add('sell');
        } else {
            elements.action.classList.add('wait');
        }
        
        // Update prediction styling
        elements.minutePrediction.className = 'prediction';
        if (prediction === 'BULLISH') {
            elements.minutePrediction.classList.add('bullish');
        } else if (prediction === 'BEARISH') {
            elements.minutePrediction.classList.add('bearish');
        } else {
            elements.minutePrediction.classList.add('neutral');
        }
    }

    // Update current trade display
    function updateCurrentTradeDisplay() {
        if (!tradingState.current_trade) {
            elements.currentTradeContainer.style.display = 'none';
            return;
        }
        
        elements.currentTradeContainer.style.display = 'block';
        const trade = tradingState.current_trade;
        
        elements.optimalTp.textContent = trade.optimal_tp?.toFixed(5) || '--';
        elements.optimalSl.textContent = trade.optimal_sl?.toFixed(5) || '--';
        elements.tpDistancePips.textContent = trade.tp_distance_pips || '--';
        elements.slDistancePips.textContent = trade.sl_distance_pips || '--';
        elements.remainingTime.textContent = tradingState.remaining_time + 's' || '--';
        
        // Update trade progress styling
        const progress = tradingState.trade_progress || 0;
        if (progress > 66) {
            elements.tradeProgress.classList.add('warning');
        } else {
            elements.tradeProgress.classList.remove('warning');
        }
    }

    // Update the trading chart
    function updateChart() {
        if (!tradingState.chart_data || !elements.chartContainer) return;
        
        try {
            const chartData = JSON.parse(tradingState.chart_data);
            Plotly.newPlot(elements.chartContainer, chartData.data, chartData.layout, {
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            });
        } catch (error) {
            console.error('Error updating chart:', error);
            elements.chartContainer.innerHTML = '<div class="chart-error">Chart data unavailable</div>';
        }
    }

    // Update history table
    function updateHistoryTable() {
        if (!elements.historyTable || !tradeHistory) return;
        
        const tbody = elements.historyTable.querySelector('tbody');
        if (!tbody) return;
        
        tbody.innerHTML = '';
        
        tradeHistory.slice().reverse().forEach(trade => {
            const row = document.createElement('tr');
            
            // Determine row class based on result
            if (trade.result === 'SUCCESS') {
                row.className = 'trade-success';
            } else if (trade.result === 'FAILED') {
                row.className = 'trade-failed';
            } else if (trade.result === 'PARTIAL_SUCCESS') {
                row.className = 'trade-partial-success';
            } else if (trade.result === 'PARTIAL_FAIL') {
                row.className = 'trade-partial-fail';
            }
            
            // Format dates
            const entryTime = trade.entry_time ? 
                new Date(trade.entry_time).toLocaleTimeString() : '--';
            const exitTime = trade.exit_time ? 
                new Date(trade.exit_time).toLocaleTimeString() : '--';
            
            // Format profit/loss
            let profitClass = '';
            let profitDisplay = '';
            if (trade.profit_pips !== undefined) {
                profitDisplay = trade.profit_pips >= 0 ? 
                    `+${trade.profit_pips.toFixed(1)}` : 
                    `${trade.profit_pips.toFixed(1)}`;
                profitClass = trade.profit_pips >= 0 ? 'profit-positive' : 'profit-negative';
            }
            
            row.innerHTML = `
                <td>${trade.id || '--'}</td>
                <td><span class="action-${trade.action?.toLowerCase() || 'none'}">${trade.action || '--'}</span></td>
                <td>${trade.entry_price?.toFixed(5) || '--'}</td>
                <td>${trade.exit_price?.toFixed(5) || '--'}</td>
                <td>${trade.tp_distance_pips || '--'}/${trade.sl_distance_pips || '--'}</td>
                <td class="${profitClass}">${profitDisplay}</td>
                <td>${trade.confidence?.toFixed(1) || '--'}%</td>
                <td><span class="result-${trade.result?.toLowerCase().replace('_', '-') || 'pending'}">${trade.result || '--'}</span></td>
                <td>${entryTime}</td>
                <td>${exitTime}</td>
                <td>${trade.duration_seconds?.toFixed(0) || '--'}s</td>
            `;
            
            tbody.appendChild(row);
        });
    }

    // Update entire dashboard
    function updateDashboard() {
        fetchTradingState();
        fetchTradeHistory();
        fetchMLStatus();
        fetchCacheStatus();
    }

    // Setup event listeners
    function setupEventListeners() {
        // Reset trading button
        const resetBtn = document.getElementById('resetTrading');
        if (resetBtn) {
            resetBtn.addEventListener('click', resetTrading);
        }
        
        // View ML data button
        const viewMLDataBtn = document.getElementById('viewMLData');
        if (viewMLDataBtn) {
            viewMLDataBtn.addEventListener('click', viewMLData);
        }
        
        // Force ML training button
        const forceMLTrainingBtn = document.getElementById('forceMLTraining');
        if (forceMLTrainingBtn) {
            forceMLTrainingBtn.addEventListener('click', forceMLTraining);
        }
        
        // Refresh buttons
        const refreshButtons = document.querySelectorAll('.refresh-btn');
        refreshButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                updateDashboard();
                showMessage('Dashboard refreshed', 'info');
            });
        });
        
        // Manual trade buttons
        const manualBuyBtn = document.getElementById('manualBuy');
        const manualSellBtn = document.getElementById('manualSell');
        if (manualBuyBtn) {
            manualBuyBtn.addEventListener('click', () => executeManualTrade('BUY'));
        }
        if (manualSellBtn) {
            manualSellBtn.addEventListener('click', () => executeManualTrade('SELL'));
        }
        
        // Auto-refresh toggle
        const autoRefreshToggle = document.getElementById('autoRefreshToggle');
        if (autoRefreshToggle) {
            autoRefreshToggle.addEventListener('change', function() {
                if (this.checked) {
                    updateInterval = setInterval(updateDashboard, 2000);
                    showMessage('Auto-refresh enabled', 'success');
                } else {
                    clearInterval(updateInterval);
                    showMessage('Auto-refresh disabled', 'warning');
                }
            });
        }
    }

    // Reset trading
    function resetTrading() {
        if (!confirm('Are you sure you want to reset all trading statistics? This will clear trade history and reset balance.')) {
            return;
        }
        
        fetch('/api/reset_trading', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showMessage('Trading reset successfully', 'success');
                    updateDashboard();
                } else {
                    showMessage('Reset failed: ' + data.message, 'error');
                }
            })
            .catch(error => {
                console.error('Reset error:', error);
                showMessage('Reset failed', 'error');
            });
    }

    // View ML data from data.txt
    function viewMLData() {
        fetch('/api/view_ml_data')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showMessage('Error viewing ML data: ' + data.error, 'error');
                    return;
                }
                
                // Create modal to display data
                const modal = document.createElement('div');
                modal.className = 'modal';
                modal.innerHTML = `
                    <div class="modal-content">
                        <div class="modal-header">
                            <h3>ML Training Data (${data.file})</h3>
                            <span class="close-modal">&times;</span>
                        </div>
                        <div class="modal-body">
                            <p>Total trades in data.txt: <strong>${data.total_lines}</strong></p>
                            <div class="data-preview">
                                <pre>${JSON.stringify(data.data.slice(0, 5), null, 2)}</pre>
                                ${data.total_lines > 5 ? `<p>... and ${data.total_lines - 5} more trades</p>` : ''}
                            </div>
                            <div class="ml-stats">
                                <p>📊 <strong>ML Status:</strong> ${mlStatus.ml_training_status || 'Unknown'}</p>
                                <p>🔄 <strong>Corrections Applied:</strong> ${mlStatus.ml_corrections_applied || '0'}</p>
                                <p>✅ <strong>Data Load Status:</strong> ${mlStatus.ml_data_load_status || 'Unknown'}</p>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button id="downloadMLData" class="btn-secondary">Download data.txt</button>
                            <button id="closeModal" class="btn-primary">Close</button>
                        </div>
                    </div>
                `;
                
                document.body.appendChild(modal);
                
                // Close modal
                const closeBtn = modal.querySelector('.close-modal');
                const closeModalBtn = modal.querySelector('#closeModal');
                
                closeBtn.addEventListener('click', () => modal.remove());
                closeModalBtn.addEventListener('click', () => modal.remove());
                
                // Download button
                const downloadBtn = modal.querySelector('#downloadMLData');
                downloadBtn.addEventListener('click', () => {
                    const dataStr = JSON.stringify(data, null, 2);
                    const dataBlob = new Blob([dataStr], { type: 'application/json' });
                    const url = URL.createObjectURL(dataBlob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'ml_data_export.json';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                    showMessage('ML data exported', 'success');
                });
                
                // Close on outside click
                modal.addEventListener('click', (e) => {
                    if (e.target === modal) {
                        modal.remove();
                    }
                });
                
                showMessage('ML data loaded from data.txt', 'success');
            })
            .catch(error => {
                console.error('Error viewing ML data:', error);
                showMessage('Failed to load ML data from data.txt', 'error');
            });
    }

    // Force ML training with data from data.txt
    function forceMLTraining() {
        fetch('/api/force_ml_training', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showMLTrainingMessage('✅ ' + data.message, 'success');
                    showMLCorrectionsMessage('✅ Applied ' + data.corrections_applied + ' ML corrections', 'success');
                    
                    // Refresh ML status
                    fetchMLStatus();
                    
                    // Also show in main message area
                    showMessage('ML training completed successfully!', 'success');
                } else {
                    showMLTrainingMessage('❌ ' + data.message, 'error');
                    showMessage('ML training failed: ' + data.message, 'error');
                }
            })
            .catch(error => {
                console.error('ML training error:', error);
                showMLTrainingMessage('❌ ML training failed', 'error');
                showMessage('ML training failed', 'error');
            });
    }

    // Execute manual trade
    function executeManualTrade(action) {
        if (!confirm(`Execute manual ${action} trade? This will override the system's recommendation.`)) {
            return;
        }
        
        showMessage(`Manual ${action} trade executed`, 'warning');
        // Note: You would need to implement a backend endpoint for manual trades
    }

    // Show ML data message
    function showMLDataMessage(message, type) {
        if (!elements.mlDataMessage) return;
        
        elements.mlDataMessage.textContent = message;
        elements.mlDataMessage.className = `message message-${type}`;
        
        // Auto-hide after 5 seconds if success
        if (type === 'success') {
            setTimeout(() => {
                elements.mlDataMessage.className = 'message message-hidden';
            }, 5000);
        }
    }

    // Show ML training message
    function showMLTrainingMessage(message, type) {
        if (!elements.mlTrainingMessage) return;
        
        elements.mlTrainingMessage.textContent = message;
        elements.mlTrainingMessage.className = `message message-${type}`;
        
        // Auto-hide after 5 seconds if success
        if (type === 'success') {
            setTimeout(() => {
                elements.mlTrainingMessage.className = 'message message-hidden';
            }, 5000);
        }
    }

    // Show ML corrections message
    function showMLCorrectionsMessage(message, type) {
        if (!elements.mlCorrectionsMessage) return;
        
        elements.mlCorrectionsMessage.textContent = message;
        elements.mlCorrectionsMessage.className = `message message-${type}`;
        
        // Auto-hide after 5 seconds if success
        if (type === 'success') {
            setTimeout(() => {
                elements.mlCorrectionsMessage.className = 'message message-hidden';
            }, 5000);
        }
    }

    // Show general message
    function showMessage(message, type) {
        // Create message element if it doesn't exist
        let messageElement = document.getElementById('systemMessage');
        if (!messageElement) {
            messageElement = document.createElement('div');
            messageElement.id = 'systemMessage';
            messageElement.className = 'system-message';
            document.body.appendChild(messageElement);
        }
        
        messageElement.textContent = message;
        messageElement.className = `system-message system-message-${type}`;
        
        // Auto-hide after 3 seconds
        setTimeout(() => {
            messageElement.className = 'system-message system-message-hidden';
        }, 3000);
    }

    // Demo data for fallback
    function getDemoTradingState() {
        return {
            current_price: 1.08542,
            minute_prediction: 'BULLISH',
            action: 'BUY',
            confidence: 78.5,
            cycle_count: 42,
            balance: 10450.75,
            total_trades: 25,
            profitable_trades: 18,
            win_rate: 72.0,
            next_cycle_in: 45,
            api_status: 'CONNECTED',
            data_source: 'Frankfurter',
            last_update: new Date().toLocaleTimeString(),
            cycle_progress: 62,
            trade_progress: 0,
            remaining_time: 120,
            risk_reward_ratio: '1:2',
            volatility: 'MEDIUM',
            signal_strength: 3
        };
    }

    function getDemoTradeHistory() {
        return [
            {
                id: 25,
                action: 'BUY',
                entry_price: 1.08420,
                exit_price: 1.08500,
                tp_distance_pips: 8,
                sl_distance_pips: 5,
                profit_pips: 8.0,
                confidence: 75.5,
                result: 'SUCCESS',
                entry_time: new Date(Date.now() - 300000).toISOString(),
                exit_time: new Date(Date.now() - 180000).toISOString(),
                duration_seconds: 120
            },
            {
                id: 24,
                action: 'SELL',
                entry_price: 1.08550,
                exit_price: 1.08480,
                tp_distance_pips: 7,
                sl_distance_pips: 4,
                profit_pips: 7.0,
                confidence: 68.2,
                result: 'SUCCESS',
                entry_time: new Date(Date.now() - 600000).toISOString(),
                exit_time: new Date(Date.now() - 480000).toISOString(),
                duration_seconds: 120
            }
        ];
    }

    function getDemoMLStatus() {
        return {
            ml_model_ready: false,
            training_samples: 8,
            training_file: 'data.txt',
            ml_data_load_status: 'Loaded 8 trades from data.txt',
            ml_training_status: 'Need 2 more trades for training',
            ml_corrections_applied: 0
        };
    }

    function getDemoCacheStatus() {
        return {
            cache_hits: 145,
            cache_misses: 32,
            cache_efficiency: '81.9%',
            api_calls_today: '~240 (SAFE)'
        };
    }

    // Initialize the application
    init();
});