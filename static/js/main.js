// Trading System Main JavaScript
class TradingSystem {
    constructor() {
        this.state = {
            currentPrice: 1.08500,
            previousPrice: 1.08500,
            prediction: 'ANALYZING',
            action: 'WAIT',
            confidence: 0,
            balance: 10000,
            totalProfit: 0,
            winRate: 0,
            totalTrades: 0,
            profitableTrades: 0,
            mlAccuracy: 0,
            mlSamples: 0,
            mlCorrections: 0,
            learningRate: 0,
            cycleCount: 0,
            nextCycle: 120,
            cacheEfficiency: '0%',
            apiCalls: '~240/day',
            tradeStatus: 'NO_TRADE',
            signalStrength: 0,
            systemStatus: 'INITIALIZING',
            activeTrade: null,
            tradeHistory: [],
            chartData: null,
            chartType: 'price',
            priceHistory: [],
            mlProgress: 0,
            accuracyProgress: 0,
            modelConfidence: 0,
            dataQuality: 0,
            lastUpdate: null,
            eventSource: null,
            updateInterval: null,
            chartInstance: null,
            mlChartInstance: null
        };

        this.init();
    }

    async init() {
        console.log('🚀 Initializing Trading System...');
        
        // Set up real-time updates via Server-Sent Events
        this.setupEventSource();
        
        // Initial data fetch
        await this.fetchTradingState();
        await this.fetchTradeHistory();
        await this.fetchMLStatus();
        
        // Set up periodic updates
        this.updateInterval = setInterval(() => this.updateAllData(), 2000);
        
        // Set up chart refresh
        setInterval(() => this.updateChart(), 5000);
        
        // Update ML learning visualization
        setInterval(() => this.updateMLVisualization(), 3000);
        
        console.log('✅ Trading System initialized');
    }

    setupEventSource() {
        if (this.state.eventSource) {
            this.state.eventSource.close();
        }

        this.state.eventSource = new EventSource('/api/events');
        
        this.state.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleEvent(data);
            } catch (e) {
                console.error('Error parsing SSE data:', e);
            }
        };

        this.state.eventSource.onerror = (error) => {
            console.error('SSE connection error:', error);
            this.showNotification('Connection lost. Reconnecting...', 'warning');
            
            // Reconnect after 5 seconds
            setTimeout(() => this.setupEventSource(), 5000);
        };
    }

    handleEvent(data) {
        switch (data.type) {
            case 'update':
                this.handleStateUpdate(data.data);
                break;
            case 'trade':
                this.handleTradeUpdate(data);
                break;
            case 'notification':
                this.showNotification(data.data.message, data.data.success ? 'success' : 'error');
                break;
            case 'ml_update':
                this.handleMLUpdate(data.data);
                break;
        }
    }

    async fetchTradingState() {
        try {
            const response = await fetch('/api/trading_state');
            const data = await response.json();
            this.handleStateUpdate(data);
        } catch (error) {
            console.error('Error fetching trading state:', error);
            this.showNotification('Failed to fetch trading state', 'error');
        }
    }

    async fetchTradeHistory() {
        try {
            const response = await fetch('/api/trade_history');
            const data = await response.json();
            this.state.tradeHistory = data.trades || [];
            this.updateTradeHistoryTable();
        } catch (error) {
            console.error('Error fetching trade history:', error);
        }
    }

    async fetchMLStatus() {
        try {
            const response = await fetch('/api/ml_status');
            const data = await response.json();
            this.updateMLStatus(data);
        } catch (error) {
            console.error('Error fetching ML status:', error);
        }
    }

    handleStateUpdate(data) {
        // Calculate price change
        const priceChange = data.current_price - this.state.currentPrice;
        
        // Update state
        this.state.previousPrice = this.state.currentPrice;
        this.state.currentPrice = data.current_price;
        this.state.prediction = data.minute_prediction;
        this.state.action = data.action;
        this.state.confidence = data.confidence;
        this.state.balance = data.balance;
        this.state.totalProfit = data.total_profit;
        this.state.winRate = data.win_rate;
        this.state.totalTrades = data.total_trades;
        this.state.profitableTrades = data.profitable_trades;
        this.state.tradeStatus = data.trade_status;
        this.state.signalStrength = data.signal_strength;
        this.state.cycleCount = data.cycle_count;
        this.state.nextCycle = data.next_cycle_in;
        this.state.cacheEfficiency = data.cache_efficiency;
        this.state.apiCalls = data.api_calls_today;
        this.state.systemStatus = data.system_status;
        this.state.activeTrade = data.current_trade;
        this.state.lastUpdate = new Date().toLocaleTimeString();
        
        // Update price source
        const source = data.data_source || 'Unknown';
        
        // Calculate ML metrics (simulated for demo)
        this.calculateMLMetrics(data);
        
        // Update UI
        this.updateUI(priceChange);
        
        // Update chart if data available
        if (data.chart_data) {
            this.state.chartData = JSON.parse(data.chart_data);
            this.updateChart();
        }
    }

    calculateMLMetrics(data) {
        // Simulate ML accuracy improvement over time
        const baseAccuracy = 50; // Start at 50%
        const improvementPerTrade = 0.5; // 0.5% improvement per trade
        const maxAccuracy = 95; // Max 95% accuracy
        
        // Calculate accuracy based on trades
        const tradeCount = data.total_trades || 0;
        const profitableCount = data.profitable_trades || 0;
        
        // Basic accuracy calculation
        let accuracy = baseAccuracy + (tradeCount * improvementPerTrade);
        
        // Adjust based on win rate
        if (tradeCount > 0) {
            const winRate = (profitableCount / tradeCount) * 100;
            accuracy += (winRate - 50) * 0.1; // Adjust based on actual performance
        }
        
        // Cap accuracy
        accuracy = Math.min(maxAccuracy, Math.max(baseAccuracy, accuracy));
        
        // Update ML state
        this.state.mlAccuracy = Math.round(accuracy);
        this.state.mlSamples = data.ml_data_points || 0;
        this.state.mlCorrections = data.ml_corrections_applied || 0;
        
        // Calculate learning progress
        this.state.mlProgress = Math.min(100, (this.state.mlSamples / 10) * 100);
        this.state.accuracyProgress = this.state.mlAccuracy;
        this.state.modelConfidence = Math.min(100, (this.state.mlAccuracy / 95) * 100);
        this.state.dataQuality = Math.min(100, (this.state.mlSamples / 50) * 100);
        
        // Calculate learning rate (improvement per trade)
        this.state.learningRate = tradeCount > 0 ? 
            ((this.state.mlAccuracy - baseAccuracy) / tradeCount).toFixed(2) : 0;
    }

    updateUI(priceChange) {
        // Update price
        document.getElementById('current-price').textContent = this.state.currentPrice.toFixed(5);
        document.getElementById('price-source').textContent = `Source: ${this.state.systemStatus}`;
        
        // Update price change
        const priceChangeElement = document.getElementById('price-change');
        priceChangeElement.textContent = `${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(4)}`;
        priceChangeElement.className = `stat-change ${priceChange >= 0 ? 'positive' : 'negative'}`;
        
        // Update prediction
        const predictionElement = document.getElementById('prediction');
        predictionElement.textContent = this.state.prediction;
        predictionElement.style.color = this.getPredictionColor(this.state.prediction);
        
        // Update confidence
        document.getElementById('confidence').textContent = `${this.state.confidence.toFixed(1)}%`;
        
        // Update signal strength
        this.updateSignalStrength(this.state.signalStrength);
        
        // Update action
        const actionElement = document.getElementById('action');
        actionElement.textContent = this.state.action;
        actionElement.style.color = this.getActionColor(this.state.action);
        
        // Update trade status
        document.getElementById('trade-status').textContent = this.state.tradeStatus;
        document.getElementById('signal-value').textContent = `${this.state.signalStrength}/3`;
        
        // Update balance and profit
        document.getElementById('balance').textContent = `$${this.state.balance.toFixed(2)}`;
        document.getElementById('total-profit').textContent = `$${this.state.totalProfit.toFixed(2)}`;
        
        // Update profit change
        const profitChangeElement = document.getElementById('profit-change');
        profitChangeElement.textContent = `${this.state.totalProfit >= 0 ? '+$' : '-$'}${Math.abs(this.state.totalProfit).toFixed(2)}`;
        profitChangeElement.className = `stat-change ${this.state.totalProfit >= 0 ? 'positive' : 'negative'}`;
        
        // Update performance
        document.getElementById('win-rate').textContent = `${this.state.winRate.toFixed(1)}%`;
        document.getElementById('total-trades').textContent = this.state.totalTrades;
        document.getElementById('profitable-trades').textContent = this.state.profitableTrades;
        
        // Update ML metrics
        document.getElementById('ml-accuracy').textContent = `${this.state.mlAccuracy}%`;
        document.getElementById('ml-samples').textContent = this.state.mlSamples;
        document.getElementById('ml-corrections').textContent = this.state.mlCorrections;
        document.getElementById('learning-rate').textContent = `${this.state.learningRate}%`;
        
        // Update system status
        document.getElementById('system-status-text').textContent = this.state.systemStatus;
        document.getElementById('cycle-count').textContent = this.state.cycleCount;
        document.getElementById('next-cycle').textContent = this.state.nextCycle;
        document.getElementById('cache-efficiency').textContent = this.state.cacheEfficiency;
        document.getElementById('api-calls').textContent = this.state.apiCalls;
        
        // Update status indicator
        const indicator = document.getElementById('system-status-indicator');
        indicator.className = `status-indicator ${this.getStatusColor(this.state.systemStatus)}`;
        
        // Update ML progress bars
        this.updateProgressBars();
        
        // Update active trade panel
        this.updateActiveTradePanel();
    }

    updateSignalStrength(strength) {
        const signalElement = document.getElementById('signal-strength');
        const dots = signalElement.querySelectorAll('.signal-dot');
        
        dots.forEach((dot, index) => {
            dot.classList.toggle('active', index < strength);
        });
    }

    updateProgressBars() {
        // Training progress
        document.getElementById('training-progress').style.width = `${this.state.mlProgress}%`;
        document.getElementById('training-progress-text').textContent = 
            `${this.state.mlSamples}/10 trades`;
        
        // Accuracy progress
        document.getElementById('accuracy-progress').style.width = `${this.state.accuracyProgress}%`;
        document.getElementById('accuracy-text').textContent = `${this.state.mlAccuracy}%`;
        
        // Model confidence
        document.getElementById('model-confidence').style.width = `${this.state.modelConfidence}%`;
        document.getElementById('model-confidence-text').textContent = `${Math.round(this.state.modelConfidence)}%`;
        
        // ML stats
        document.getElementById('ml-corrections-count').textContent = this.state.mlCorrections;
        document.getElementById('learning-rate-value').textContent = `${this.state.learningRate}%`;
        document.getElementById('data-quality').textContent = `${this.state.dataQuality}%`;
    }

    updateActiveTradePanel() {
        const tradePanel = document.getElementById('trade-panel');
        const tradeDetails = document.getElementById('trade-details');
        const tradeInfo = document.getElementById('trade-info');
        
        if (this.state.activeTrade) {
            // Trade is active
            tradePanel.classList.add('trade-active');
            tradeDetails.style.display = 'none';
            tradeInfo.style.display = 'grid';
            
            const trade = this.state.activeTrade;
            const currentPrice = this.state.currentPrice;
            
            // Calculate current P/L
            let currentPL = 0;
            if (trade.action === 'BUY') {
                currentPL = (currentPrice - trade.entry_price) * 10000;
            } else {
                currentPL = (trade.entry_price - currentPrice) * 10000;
            }
            
            // Calculate time remaining
            const entryTime = new Date(trade.entry_time);
            const now = new Date();
            const elapsed = (now - entryTime) / 1000;
            const remaining = Math.max(0, 120 - elapsed);
            
            // Calculate progress percentage
            const progress = (elapsed / 120) * 100;
            
            // Update trade info
            document.getElementById('trade-entry').textContent = trade.entry_price.toFixed(5);
            document.getElementById('trade-pl').textContent = `${currentPL >= 0 ? '+' : ''}${currentPL.toFixed(1)} pips`;
            document.getElementById('trade-pl').style.color = currentPL >= 0 ? 'var(--primary-color)' : 'var(--danger-color)';
            document.getElementById('trade-tp').textContent = trade.optimal_tp.toFixed(5);
            document.getElementById('trade-sl').textContent = trade.optimal_sl.toFixed(5);
            document.getElementById('tp-distance').textContent = `${trade.tp_distance_pips} pips`;
            document.getElementById('sl-distance').textContent = `${trade.sl_distance_pips} pips`;
            document.getElementById('trade-time').textContent = `${Math.round(remaining)}s`;
            document.getElementById('trade-progress-bar').style.width = `${progress}%`;
            
        } else {
            // No active trade
            tradePanel.classList.remove('trade-active');
            tradeDetails.style.display = 'block';
            tradeInfo.style.display = 'none';
        }
    }

    updateTradeHistoryTable() {
        const tbody = document.getElementById('trade-history-body');
        const trades = this.state.tradeHistory.slice(-10).reverse(); // Show last 10 trades, newest first
        
        if (trades.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="8" style="text-align: center; padding: 30px; color: #666;">
                        <i class="fas fa-clock"></i> Waiting for trades...
                    </td>
                </tr>
            `;
            return;
        }
        
        tbody.innerHTML = trades.map(trade => `
            <tr>
                <td>${trade.id || '-'}</td>
                <td>
                    <span style="color: ${trade.action === 'BUY' ? 'var(--primary-color)' : 'var(--danger-color)'}; 
                          font-weight: bold;">
                        ${trade.action || '-'}
                    </span>
                </td>
                <td>${trade.entry_price ? trade.entry_price.toFixed(5) : '-'}</td>
                <td>${trade.exit_price ? trade.exit_price.toFixed(5) : '-'}</td>
                <td class="${trade.profit_pips >= 0 ? 'profit-positive' : 'profit-negative'}">
                    ${trade.profit_pips ? `${trade.profit_pips >= 0 ? '+' : ''}${trade.profit_pips.toFixed(1)}` : '-'}
                </td>
                <td>
                    <span style="color: ${this.getResultColor(trade.result)}; font-weight: bold;">
                        ${trade.result || '-'}
                    </span>
                </td>
                <td>${trade.confidence ? `${trade.confidence.toFixed(1)}%` : '-'}</td>
                <td>${trade.entry_time ? new Date(trade.entry_time).toLocaleTimeString() : '-'}</td>
            </tr>
        `).join('');
    }

    updateChart() {
        if (!this.state.chartData) return;
        
        try {
            if (this.state.chartInstance) {
                Plotly.react('chart', this.state.chartData.data, this.state.chartData.layout);
            } else {
                this.state.chartInstance = Plotly.newPlot('chart', this.state.chartData.data, this.state.chartData.layout);
            }
        } catch (error) {
            console.error('Error updating chart:', error);
        }
    }

    updateMLVisualization() {
        // This would update ML-specific visualizations
        // For now, we just update the ML progress bars
        this.updateProgressBars();
    }

    async updateAllData() {
        await this.fetchTradingState();
        await this.fetchTradeHistory();
        await this.fetchMLStatus();
    }

    // Utility functions
    getPredictionColor(prediction) {
        switch(prediction) {
            case 'BULLISH': return 'var(--primary-color)';
            case 'BEARISH': return 'var(--danger-color)';
            case 'NEUTRAL': return 'var(--warning-color)';
            default: return '#fff';
        }
    }

    getActionColor(action) {
        switch(action) {
            case 'BUY': return 'var(--primary-color)';
            case 'SELL': return 'var(--danger-color)';
            case 'WAIT': return 'var(--warning-color)';
            default: return '#fff';
        }
    }

    getStatusColor(status) {
        if (status.includes('RUNNING') || status.includes('CONNECTED')) return 'status-green';
        if (status.includes('INITIALIZING') || status.includes('CACHED')) return 'status-yellow';
        return 'status-red';
    }

    getResultColor(result) {
        switch(result) {
            case 'SUCCESS': return 'var(--primary-color)';
            case 'FAILED': return 'var(--danger-color)';
            case 'PARTIAL_SUCCESS': return 'var(--warning-color)';
            default: return '#fff';
        }
    }

    showNotification(message, type = 'success') {
        const notification = document.getElementById('notification');
        const messageElement = document.getElementById('notification-message');
        
        messageElement.textContent = message;
        notification.className = `notification ${type}`;
        notification.style.display = 'block';
        
        // Auto-hide after 5 seconds
        setTimeout(() => {
            notification.style.display = 'none';
        }, 5000);
    }

    // User actions
    async refreshAllData() {
        this.showNotification('Refreshing all data...', 'success');
        await this.updateAllData();
        this.showNotification('Data refreshed successfully', 'success');
    }

    async viewTradeHistory() {
        try {
            const response = await fetch('/api/trade_history');
            const data = await response.json();
            
            const message = `
                Trade History Summary:
                
                Total Trades: ${data.total}
                Profitable Trades: ${data.profitable}
                Win Rate: ${data.win_rate ? data.win_rate.toFixed(1) : 0}%
                
                Check console for detailed history.
            `;
            
            alert(message);
            console.log('Detailed Trade History:', data);
        } catch (error) {
            console.error('Error fetching trade history:', error);
            this.showNotification('Failed to fetch trade history', 'error');
        }
    }

    async viewMLData() {
        try {
            const response = await fetch('/api/view_ml_data');
            const data = await response.json();
            
            if (data.error) {
                this.showNotification(data.error, 'error');
                return;
            }
            
            const message = `
                ML Training Data (${TRAINING_FILE}):
                
                Total Lines: ${data.total_lines}
                Training Samples: ${data.ml_status?.samples || 0}
                ML Model Ready: ${data.ml_status?.ready ? 'Yes 🤖' : 'No'}
                
                Check console for data preview.
            `;
            
            alert(message);
            console.log('ML Data Preview:', data.preview);
        } catch (error) {
            console.error('Error viewing ML data:', error);
            this.showNotification('Failed to fetch ML data', 'error');
        }
    }

    async viewAdvancedMetrics() {
        try {
            const response = await fetch('/api/advanced_metrics');
            const data = await response.json();
            
            const message = `
                Advanced Trading Metrics:
                
                Total Pips: ${data.total_pips?.toFixed(1) || 0}
                Best Trade: ${data.best_trade_pips?.toFixed(1) || 0} pips
                Worst Trade: ${data.worst_trade_pips?.toFixed(1) || 0} pips
                Avg Win: ${data.avg_win_pips?.toFixed(1) || 0} pips
                Avg Loss: ${data.avg_loss_pips?.toFixed(1) || 0} pips
                Profit Factor: ${data.profit_factor?.toFixed(2) || 0}
                Expectancy: ${data.expectancy?.toFixed(2) || 0} pips
            `;
            
            alert(message);
            console.log('Advanced Metrics:', data);
        } catch (error) {
            console.error('Error fetching advanced metrics:', error);
            this.showNotification('Failed to fetch advanced metrics', 'error');
        }
    }

    async forceMLTraining() {
        try {
            const response = await fetch('/api/force_ml_training', {
                method: 'POST'
            });
            const data = await response.json();
            
            if (data.success) {
                this.showNotification(data.message, 'success');
                await this.fetchMLStatus();
            } else {
                this.showNotification(data.message, 'error');
            }
        } catch (error) {
            console.error('Error forcing ML training:', error);
            this.showNotification('Failed to force ML training', 'error');
        }
    }

    async resetTrading() {
        if (!confirm('⚠️ Are you sure you want to reset the trading system?\n\nThis will:\n• Clear all trade history\n• Reset balance to $10,000\n• Clear ML training data\n• Reset all statistics\n\nThis action cannot be undone!')) {
            return;
        }
        
        try {
            const response = await fetch('/api/reset_trading', {
                method: 'POST'
            });
            const data = await response.json();
            
            if (data.success) {
                this.showNotification(data.message, 'success');
                // Refresh all data after reset
                setTimeout(() => {
                    this.refreshAllData();
                }, 1000);
            } else {
                this.showNotification(data.message, 'error');
            }
        } catch (error) {
            console.error('Error resetting trading:', error);
            this.showNotification('Failed to reset trading', 'error');
        }
    }

    toggleChartType() {
        this.state.chartType = this.state.chartType === 'price' ? 'ml' : 'price';
        this.showNotification(`Switched to ${this.state.chartType.toUpperCase()} view`, 'success');
    }

    handleTradeUpdate(data) {
        if (data.status === 'executed') {
            this.showNotification(`🔔 ${data.trade.action} executed at ${data.trade.entry_price}`, 'success');
        } else if (data.status === 'closed') {
            this.showNotification(`💰 Trade closed: ${data.trade.exit_reason}`, 'success');
        }
    }

    handleMLUpdate(data) {
        if (data.type === 'training_complete') {
            this.showNotification(`🤖 ML model updated with ${data.samples} samples`, 'success');
        } else if (data.type === 'correction_applied') {
            this.showNotification(`🔧 ML auto-correction applied: ${data.message}`, 'warning');
        }
    }

    updateMLStatus(data) {
        // Update ML-specific UI elements
        const mlStatusElement = document.getElementById('ml-status');
        mlStatusElement.textContent = data.ml_model_ready ? 'ACTIVE 🤖' : 'LEARNING...';
        mlStatusElement.style.color = data.ml_model_ready ? 'var(--primary-color)' : 'var(--warning-color)';
    }
}

// Initialize the trading system when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.tradingSystem = new TradingSystem();
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'r') {
            e.preventDefault();
            window.tradingSystem.refreshAllData();
        } else if (e.ctrlKey && e.key === 'h') {
            e.preventDefault();
            window.tradingSystem.viewTradeHistory();
        } else if (e.ctrlKey && e.key === 'm') {
            e.preventDefault();
            window.tradingSystem.viewMLData();
        } else if (e.key === 'Escape') {
            const notification = document.getElementById('notification');
            notification.style.display = 'none';
        }
    });
});

// Global functions for HTML onclick handlers
function refreshAllData() {
    if (window.tradingSystem) {
        window.tradingSystem.refreshAllData();
    }
}

function viewTradeHistory() {
    if (window.tradingSystem) {
        window.tradingSystem.viewTradeHistory();
    }
}

function viewMLData() {
    if (window.tradingSystem) {
        window.tradingSystem.viewMLData();
    }
}

function viewAdvancedMetrics() {
    if (window.tradingSystem) {
        window.tradingSystem.viewAdvancedMetrics();
    }
}

function forceMLTraining() {
    if (window.tradingSystem) {
        window.tradingSystem.forceMLTraining();
    }
}

function resetTrading() {
    if (window.tradingSystem) {
        window.tradingSystem.resetTrading();
    }
}

function toggleChartType() {
    if (window.tradingSystem) {
        window.tradingSystem.toggleChartType();
    }
}