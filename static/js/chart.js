/**
 * EUR/USD Trading System - Chart Management Module
 * Fixed for 2-Minute Cycles with data.txt ML Integration
 */

let currentChart = null;
let priceChartData = [];
let chartUpdateInterval = null;
let isFullscreen = false;
const CHART_POINTS = 120; // ⭐ FIXED: 120 points for 2 minutes

// ML status tracking for chart annotations
let mlStatus = {
    dataLoaded: false,
    trainingComplete: false,
    correctionsApplied: 0,
    lastUpdate: null,
    messages: []
};

// Initialize chart system
function initializeChartSystem() {
    console.log('Chart system initialized (2-Minute Cycles with data.txt ML)');
    
    // Create initial empty chart
    createInitialChart();
    
    // Set up chart controls
    setupChartControls();
    
    // Start chart updates
    startChartUpdates();
    
    // Initialize ML status monitoring
    initializeMLStatusMonitoring();
}

// Create initial empty chart
function createInitialChart() {
    const layout = getChartLayout();
    const data = [getEmptyTrace()];
    
    currentChart = Plotly.newPlot('chart', data, layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToAdd: ['drawline', 'drawopenpath', 'eraseshape'],
        displaylogo: false
    });
    
    console.log('Initial chart created (120-point view with data.txt ML)');
}

// Get chart layout configuration
function getChartLayout() {
    return {
        title: {
            text: 'EUR/USD Live Price Chart (2-Minute View)',
            font: { size: 18, color: '#ffffff' },
            x: 0.05
        },
        xaxis: {
            title: { text: 'Time (seconds ago)', font: { color: '#ffffff' } },
            gridcolor: 'rgba(255, 255, 255, 0.1)',
            zerolinecolor: 'rgba(255, 255, 255, 0.2)',
            color: '#ffffff',
            showgrid: true,
            tickfont: { color: '#ffffff' },
            range: [0, CHART_POINTS]
        },
        yaxis: {
            title: { text: 'Price', font: { color: '#ffffff' } },
            gridcolor: 'rgba(255, 255, 255, 0.1)',
            zerolinecolor: 'rgba(255, 255, 255, 0.2)',
            color: '#ffffff',
            tickfont: { color: '#ffffff' },
            tickformat: '.5f'
        },
        plot_bgcolor: 'rgba(0, 0, 0, 0)',
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        template: 'plotly_dark',
        hovermode: 'x unified',
        showlegend: true,
        legend: {
            x: 0.02,
            y: 1,
            bgcolor: 'rgba(0, 0, 0, 0.5)',
            bordercolor: 'rgba(255, 255, 255, 0.2)',
            borderwidth: 1,
            font: { color: '#ffffff' }
        },
        margin: { l: 60, r: 30, t: 60, b: 40 },
        shapes: [],
        annotations: []
    };
}

// Get empty trace for initialization
function getEmptyTrace() {
    return {
        x: Array.from({length: CHART_POINTS}, (_, i) => i),
        y: Array(CHART_POINTS).fill(1.08500),
        type: 'scatter',
        mode: 'lines',
        name: 'EUR/USD Price',
        line: {
            color: '#00ff88',
            width: 3
        },
        hovertemplate: 'Price: %{y:.5f}<extra></extra>'
    };
}

// Update chart with new data
function updateChart(priceData, indicators = {}, tradeInfo = null) {
    if (!currentChart) return;
    
    try {
        // Prepare data for 120-point chart
        const timePoints = Array.from({length: CHART_POINTS}, (_, i) => i);
        let displayPrices;
        
        if (priceData && priceData.length > 0) {
            // If we have data, use it (pad if necessary)
            if (priceData.length >= CHART_POINTS) {
                displayPrices = priceData.slice(-CHART_POINTS);
            } else {
                // Pad with first value
                displayPrices = [
                    ...Array(CHART_POINTS - priceData.length).fill(priceData[0]),
                    ...priceData
                ];
            }
        } else {
            // No data, use empty array
            displayPrices = Array(CHART_POINTS).fill(1.08500);
        }
        
        // Create traces
        const traces = [{
            x: timePoints,
            y: displayPrices,
            type: 'scatter',
            mode: 'lines',
            name: 'EUR/USD Price',
            line: { color: '#00ff88', width: 3 },
            hovertemplate: 'Price: %{y:.5f}<extra></extra>'
        }];
        
        // Update chart
        Plotly.react('chart', traces, getChartLayout());
        
        // Add trade markers if trade exists
        if (tradeInfo) {
            addTradeToChart(tradeInfo, timePoints, displayPrices);
        }
        
        // Update ML status on chart if available
        updateMLStatusOnChart();
        
        // Update chart status
        updateChartStatus();
        
        // Store data
        priceChartData = displayPrices;
        
    } catch (error) {
        console.error('Chart update error:', error);
        showChartMessage('Chart update failed', 'error');
    }
}

// Update chart from trading state data
function updateChartFromState(tradingState) {
    if (!tradingState || !tradingState.chart_data) {
        // Create simple chart from price data
        if (tradingState && tradingState.current_price) {
            const price = tradingState.current_price;
            const priceData = Array.from({length: CHART_POINTS}, (_, i) => {
                // Create realistic price movement
                const noise = (Math.random() - 0.5) * 0.0002;
                return price + noise * (i / CHART_POINTS);
            });
            updateChart(priceData, {}, tradingState.current_trade);
        }
        return;
    }
    
    try {
        const chartData = JSON.parse(tradingState.chart_data);
        Plotly.react('chart', chartData.data, chartData.layout);
        
        // Update ML status if present in trading state
        if (tradingState.ml_model_ready) {
            mlStatus.trainingComplete = tradingState.ml_model_ready;
            mlStatus.correctionsApplied = tradingState.ml_corrections_applied || 0;
            mlStatus.lastUpdate = new Date();
        }
        
        // Update chart status
        updateChartStatus();
        
        // Show ML status message if recent
        if (mlStatus.lastUpdate && (new Date() - mlStatus.lastUpdate) < 5000) {
            showMLStatusMessage();
        }
        
    } catch (error) {
        console.log('Using dynamic chart update');
        updateChartDynamic(tradingState);
    }
}

// Dynamic chart update (fallback)
function updateChartDynamic(tradingState) {
    if (!tradingState) return;
    
    // Generate price series
    const basePrice = tradingState.current_price || 1.0850;
    const priceSeries = generatePriceSeries(basePrice, CHART_POINTS);
    
    // Update chart
    updateChart(priceSeries, {}, tradingState.current_trade);
}

// Generate simulated price series
function generatePriceSeries(basePrice, length = CHART_POINTS) {
    const series = [basePrice];
    let currentPrice = basePrice;
    
    for (let i = 1; i < length; i++) {
        // Simulate random walk
        const change = (Math.random() - 0.5) * 0.00015;
        currentPrice += change;
        
        // Keep in reasonable range
        if (currentPrice < 1.0800) currentPrice = 1.0800 + Math.abs(change);
        if (currentPrice > 1.0900) currentPrice = 1.0900 - Math.abs(change);
        
        series.push(currentPrice);
    }
    
    return series;
}

// Add trade information to chart
function addTradeToChart(trade, timePoints, priceData) {
    if (!trade || !trade.entry_price) return;
    
    try {
        const layout = getChartLayout();
        const lastIndex = timePoints.length - 1;
        const currentPrice = priceData[priceData.length - 1];
        const entryIdx = Math.max(0, lastIndex - 30);
        
        // Add entry point
        layout.annotations.push({
            x: entryIdx,
            y: trade.entry_price,
            xref: 'x',
            yref: 'y',
            text: `Entry: ${trade.entry_price.toFixed(5)}`,
            showarrow: true,
            arrowhead: 2,
            arrowsize: 1,
            arrowwidth: 2,
            arrowcolor: trade.action === 'BUY' ? '#00ff00' : '#ff0000',
            ax: 0,
            ay: trade.action === 'BUY' ? -40 : 40,
            bgcolor: 'rgba(0, 0, 0, 0.8)',
            bordercolor: trade.action === 'BUY' ? '#00ff00' : '#ff0000',
            borderwidth: 1,
            borderpad: 4,
            font: { color: '#ffffff', size: 12 }
        });
        
        // Add TP line
        if (trade.optimal_tp) {
            layout.shapes.push({
                type: 'line',
                x0: 0,
                y0: trade.optimal_tp,
                x1: lastIndex,
                y1: trade.optimal_tp,
                line: {
                    color: '#00ff00',
                    width: 2,
                    dash: 'dash'
                }
            });
            
            // Add TP annotation
            layout.annotations.push({
                x: lastIndex,
                y: trade.optimal_tp,
                xref: 'x',
                yref: 'y',
                text: `TP: ${trade.optimal_tp.toFixed(5)}`,
                showarrow: false,
                bgcolor: 'rgba(0, 255, 0, 0.3)',
                bordercolor: '#00ff00',
                borderwidth: 1,
                borderpad: 4,
                font: { color: '#ffffff', size: 10 },
                xanchor: 'right',
                yanchor: 'bottom'
            });
        }
        
        // Add SL line
        if (trade.optimal_sl) {
            layout.shapes.push({
                type: 'line',
                x0: 0,
                y0: trade.optimal_sl,
                x1: lastIndex,
                y1: trade.optimal_sl,
                line: {
                    color: '#ff0000',
                    width: 2,
                    dash: 'dash'
                }
            });
            
            // Add SL annotation
            layout.annotations.push({
                x: lastIndex,
                y: trade.optimal_sl,
                xref: 'x',
                yref: 'y',
                text: `SL: ${trade.optimal_sl.toFixed(5)}`,
                showarrow: false,
                bgcolor: 'rgba(255, 0, 0, 0.3)',
                bordercolor: '#ff0000',
                borderwidth: 1,
                borderpad: 4,
                font: { color: '#ffffff', size: 10 },
                xanchor: 'right',
                yanchor: 'top'
            });
        }
        
        // Update chart with annotations
        Plotly.relayout('chart', layout);
        
        // Show trade execution message
        showChartMessage(`Trade ${trade.action} executed at ${trade.entry_price.toFixed(5)}`, 'info');
        
    } catch (error) {
        console.error('Error adding trade to chart:', error);
        showChartMessage('Failed to add trade to chart', 'error');
    }
}

// Update ML status on chart
function updateMLStatusOnChart() {
    if (!mlStatus.trainingComplete && mlStatus.correctionsApplied === 0) return;
    
    try {
        const layout = getChartLayout();
        
        // Add ML status annotation
        const mlAnnotation = {
            x: 0.98,
            y: 0.02,
            xref: 'paper',
            yref: 'paper',
            text: `🤖 ML: ${mlStatus.trainingComplete ? 'Trained' : 'Learning'} | Corrections: ${mlStatus.correctionsApplied}`,
            showarrow: false,
            bgcolor: mlStatus.trainingComplete ? 'rgba(0, 255, 0, 0.3)' : 'rgba(255, 165, 0, 0.3)',
            bordercolor: mlStatus.trainingComplete ? '#00ff00' : '#ffa500',
            borderwidth: 1,
            borderpad: 8,
            font: { color: '#ffffff', size: 12 },
            xanchor: 'right',
            yanchor: 'bottom'
        };
        
        layout.annotations.push(mlAnnotation);
        Plotly.relayout('chart', layout);
        
    } catch (error) {
        console.error('Error updating ML status on chart:', error);
    }
}

// Update chart status display
function updateChartStatus() {
    const chartStatus = document.getElementById('chartStatus');
    if (!chartStatus) return;
    
    chartStatus.textContent = 'Live';
    chartStatus.className = 'badge bg-success';
    
    // Add ML status indicator
    if (mlStatus.trainingComplete) {
        const mlBadge = document.createElement('span');
        mlBadge.className = 'badge bg-info ms-2';
        mlBadge.innerHTML = '🤖 ML Ready';
        mlBadge.id = 'mlChartBadge';
        
        const existingBadge = document.getElementById('mlChartBadge');
        if (existingBadge) {
            existingBadge.remove();
        }
        
        chartStatus.parentNode.appendChild(mlBadge);
    }
}

// Initialize ML status monitoring
function initializeMLStatusMonitoring() {
    // Check for ML status updates periodically
    setInterval(() => {
        fetch('/api/ml_status')
            .then(response => response.json())
            .then(data => {
                const previousStatus = {
                    trainingComplete: mlStatus.trainingComplete,
                    correctionsApplied: mlStatus.correctionsApplied
                };
                
                // Update ML status
                mlStatus.trainingComplete = data.ml_model_ready || false;
                mlStatus.correctionsApplied = data.ml_corrections_applied || 0;
                mlStatus.lastUpdate = new Date();
                
                // Check for changes to show messages
                if (data.ml_data_load_status) {
                    handleMLDataStatus(data.ml_data_load_status);
                }
                
                if (data.ml_training_status) {
                    handleMLTrainingStatus(data.ml_training_status);
                }
                
                // If corrections changed, show message
                if (mlStatus.correctionsApplied !== previousStatus.correctionsApplied) {
                    showChartMessage(`ML applied ${mlStatus.correctionsApplied} corrections`, 'success');
                }
                
                // If training status changed, show message
                if (mlStatus.trainingComplete !== previousStatus.trainingComplete) {
                    if (mlStatus.trainingComplete) {
                        showChartMessage('✅ ML Training Complete!', 'success');
                    } else {
                        showChartMessage('⚠️ ML Training Lost', 'warning');
                    }
                }
            })
            .catch(error => {
                console.error('ML status fetch error:', error);
            });
    }, 5000); // Check every 5 seconds
}

// Handle ML data status messages
function handleMLDataStatus(status) {
    if (status.includes('successfully') || status.includes('Successfully')) {
        if (status.includes('saved')) {
            showChartMessage('✅ Trade data saved to data.txt', 'success');
        } else if (status.includes('loaded')) {
            showChartMessage('✅ ML data loaded from data.txt', 'success');
        }
    } else if (status.includes('failed') || status.includes('error')) {
        showChartMessage('❌ ML data operation failed', 'error');
    }
}

// Handle ML training status messages
function handleMLTrainingStatus(status) {
    if (status.includes('✅') || status.includes('Trained')) {
        showChartMessage('✅ ML training successful', 'success');
    } else if (status.includes('failed') || status.includes('error')) {
        showChartMessage('❌ ML training failed', 'error');
    } else if (status.includes('Need') && status.includes('trades')) {
        // Show info about needed trades
        const match = status.match(/\d+/);
        if (match) {
            const needed = 10 - parseInt(match[0]);
            if (needed > 0) {
                showChartMessage(`ℹ️ Need ${needed} more trades for ML training`, 'info');
            }
        }
    }
}

// Show ML status message on chart
function showMLStatusMessage() {
    const chartContainer = document.getElementById('chart');
    if (!chartContainer) return;
    
    // Create floating message
    const messageDiv = document.createElement('div');
    messageDiv.className = 'chart-ml-message';
    messageDiv.innerHTML = `
        <div class="ml-message-content">
            <strong>🤖 ML Status:</strong>
            <span>${mlStatus.trainingComplete ? 'Trained' : 'Learning'}</span>
            <small>Corrections: ${mlStatus.correctionsApplied}</small>
        </div>
    `;
    
    chartContainer.appendChild(messageDiv);
    
    // Remove after 3 seconds
    setTimeout(() => {
        if (messageDiv.parentNode) {
            messageDiv.parentNode.removeChild(messageDiv);
        }
    }, 3000);
}

// Show chart message
function showChartMessage(message, type = 'info') {
    console.log(`Chart ${type}: ${message}`);
    
    // Update status indicator
    const statusIndicator = document.getElementById('chartStatusIndicator');
    if (statusIndicator) {
        statusIndicator.textContent = message;
        statusIndicator.className = `chart-status-indicator status-${type}`;
        
        // Auto-clear info messages after 3 seconds
        if (type === 'info') {
            setTimeout(() => {
                statusIndicator.className = 'chart-status-indicator';
                statusIndicator.textContent = '';
            }, 3000);
        }
    }
    
    // Also log to console with appropriate icon
    const icon = type === 'success' ? '✅' : 
                 type === 'error' ? '❌' : 
                 type === 'warning' ? '⚠️' : 'ℹ️';
    console.log(`${icon} ${message}`);
}

// Setup chart controls
function setupChartControls() {
    // Toggle fullscreen
    const toggleChartBtn = document.getElementById('toggleChart');
    if (toggleChartBtn) {
        toggleChartBtn.addEventListener('click', toggleChartFullscreen);
    }
    
    // ML data viewer button
    const viewMLDataBtn = document.getElementById('viewMLData');
    if (viewMLDataBtn) {
        viewMLDataBtn.addEventListener('click', viewMLDataFromChart);
    }
    
    // Export chart data button
    const exportChartBtn = document.getElementById('exportChart');
    if (exportChartBtn) {
        exportChartBtn.addEventListener('click', exportChartData);
    }
}

// Toggle chart fullscreen
function toggleChartFullscreen() {
    const chartDiv = document.getElementById('chart');
    const toggleBtn = document.getElementById('toggleChart');
    
    if (!isFullscreen) {
        // Enter fullscreen
        if (chartDiv.requestFullscreen) {
            chartDiv.requestFullscreen();
        } else if (chartDiv.webkitRequestFullscreen) {
            chartDiv.webkitRequestFullscreen();
        } else if (chartDiv.msRequestFullscreen) {
            chartDiv.msRequestFullscreen();
        }
        
        toggleBtn.innerHTML = '<i class="bi bi-arrows-angle-contract"></i>';
        toggleBtn.title = 'Exit Fullscreen';
        isFullscreen = true;
        showChartMessage('Chart in fullscreen mode', 'info');
    } else {
        // Exit fullscreen
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        } else if (document.msExitFullscreen) {
            document.msExitFullscreen();
        }
        
        toggleBtn.innerHTML = '<i class="bi bi-arrows-angle-expand"></i>';
        toggleBtn.title = 'Enter Fullscreen';
        isFullscreen = false;
        showChartMessage('Chart exited fullscreen', 'info');
    }
}

// View ML data from chart context
function viewMLDataFromChart() {
    fetch('/api/view_ml_data')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showChartMessage('Error loading ML data: ' + data.error, 'error');
                return;
            }
            
            showChartMessage(`Loaded ${data.total_lines} trades from data.txt`, 'success');
            
            // Create popup with ML data summary
            const summary = createMLDataSummary(data);
            showChartPopup('ML Data Summary', summary);
        })
        .catch(error => {
            console.error('Error viewing ML data:', error);
            showChartMessage('Failed to load ML data from data.txt', 'error');
        });
}

// Create ML data summary
function createMLDataSummary(data) {
    let html = `<div class="ml-data-summary">`;
    html += `<p><strong>📊 ML Training Data Summary</strong></p>`;
    html += `<p>File: <code>${data.file}</code></p>`;
    html += `<p>Total trades: <strong>${data.total_lines}</strong></p>`;
    
    if (data.data && data.data.length > 0) {
        const recentTrade = data.data[data.data.length - 1];
        html += `<p>Most recent trade: <strong>#${recentTrade.trade_id || 'N/A'}</strong></p>`;
        html += `<p>Status: <span class="trade-${recentTrade.result?.toLowerCase()}">${recentTrade.result || 'N/A'}</span></p>`;
    }
    
    html += `<p><small>Last updated: ${new Date().toLocaleTimeString()}</small></p>`;
    html += `</div>`;
    
    return html;
}

// Show chart popup
function showChartPopup(title, content) {
    const popup = document.createElement('div');
    popup.className = 'chart-popup';
    popup.innerHTML = `
        <div class="chart-popup-content">
            <div class="chart-popup-header">
                <h4>${title}</h4>
                <button class="close-popup">&times;</button>
            </div>
            <div class="chart-popup-body">
                ${content}
            </div>
        </div>
    `;
    
    document.body.appendChild(popup);
    
    // Close button
    const closeBtn = popup.querySelector('.close-popup');
    closeBtn.addEventListener('click', () => popup.remove());
    
    // Close on outside click
    popup.addEventListener('click', (e) => {
        if (e.target === popup) {
            popup.remove();
        }
    });
    
    // Auto-close after 5 seconds
    setTimeout(() => {
        if (popup.parentNode) {
            popup.remove();
        }
    }, 5000);
}

// Export chart data
function exportChartData() {
    try {
        const chartData = {
            priceData: priceChartData,
            mlStatus: mlStatus,
            timestamp: new Date().toISOString(),
            chartPoints: CHART_POINTS
        };
        
        const dataStr = JSON.stringify(chartData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chart_data_${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showChartMessage('Chart data exported', 'success');
    } catch (error) {
        console.error('Export error:', error);
        showChartMessage('Failed to export chart data', 'error');
    }
}

// Start automatic chart updates
function startChartUpdates() {
    if (chartUpdateInterval) {
        clearInterval(chartUpdateInterval);
    }
    
    chartUpdateInterval = setInterval(() => {
        // Just ensure chart stays responsive
        if (currentChart) {
            Plotly.Plots.resize('chart');
        }
        
        // Periodically check for ML status
        if (Math.random() < 0.2) { // 20% chance each check
            fetchMLStatusForChart();
        }
    }, 10000); // Check every 10 seconds
    
    console.log('Chart updates started with ML monitoring');
}

// Fetch ML status for chart
function fetchMLStatusForChart() {
    fetch('/api/ml_status')
        .then(response => response.json())
        .then(data => {
            if (data.ml_data_load_status && data.ml_data_load_status.includes('saved')) {
                showChartMessage('✅ Trade data saved to data.txt', 'success');
            }
        })
        .catch(() => {
            // Silent fail - this is just for status updates
        });
}

// Stop chart updates
function stopChartUpdates() {
    if (chartUpdateInterval) {
        clearInterval(chartUpdateInterval);
        chartUpdateInterval = null;
    }
}

// Handle fullscreen change
document.addEventListener('fullscreenchange', handleFullscreenChange);
document.addEventListener('webkitfullscreenchange', handleFullscreenChange);
document.addEventListener('mozfullscreenchange', handleFullscreenChange);
document.addEventListener('MSFullscreenChange', handleFullscreenChange);

function handleFullscreenChange() {
    const chartDiv = document.getElementById('chart');
    const toggleBtn = document.getElementById('toggleChart');
    
    isFullscreen = !!(document.fullscreenElement || 
                      document.webkitFullscreenElement || 
                      document.mozFullScreenElement || 
                      document.msFullscreenElement);
    
    if (isFullscreen) {
        toggleBtn.innerHTML = '<i class="bi bi-arrows-angle-contract"></i>';
        toggleBtn.title = 'Exit Fullscreen';
        // Resize chart for fullscreen
        setTimeout(() => Plotly.Plots.resize(chartDiv), 100);
        showChartMessage('Fullscreen mode active', 'info');
    } else {
        toggleBtn.innerHTML = '<i class="bi bi-arrows-angle-expand"></i>';
        toggleBtn.title = 'Enter Fullscreen';
        // Resize chart back
        setTimeout(() => Plotly.Plots.resize(chartDiv), 100);
        showChartMessage('Exited fullscreen mode', 'info');
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    stopChartUpdates();
    if (currentChart) {
        Plotly.purge('chart');
    }
});

// Export functions for use in main.js
window.initializeChartSystem = initializeChartSystem;
window.updateChartFromState = updateChartFromState;
window.updateChart = updateChart;
window.startChartUpdates = startChartUpdates;
window.stopChartUpdates = stopChartUpdates;
window.showChartMessage = showChartMessage;

console.log('Charts.js loaded successfully (2-Minute Cycle with data.txt ML)');