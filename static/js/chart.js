/**
 * EUR/USD Trading System - Chart Management Module (Google Sheets Edition)
 * Fixed for 2-Minute Cycles with Google Sheets Storage Indicators
 */

let currentChart = null;
let priceChartData = [];
let chartUpdateInterval = null;
let isFullscreen = false;
const CHART_POINTS = 120; // ⭐ FIXED: 120 points for 2 minutes

// Initialize chart system
function initializeChartSystem() {
    console.log('Chart system initialized (2-Minute Cycles, Google Sheets)');
    
    // Create initial empty chart
    createInitialChart();
    
    // Set up chart controls
    setupChartControls();
    
    // Start chart updates
    startChartUpdates();
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
    
    console.log('Initial chart created (120-point view, Google Sheets)');
}

// Get chart layout configuration
function getChartLayout() {
    return {
        title: {
            text: 'EUR/USD Live Price Chart (2-Minute View) - Google Sheets Storage',
            font: { size: 16, color: '#ffffff' },
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
        const layout = getChartLayout();
        Plotly.react('chart', traces, layout);
        
        // Add trade markers if trade exists
        if (tradeInfo) {
            addTradeToChart(tradeInfo, timePoints, displayPrices);
        }
        
        // Update chart status with storage indicator
        updateChartStatus('Live');
        
        // Store data
        priceChartData = displayPrices;
        
    } catch (error) {
        console.error('Chart update error:', error);
        updateChartStatus('Error', 'danger');
    }
}

// Update chart status with storage indicator
function updateChartStatus(status, type = 'success') {
    const chartStatus = document.getElementById('chartStatus');
    const sheetsStatus = document.getElementById('sheetsStatus');
    
    if (chartStatus) {
        chartStatus.textContent = status;
        chartStatus.className = `badge bg-${type}`;
    }
    
    // Update sheets status if exists
    if (sheetsStatus) {
        // Try to get storage status from dashboard
        const storageStatus = document.getElementById('storageStatus');
        if (storageStatus) {
            const storageClass = storageStatus.className;
            if (storageClass.includes('bg-success')) {
                sheetsStatus.className = 'badge bg-success';
                sheetsStatus.textContent = 'Sheets: Connected';
            } else if (storageClass.includes('bg-danger')) {
                sheetsStatus.className = 'badge bg-danger';
                sheetsStatus.textContent = 'Sheets: Error';
            } else {
                sheetsStatus.className = 'badge bg-warning';
                sheetsStatus.textContent = 'Sheets: Checking...';
            }
        }
    }
}

// Update chart from trading state data
function updateChartFromState(tradingState) {
    if (!tradingState) {
        updateChartStatus('No Data', 'warning');
        return;
    }
    
    // Update chart status based on storage
    const storageStatus = tradingState.google_sheets_status || 'UNKNOWN';
    if (storageStatus.includes('CONNECTED')) {
        updateChartStatus('Live', 'success');
    } else if (storageStatus.includes('ERROR')) {
        updateChartStatus('Live (Storage Issue)', 'warning');
    } else {
        updateChartStatus('Live', 'info');
    }
    
    // Update title with cycle info
    updateChartTitle(tradingState);
    
    if (!tradingState.chart_data) {
        // Create simple chart from price data
        if (tradingState.current_price) {
            const price = tradingState.current_price;
            const priceData = generatePriceSeries(price, CHART_POINTS);
            updateChart(priceData, {}, tradingState.current_trade);
        }
        return;
    }
    
    try {
        const chartData = JSON.parse(tradingState.chart_data);
        
        // Enhance chart data with storage info
        if (chartData.layout && chartData.layout.title) {
            chartData.layout.title.text = `EUR/USD Live Chart - Cycle #${tradingState.cycle_count || 0}`;
            
            // Add storage indicator to subtitle
            const subtitleText = storageStatus.includes('CONNECTED') 
                ? 'Data saved to Google Sheets ✓' 
                : 'Data storage: Checking...';
            
            if (!chartData.layout.annotations) {
                chartData.layout.annotations = [];
            }
            
            // Remove existing storage annotation if exists
            chartData.layout.annotations = chartData.layout.annotations.filter(
                ann => !ann.text || !ann.text.includes('Google Sheets')
            );
            
            // Add storage annotation
            chartData.layout.annotations.push({
                x: 0.02,
                y: 1.05,
                xref: 'paper',
                yref: 'paper',
                text: subtitleText,
                showarrow: false,
                font: {
                    size: 12,
                    color: storageStatus.includes('CONNECTED') ? '#00ff88' : '#ffaa00'
                },
                bgcolor: 'rgba(0, 0, 0, 0.5)',
                borderpad: 4,
                borderwidth: 1,
                bordercolor: storageStatus.includes('CONNECTED') ? '#00ff88' : '#ffaa00'
            });
        }
        
        Plotly.react('chart', chartData.data, chartData.layout);
        
        // Store data
        if (chartData.data && chartData.data[0] && chartData.data[0].y) {
            priceChartData = chartData.data[0].y;
        }
        
    } catch (error) {
        console.log('Using dynamic chart update:', error);
        updateChartDynamic(tradingState);
    }
}

// Update chart title with cycle and storage info
function updateChartTitle(tradingState) {
    if (!tradingState) return;
    
    const nextCycle = tradingState.next_cycle_in || 120;
    const cycleCount = tradingState.cycle_count || 0;
    const storageStatus = tradingState.google_sheets_status || 'UNKNOWN';
    
    let storageIcon = '🔄';
    if (storageStatus.includes('CONNECTED')) storageIcon = '✅';
    else if (storageStatus.includes('ERROR')) storageIcon = '⚠️';
    
    const layoutUpdate = {
        title: {
            text: `EUR/USD 2-Minute Trading - Cycle #${cycleCount} - Next: ${nextCycle}s ${storageIcon}`
        }
    };
    
    Plotly.relayout('chart', layoutUpdate);
}

// Dynamic chart update (fallback)
function updateChartDynamic(tradingState) {
    if (!tradingState) return;
    
    // Generate price series
    const basePrice = tradingState.current_price || 1.0850;
    const priceSeries = generatePriceSeries(basePrice, CHART_POINTS);
    
    // Update chart
    updateChart(priceSeries, {}, tradingState.current_trade);
    
    // Update title
    updateChartTitle(tradingState);
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
        const layoutUpdates = {
            shapes: [],
            annotations: []
        };
        
        const lastIndex = timePoints.length - 1;
        const currentPrice = priceData[priceData.length - 1];
        const entryIdx = Math.max(0, lastIndex - 30);
        
        // Determine trade color based on action
        const tradeColor = trade.action === 'BUY' ? '#00ff00' : '#ff0000';
        const tradeArrow = trade.action === 'BUY' ? '▲' : '▼';
        
        // Add entry point annotation
        layoutUpdates.annotations.push({
            x: entryIdx,
            y: trade.entry_price,
            xref: 'x',
            yref: 'y',
            text: `${tradeArrow} Entry: ${trade.entry_price.toFixed(5)}`,
            showarrow: true,
            arrowhead: 2,
            arrowsize: 1,
            arrowwidth: 2,
            arrowcolor: tradeColor,
            ax: 0,
            ay: trade.action === 'BUY' ? -40 : 40,
            bgcolor: 'rgba(0, 0, 0, 0.8)',
            bordercolor: tradeColor,
            borderwidth: 1,
            borderpad: 4,
            font: { color: '#ffffff', size: 12 }
        });
        
        // Add TP line
        if (trade.optimal_tp) {
            layoutUpdates.shapes.push({
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
            layoutUpdates.annotations.push({
                x: lastIndex - 10,
                y: trade.optimal_tp,
                xref: 'x',
                yref: 'y',
                text: `TP: ${trade.optimal_tp.toFixed(5)}`,
                showarrow: false,
                bgcolor: 'rgba(0, 255, 0, 0.3)',
                bordercolor: '#00ff00',
                borderwidth: 1,
                borderpad: 3,
                font: { color: '#ffffff', size: 10 }
            });
        }
        
        // Add SL line
        if (trade.optimal_sl) {
            layoutUpdates.shapes.push({
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
            layoutUpdates.annotations.push({
                x: lastIndex - 10,
                y: trade.optimal_sl,
                xref: 'x',
                yref: 'y',
                text: `SL: ${trade.optimal_sl.toFixed(5)}`,
                showarrow: false,
                bgcolor: 'rgba(255, 0, 0, 0.3)',
                bordercolor: '#ff0000',
                borderwidth: 1,
                borderpad: 3,
                font: { color: '#ffffff', size: 10 }
            });
        }
        
        // Add current P/L annotation if available
        if (trade.profit_pips !== undefined) {
            const profitColor = trade.profit_pips >= 0 ? '#00ff00' : '#ff0000';
            const profitSign = trade.profit_pips >= 0 ? '+' : '';
            
            layoutUpdates.annotations.push({
                x: lastIndex - 20,
                y: currentPrice,
                xref: 'x',
                yref: 'y',
                text: `P/L: ${profitSign}${trade.profit_pips.toFixed(1)} pips`,
                showarrow: true,
                arrowhead: 1,
                arrowsize: 0.5,
                arrowwidth: 1,
                arrowcolor: profitColor,
                ax: 0,
                ay: -20,
                bgcolor: 'rgba(0, 0, 0, 0.8)',
                bordercolor: profitColor,
                borderwidth: 1,
                borderpad: 3,
                font: { color: profitColor, size: 11 }
            });
        }
        
        // Add storage indicator for the trade
        layoutUpdates.annotations.push({
            x: lastIndex - 5,
            y: trade.entry_price,
            xref: 'x',
            yref: 'y',
            text: '💾', // Storage emoji
            showarrow: false,
            font: { size: 14 },
            xanchor: 'center',
            yanchor: 'middle'
        });
        
        // Update chart with annotations
        Plotly.relayout('chart', layoutUpdates);
        
    } catch (error) {
        console.error('Error adding trade to chart:', error);
    }
}

// Setup chart controls
function setupChartControls() {
    // Toggle fullscreen
    const toggleChartBtn = document.getElementById('toggleChart');
    if (toggleChartBtn) {
        toggleChartBtn.addEventListener('click', toggleChartFullscreen);
    }
    
    // Add storage indicator click handler
    const sheetsStatus = document.getElementById('sheetsStatus');
    if (sheetsStatus) {
        sheetsStatus.addEventListener('click', function() {
            if (typeof checkGoogleSheetsConnection === 'function') {
                checkGoogleSheetsConnection();
            }
        });
        sheetsStatus.style.cursor = 'pointer';
        sheetsStatus.title = 'Click to check Google Sheets connection';
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
        isFullscreen = true;
        
        // Update chart size for fullscreen
        setTimeout(() => {
            Plotly.Plots.resize(chartDiv);
            updateFullscreenLayout();
        }, 100);
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
        isFullscreen = false;
        
        // Update chart size back to normal
        setTimeout(() => {
            Plotly.Plots.resize(chartDiv);
            updateNormalLayout();
        }, 100);
    }
}

// Update layout for fullscreen mode
function updateFullscreenLayout() {
    const update = {
        margin: { l: 80, r: 50, t: 80, b: 60 },
        title: { font: { size: 24 } },
        xaxis: { title: { font: { size: 16 } } },
        yaxis: { title: { font: { size: 16 } } },
        legend: { font: { size: 14 } }
    };
    
    Plotly.relayout('chart', update);
}

// Update layout for normal mode
function updateNormalLayout() {
    const update = {
        margin: { l: 60, r: 30, t: 60, b: 40 },
        title: { font: { size: 16 } },
        xaxis: { title: { font: { size: 12 } } },
        yaxis: { title: { font: { size: 12 } } },
        legend: { font: { size: 12 } }
    };
    
    Plotly.relayout('chart', update);
}

// Start automatic chart updates
function startChartUpdates() {
    if (chartUpdateInterval) {
        clearInterval(chartUpdateInterval);
    }
    
    chartUpdateInterval = setInterval(() => {
        // Ensure chart stays responsive
        if (currentChart) {
            Plotly.Plots.resize('chart');
        }
        
        // Periodically update storage status on chart
        updateStorageIndicatorOnChart();
        
    }, 15000); // Check every 15 seconds
    
    console.log('Chart updates started with storage monitoring');
}

// Update storage indicator on chart
function updateStorageIndicatorOnChart() {
    // Check if we can get storage status
    const storageStatus = document.getElementById('storageStatus');
    const sheetsStatus = document.getElementById('sheetsStatus');
    
    if (!storageStatus || !sheetsStatus) return;
    
    const storageClass = storageStatus.className;
    let statusText = 'Checking...';
    let statusClass = 'bg-warning';
    
    if (storageClass.includes('bg-success')) {
        statusText = 'Sheets: Connected';
        statusClass = 'bg-success';
    } else if (storageClass.includes('bg-danger')) {
        statusText = 'Sheets: Error';
        statusClass = 'bg-danger';
    }
    
    sheetsStatus.className = `badge ${statusClass}`;
    sheetsStatus.textContent = statusText;
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
        // Resize chart for fullscreen
        setTimeout(() => {
            Plotly.Plots.resize(chartDiv);
            updateFullscreenLayout();
        }, 100);
    } else {
        toggleBtn.innerHTML = '<i class="bi bi-arrows-angle-expand"></i>';
        // Resize chart back
        setTimeout(() => {
            Plotly.Plots.resize(chartDiv);
            updateNormalLayout();
        }, 100);
    }
}

// Add price history visualization
function updatePriceHistory(priceHistory) {
    if (!priceHistory || !Array.isArray(priceHistory) || priceHistory.length === 0) {
        return;
    }
    
    // This function can be extended to show price history on a separate chart
    // For now, we'll just update the main chart with recent prices
    if (priceHistory.length > 0) {
        const recentPrices = priceHistory.slice(-10).map(p => p.price);
        if (recentPrices.length > 0) {
            // Update the last part of the chart with real prices
            const startIdx = Math.max(0, CHART_POINTS - recentPrices.length);
            
            const updates = {};
            for (let i = 0; i < recentPrices.length; i++) {
                const chartIdx = startIdx + i;
                if (chartIdx < CHART_POINTS) {
                    updates[`y[${chartIdx}]`] = recentPrices[i];
                }
            }
            
            if (Object.keys(updates).length > 0) {
                Plotly.restyle('chart', updates);
            }
        }
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
window.updatePriceHistory = updatePriceHistory;

console.log('Charts.js loaded successfully (2-Minute Cycle, Google Sheets Edition)');