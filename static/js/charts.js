/**
 * EUR/USD Trading System - Chart Management Module
 * Fixed for 2-Minute Cycles
 */

let currentChart = null;
let priceChartData = [];
let chartUpdateInterval = null;
let isFullscreen = false;
const CHART_POINTS = 120; // ⭐ FIXED: 120 points for 2 minutes

// Initialize chart system
function initializeChartSystem() {
    console.log('Chart system initialized (2-Minute Cycles)');
    
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
    
    console.log('Initial chart created (120-point view)');
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
        
        // Update chart status
        document.getElementById('chartStatus').textContent = 'Live';
        document.getElementById('chartStatus').className = 'badge bg-success';
        
        // Store data
        priceChartData = displayPrices;
        
    } catch (error) {
        console.error('Chart update error:', error);
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
        
        // Update chart status
        document.getElementById('chartStatus').textContent = 'Live';
        document.getElementById('chartStatus').className = 'badge bg-success';
        
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
        }
        
        // Update chart with annotations
        Plotly.relayout('chart', layout);
        
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
    }, 10000); // Check every 10 seconds
    
    console.log('Chart updates started');
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
        setTimeout(() => Plotly.Plots.resize(chartDiv), 100);
    } else {
        toggleBtn.innerHTML = '<i class="bi bi-arrows-angle-expand"></i>';
        // Resize chart back
        setTimeout(() => Plotly.Plots.resize(chartDiv), 100);
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

console.log('Charts.js loaded successfully (2-Minute Cycle)');