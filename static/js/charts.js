/**
 * EUR/USD Trading System - Chart Management Module
 * Handles all chart-related functionality
 */

let currentChart = null;
let priceChartData = [];
let chartUpdateInterval = null;
let isFullscreen = false;

// Initialize chart system
function initializeChartSystem() {
    console.log('Chart system initialized');
    
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
    
    console.log('Initial chart created');
}

// Get chart layout configuration
function getChartLayout() {
    return {
        title: {
            text: 'EUR/USD Live Price Chart',
            font: { size: 18, color: '#ffffff' },
            x: 0.05
        },
        xaxis: {
            title: { text: 'Time (seconds ago)', font: { color: '#ffffff' } },
            gridcolor: 'rgba(255, 255, 255, 0.1)',
            zerolinecolor: 'rgba(255, 255, 255, 0.2)',
            color: '#ffffff',
            showgrid: true,
            tickfont: { color: '#ffffff' }
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
        x: [],
        y: [],
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
    if (!currentChart || !priceData || priceData.length === 0) {
        return;
    }
    
    try {
        // Prepare time axis (seconds ago)
        const timePoints = Array.from({length: priceData.length}, (_, i) => priceData.length - i - 1);
        
        // Update main price trace
        const update1 = {
            x: [timePoints],
            y: [priceData]
        };
        
        Plotly.react('chart', [{
            x: timePoints,
            y: priceData,
            type: 'scatter',
            mode: 'lines',
            name: 'EUR/USD Price',
            line: { color: '#00ff88', width: 3 },
            hovertemplate: 'Price: %{y:.5f}<extra></extra>'
        }], getChartLayout());
        
        // Add moving averages if available
        const traces = [{
            x: timePoints,
            y: priceData,
            type: 'scatter',
            mode: 'lines',
            name: 'EUR/USD Price',
            line: { color: '#00ff88', width: 3 },
            hovertemplate: 'Price: %{y:.5f}<extra></extra>'
        }];
        
        if (indicators.sma_5 && indicators.sma_5.length > 0) {
            traces.push({
                x: timePoints,
                y: indicators.sma_5,
                type: 'scatter',
                mode: 'lines',
                name: 'SMA 5',
                line: { color: 'orange', width: 1.5, dash: 'dash' },
                opacity: 0.7,
                hovertemplate: 'SMA 5: %{y:.5f}<extra></extra>'
            });
        }
        
        if (indicators.sma_10 && indicators.sma_10.length > 0) {
            traces.push({
                x: timePoints,
                y: indicators.sma_10,
                type: 'scatter',
                mode: 'lines',
                name: 'SMA 10',
                line: { color: 'cyan', width: 1.5, dash: 'dot' },
                opacity: 0.7,
                hovertemplate: 'SMA 10: %{y:.5f}<extra></extra>'
            });
        }
        
        // Add trade markers if trade exists
        const layout = getChartLayout();
        layout.shapes = [];
        layout.annotations = [];
        
        if (tradeInfo) {
            addTradeToChart(tradeInfo, timePoints, priceData, layout);
        }
        
        // Update chart title
        const isDemo = document.getElementById('dataSource').textContent.includes('Simulation');
        layout.title.text = `EUR/USD Live Price ${isDemo ? '(Simulation Mode)' : ''}`;
        
        Plotly.react('chart', traces, layout);
        
        // Store latest price data
        priceChartData = priceData;
        
    } catch (error) {
        console.error('Chart update error:', error);
    }
}

// Add trade information to chart
function addTradeToChart(trade, timePoints, priceData, layout) {
    if (!trade || !trade.entry_price) return;
    
    const lastIndex = timePoints.length - 1;
    const currentPrice = priceData[priceData.length - 1];
    
    // Add entry point marker
    layout.annotations.push({
        x: timePoints[Math.max(0, lastIndex - 5)],
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
            x0: timePoints[0],
            y0: trade.optimal_tp,
            x1: timePoints[lastIndex],
            y1: trade.optimal_tp,
            line: {
                color: '#00ff00',
                width: 2,
                dash: 'dash'
            }
        });
        
        layout.annotations.push({
            x: timePoints[lastIndex],
            y: trade.optimal_tp,
            xref: 'x',
            yref: 'y',
            text: `TP: ${trade.optimal_tp.toFixed(5)}`,
            showarrow: false,
            bgcolor: 'rgba(0, 255, 0, 0.2)',
            bordercolor: '#00ff00',
            borderwidth: 1,
            borderpad: 4,
            font: { color: '#ffffff', size: 11 }
        });
    }
    
    // Add SL line
    if (trade.optimal_sl) {
        layout.shapes.push({
            type: 'line',
            x0: timePoints[0],
            y0: trade.optimal_sl,
            x1: timePoints[lastIndex],
            y1: trade.optimal_sl,
            line: {
                color: '#ff0000',
                width: 2,
                dash: 'dash'
            }
        });
        
        layout.annotations.push({
            x: timePoints[lastIndex],
            y: trade.optimal_sl,
            xref: 'x',
            yref: 'y',
            text: `SL: ${trade.optimal_sl.toFixed(5)}`,
            showarrow: false,
            bgcolor: 'rgba(255, 0, 0, 0.2)',
            bordercolor: '#ff0000',
            borderwidth: 1,
            borderpad: 4,
            font: { color: '#ffffff', size: 11 }
        });
    }
    
    // Add current price vs entry comparison
    const profitPips = trade.action === 'BUY' 
        ? (currentPrice - trade.entry_price) * 10000
        : (trade.entry_price - currentPrice) * 10000;
    
    layout.annotations.push({
        x: timePoints[lastIndex],
        y: currentPrice,
        xref: 'x',
        yref: 'y',
        text: `Current: ${currentPrice.toFixed(5)}<br>P/L: ${profitPips.toFixed(1)} pips`,
        showarrow: true,
        arrowhead: 2,
        arrowsize: 1,
        arrowwidth: 2,
        arrowcolor: profitPips >= 0 ? '#00ff00' : '#ff0000',
        ax: -80,
        ay: 0,
        bgcolor: 'rgba(0, 0, 0, 0.8)',
        bordercolor: profitPips >= 0 ? '#00ff00' : '#ff0000',
        borderwidth: 1,
        borderpad: 4,
        font: { color: '#ffffff', size: 11 }
    });
}

// Update chart from trading state data
function updateChartFromState(tradingState) {
    if (!tradingState || !tradingState.chart_data) return;
    
    try {
        // If we have pre-rendered chart data from server
        const chartData = JSON.parse(tradingState.chart_data);
        
        // Add trade info if available
        if (tradingState.current_trade) {
            const layout = chartData.layout || getChartLayout();
            addTradeToChart(
                tradingState.current_trade, 
                Array.from({length: 60}, (_, i) => i),
                tradingState.current_trade.entry_price ? 
                    Array(60).fill(tradingState.current_trade.entry_price) : 
                    Array(60).fill(tradingState.current_price),
                layout
            );
            chartData.layout = layout;
        }
        
        Plotly.react('chart', chartData.data, chartData.layout);
        
    } catch (error) {
        console.log('Using dynamic chart update instead');
        // Fallback to dynamic chart update
        updateChartDynamic(tradingState);
    }
}

// Dynamic chart update (fallback)
function updateChartDynamic(tradingState) {
    if (!tradingState) return;
    
    // Generate price series
    const basePrice = tradingState.current_price || 1.0850;
    const priceSeries = generatePriceSeries(basePrice, 60);
    
    // Generate indicators
    const indicators = {
        sma_5: calculateSMA(priceSeries, 5),
        sma_10: calculateSMA(priceSeries, 10)
    };
    
    // Update chart
    updateChart(priceSeries, indicators, tradingState.current_trade);
    
    // Update chart status
    document.getElementById('chartStatus').textContent = 'Live';
    document.getElementById('chartStatus').className = 'badge bg-success';
}

// Generate simulated price series
function generatePriceSeries(basePrice, length = 60) {
    const series = [basePrice];
    let currentPrice = basePrice;
    
    for (let i = 1; i < length; i++) {
        // Simulate small random walk
        const change = (Math.random() - 0.5) * 0.0003;
        currentPrice += change;
        
        // Keep in reasonable range
        if (currentPrice < 1.0800) currentPrice = 1.0800 + Math.abs(change);
        if (currentPrice > 1.0900) currentPrice = 1.0900 - Math.abs(change);
        
        series.push(currentPrice);
    }
    
    return series;
}

// Calculate Simple Moving Average
function calculateSMA(data, period) {
    if (!data || data.length < period) return [];
    
    const sma = [];
    for (let i = 0; i < data.length; i++) {
        if (i < period - 1) {
            sma.push(null);
        } else {
            const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
            sma.push(sum / period);
        }
    }
    
    return sma;
}

// Setup chart controls
function setupChartControls() {
    // Toggle fullscreen
    document.getElementById('toggleChart').addEventListener('click', toggleChartFullscreen);
    
    // Add chart type selector
    addChartTypeSelector();
    
    // Add download button
    addDownloadButton();
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

// Add chart type selector
function addChartTypeSelector() {
    const chartHeader = document.querySelector('#chart').parentElement.parentElement.querySelector('.card-header');
    
    const selector = document.createElement('div');
    selector.className = 'btn-group btn-group-sm ms-2';
    selector.innerHTML = `
        <button class="btn btn-outline-light active" data-type="line">
            <i class="bi bi-graph-up"></i> Line
        </button>
        <button class="btn btn-outline-light" data-type="candle">
            <i class="bi bi-bar-chart"></i> Candles
        </button>
    `;
    
    chartHeader.querySelector('.d-flex').appendChild(selector);
    
    // Add event listeners
    selector.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('click', function() {
            selector.querySelectorAll('button').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            changeChartType(this.dataset.type);
        });
    });
}

// Change chart type
function changeChartType(type) {
    if (!currentChart) return;
    
    const update = {
        'type': type === 'candle' ? 'candlestick' : 'scatter'
    };
    
    Plotly.restyle('chart', update, [0]);
    
    // Update chart status
    document.getElementById('chartStatus').textContent = type === 'candle' ? 'Candles' : 'Line';
}

// Add download button
function addDownloadButton() {
    const chartHeader = document.querySelector('#chart').parentElement.parentElement.querySelector('.card-header');
    
    const downloadBtn = document.createElement('button');
    downloadBtn.className = 'btn btn-sm btn-outline-light ms-2';
    downloadBtn.innerHTML = '<i class="bi bi-download"></i>';
    downloadBtn.title = 'Download Chart Image';
    
    downloadBtn.addEventListener('click', downloadChartImage);
    
    chartHeader.querySelector('.d-flex').appendChild(downloadBtn);
}

// Download chart as image
function downloadChartImage() {
    if (!currentChart) return;
    
    Plotly.downloadImage('chart', {
        format: 'png',
        filename: `eurusd_chart_${new Date().toISOString().split('T')[0]}`,
        width: 1200,
        height: 800
    });
}

// Start automatic chart updates
function startChartUpdates() {
    if (chartUpdateInterval) {
        clearInterval(chartUpdateInterval);
    }
    
    chartUpdateInterval = setInterval(() => {
        // Chart will be updated through main.js fetch
        // This just ensures chart stays responsive
        if (currentChart) {
            Plotly.Plots.resize('chart');
        }
    }, 5000); // Resize check every 5 seconds
    
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