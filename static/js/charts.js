/**
 * EUR/USD Trading System - Chart Management Module
 * Updated for 2-Minute Cycles
 */

let currentChart = null;
let priceChartData = [];
let chartUpdateInterval = null;
let isFullscreen = false;
const CHART_POINTS = 120; // ⭐ Changed from 60 to 120 for 2-minute chart

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

// Get chart layout configuration (updated for 2-minute)
function getChartLayout() {
    return {
        title: {
            text: 'EUR/USD Live Price Chart (2-Minute View)',  // ⭐ 2-Minute
            font: { size: 18, color: '#ffffff' },
            x: 0.05
        },
        xaxis: {
            title: { text: 'Time (seconds ago - 120s view)', font: { color: '#ffffff' } },  // ⭐ 120s
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
        x: Array.from({length: CHART_POINTS}, (_, i) => i),  // ⭐ 120 points
        y: Array(CHART_POINTS).fill(1.08500),
        type: 'scatter',
        mode: 'lines',
        name: 'EUR/USD Price (2-Min)',  // ⭐ 2-Min
        line: {
            color: '#00ff88',
            width: 3
        },
        hovertemplate: 'Price: %{y:.5f}<extra></extra>'
    };
}

// Update chart with new data (adjusted for 120 points)
function updateChart(priceData, indicators = {}, tradeInfo = null) {
    if (!currentChart || !priceData || priceData.length === 0) {
        return;
    }
    
    try {
        // Ensure we have 120 points
        const paddedPriceData = priceData.length >= CHART_POINTS 
            ? priceData.slice(-CHART_POINTS)
            : [...Array(CHART_POINTS - priceData.length).fill(priceData[0]), ...priceData];
        
        // Prepare time axis (seconds ago)
        const timePoints = Array.from({length: CHART_POINTS}, (_, i) => CHART_POINTS - i - 1);
        
        // Update main price trace
        const traces = [{
            x: timePoints,
            y: paddedPriceData,
            type: 'scatter',
            mode: 'lines',
            name: 'EUR/USD Price (2-Min)',
            line: { color: '#00ff88', width: 3 },
            hovertemplate: 'Price: %{y:.5f}<extra></extra>'
        }];
        
        // Add moving averages if available
        if (indicators.sma_5 && indicators.sma_5.length > 0) {
            const sma5Data = indicators.sma_5.length >= CHART_POINTS 
                ? indicators.sma_5.slice(-CHART_POINTS)
                : [...Array(CHART_POINTS - indicators.sma_5.length).fill(indicators.sma_5[0]), ...indicators.sma_5];
            
            traces.push({
                x: timePoints,
                y: sma5Data,
                type: 'scatter',
                mode: 'lines',
                name: 'SMA 5 (2-Min)',
                line: { color: 'orange', width: 1.5, dash: 'dash' },
                opacity: 0.7,
                hovertemplate: 'SMA 5: %{y:.5f}<extra></extra>'
            });
        }
        
        if (indicators.sma_10 && indicators.sma_10.length > 0) {
            const sma10Data = indicators.sma_10.length >= CHART_POINTS 
                ? indicators.sma_10.slice(-CHART_POINTS)
                : [...Array(CHART_POINTS - indicators.sma_10.length).fill(indicators.sma_10[0]), ...indicators.sma_10];
            
            traces.push({
                x: timePoints,
                y: sma10Data,
                type: 'scatter',
                mode: 'lines',
                name: 'SMA 10 (2-Min)',
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
            addTradeToChart(tradeInfo, timePoints, paddedPriceData, layout);
        }
        
        // Update chart title
        const isDemo = document.getElementById('dataSource').textContent.includes('Cache') || 
                       document.getElementById('dataSource').textContent.includes('Simulation');
        layout.title.text = `EUR/USD 2-Minute Price ${isDemo ? '(Cached Data)' : ''}`;
        
        Plotly.react('chart', traces, layout);
        
        // Store latest price data
        priceChartData = paddedPriceData;
        
        // Update chart status
        document.getElementById('chartStatus').textContent = '2-Min Live';
        document.getElementById('chartStatus').className = 'badge bg-success';
        document.getElementById('chartTimeframe').textContent = '120s';
        
    } catch (error) {
        console.error('Chart update error:', error);
    }
}

// Add trade information to chart (adjusted for 120-second view)
function addTradeToChart(trade, timePoints, priceData, layout) {
    if (!trade || !trade.entry_price) return;
    
    const lastIndex = timePoints.length - 1;
    const currentPrice = priceData[priceData.length - 1];
    const tradeStartIndex = Math.max(0, lastIndex - 40);  // ⭐ Show trade starting 40 points back
    
    // Add entry point marker
    layout.annotations.push({
        x: tradeStartIndex,
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
    
    // Add TP line (span full chart width for visibility)
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
        
        layout.annotations.push({
            x: lastIndex,
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
        
        layout.annotations.push({
            x: lastIndex,
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
        x: lastIndex,
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

// Generate simulated price series for 120 seconds
function generatePriceSeries(basePrice, length = CHART_POINTS) {
    const series = [basePrice];
    let currentPrice = basePrice;
    
    for (let i = 1; i < length; i++) {
        // Simulate random walk (lower volatility for 2-min view)
        const change = (Math.random() - 0.5) * 0.00015;
        currentPrice += change;
        
        // Keep in reasonable range
        if (currentPrice < 1.0800) currentPrice = 1.0800 + Math.abs(change);
        if (currentPrice > 1.0900) currentPrice = 1.0900 - Math.abs(change);
        
        series.push(currentPrice);
    }
    
    return series;
}

// Update chart title with timeframe
function updateChartTitle() {
    if (!currentChart) return;
    
    const chartTitle = document.querySelector('.js-plotly-plot .plotly .main-svg .g-gtitle text');
    if (chartTitle) {
        chartTitle.textContent = `EUR/USD 2-Minute Trading Chart`;
    }
}

// Export functions for use in main.js
window.initializeChartSystem = initializeChartSystem;
window.updateChartFromState = updateChartFromState;
window.updateChart = updateChart;
window.startChartUpdates = startChartUpdates;
window.stopChartUpdates = stopChartUpdates;
window.updateChartTitle = updateChartTitle;

console.log('Charts.js loaded successfully (2-Minute Cycle)');