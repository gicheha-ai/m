// ==================== CHART MANAGEMENT ====================
let chartInstance = null;

// Initialize chart functions
function updateChart(chartData) {
    try {
        const chartElement = document.getElementById('chart');
        if (!chartElement) return;
        
        if (chartInstance) {
            Plotly.react('chart', JSON.parse(chartData).data, JSON.parse(chartData).layout);
        } else {
            chartInstance = Plotly.newPlot('chart', JSON.parse(chartData).data, JSON.parse(chartData).layout);
        }
        
        document.getElementById('chartStatus').textContent = 'Live Chart';
        document.getElementById('chartStatus').className = 'badge bg-success me-2';
    } catch (error) {
        console.error('Error updating chart:', error);
        document.getElementById('chartStatus').textContent = 'Chart Error';
        document.getElementById('chartStatus').className = 'badge bg-danger me-2';
    }
}

function toggleChartFullscreen() {
    const chartDiv = document.getElementById('chart');
    if (!document.fullscreenElement) {
        chartDiv.requestFullscreen().catch(err => {
            console.error(`Error attempting to enable fullscreen: ${err.message}`);
        });
        document.getElementById('toggleChart').innerHTML = '<i class="bi bi-arrows-angle-contract"></i> Exit Fullscreen';
    } else {
        document.exitFullscreen();
        document.getElementById('toggleChart').innerHTML = '<i class="bi bi-arrows-angle-expand"></i> Fullscreen';
    }
}

// Add event listener for chart toggle
document.addEventListener('DOMContentLoaded', function() {
    const toggleChartBtn = document.getElementById('toggleChart');
    if (toggleChartBtn) {
        toggleChartBtn.addEventListener('click', toggleChartFullscreen);
    }
});

// Export chart functions
window.updateChart = updateChart;
window.toggleChartFullscreen = toggleChartFullscreen;