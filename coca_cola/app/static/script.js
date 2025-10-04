let currentData = null;
let currentPredictions = null;

// Initialize dashboard when page loads
$(document).ready(function() {
    loadDashboard();
    // Refresh data every 30 seconds
    setInterval(loadDashboard, 30000);
});

function showLoading() {
    $('#loading-overlay').show();
}

function hideLoading() {
    $('#loading-overlay').hide();
}

function formatPrice(price) {
    return '$' + price.toFixed(2);
}

function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function formatPercent(percent) {
    return percent.toFixed(2) + '%';
}

function updateStatistics(stats) {
    // Update current price with animation
    $('#current-price').text(formatPrice(stats.current_price));
    $('#current-price').addClass('price-update');
    setTimeout(() => {
        $('#current-price').removeClass('price-update');
    }, 500);

    // Update price change
    const changeElement = $('#price-change');
    changeElement.text(formatPrice(stats.price_change) + ' (' + formatPercent(stats.percent_change) + ')');
    
    if (stats.price_change >= 0) {
        changeElement.removeClass('negative').addClass('positive');
    } else {
        changeElement.removeClass('positive').addClass('negative');
    }

    // Update other stats
    $('#volume').text(stats.volume);
    $('#avg-volume').text(stats.avg_volume);
    $('#high-52week').text(formatPrice(stats.high_52week));
    $('#low-52week').text(formatPrice(stats.low_52week));
}

function updateCharts(plots) {
    $('#price-chart').html(plots.price_chart);
    $('#volume-chart').html(plots.volume_chart);
    $('#returns-chart').html(plots.returns_chart);
}

function showStatus(message, type = 'info') {
    const statusElement = $('#training-status');
    const textElement = $('#training-text');
    
    statusElement.removeClass('success error').addClass(type);
    textElement.text(message);
    statusElement.show();
    
    if (type !== 'info') {
        setTimeout(() => {
            statusElement.hide();
        }, 5000);
    }
}

function hideStatus() {
    $('#training-status').hide();
}

function loadDashboard() {
    showLoading();
    
    $.ajax({
        url: '/api/dashboard',
        type: 'GET',
        success: function(response) {
            currentData = response;
            updateStatistics(response.stats);
            updateCharts(response.plots);
            $('#last-updated').text(response.last_updated);
            hideLoading();
        },
        error: function(xhr, status, error) {
            console.error('Error loading dashboard:', error);
            alert('Error loading dashboard data. Please try again.');
            hideLoading();
        }
    });
}

function refreshData() {
    loadDashboard();
}

function trainModel() {
    showStatus('Training prediction model...', 'info');
    
    $.ajax({
        url: '/api/refresh_model',
        type: 'GET',
        success: function(response) {
            showStatus(
                `Model trained successfully! Train Score: ${(response.train_score * 100).toFixed(1)}%, ` +
                `Test Score: ${(response.test_score * 100).toFixed(1)}%, ` +
                `MAE: $${response.mae.toFixed(2)}`, 
                'success'
            );
        },
        error: function(xhr, status, error) {
            console.error('Error training model:', error);
            showStatus('Error training model. Please try again.', 'error');
        }
    });
}

function refreshModel() {
    trainModel();
}

function getPredictions() {
    showStatus('Generating price predictions...', 'info');
    
    $.ajax({
        url: '/api/predict',
        type: 'GET',
        success: function(response) {
            currentPredictions = response;
            displayPredictions(response);
            showStatus('Predictions generated successfully!', 'success');
        },
        error: function(xhr, status, error) {
            console.error('Error getting predictions:', error);
            showStatus('Error generating predictions. Please try again.', 'error');
        }
    });
}

function displayPredictions(predictions) {
    const tableBody = $('#predictions-body');
    tableBody.empty();
    
    let firstPrice = null;
    let lastPrice = null;
    
    predictions.prices.forEach((price, index) => {
        if (index === 0) firstPrice = price;
        if (index === predictions.prices.length - 1) lastPrice = price;
        
        const date = predictions.dates[index];
        const change = index > 0 ? price - predictions.prices[index - 1] : 0;
        const changePercent = index > 0 ? (change / predictions.prices[index - 1]) * 100 : 0;
        
        const row = `
            <tr>
                <td>${date}</td>
                <td>${formatPrice(price)}</td>
                <td class="${change >= 0 ? 'positive' : 'negative'}">
                    ${change >= 0 ? '+' : ''}${formatPrice(change)} (${change >= 0 ? '+' : ''}${changePercent.toFixed(2)}%)
                </td>
            </tr>
        `;
        tableBody.append(row);
    });
    
    // Calculate overall change
    if (firstPrice && lastPrice) {
        const totalChange = lastPrice - firstPrice;
        const totalChangePercent = (totalChange / firstPrice) * 100;
        
        const metricsHtml = `
            <div><strong>30-Day Forecast:</strong> ${formatPrice(firstPrice)} → ${formatPrice(lastPrice)}</div>
            <div class="${totalChange >= 0 ? 'positive' : 'negative'}">
                <strong>Projected Change:</strong> ${totalChange >= 0 ? '+' : ''}${formatPrice(totalChange)} (${totalChange >= 0 ? '+' : ''}${totalChangePercent.toFixed(2)}%)
            </div>
        `;
        $('#prediction-metrics').html(metricsHtml);
    }
    
    $('#prediction-results').show();
    
    // Scroll to predictions
    $('html, body').animate({
        scrollTop: $('#prediction-results').offset().top
    }, 1000);
}

// Keyboard shortcuts
$(document).keydown(function(e) {
    // Ctrl + R to refresh
    if (e.ctrlKey && e.key === 'r') {
        e.preventDefault();
        refreshData();
    }
    // Ctrl + P for predictions
    if (e.ctrlKey && e.key === 'p') {
        e.preventDefault();
        getPredictions();
    }
    // Ctrl + T to train model
    if (e.ctrlKey && e.key === 't') {
        e.preventDefault();
        trainModel();
    }
});

// Add some visual effects
$(document).ready(function() {
    // Add hover effects to cards
    $('.stat-card').hover(
        function() {
            $(this).css('transform', 'translateY(-5px)');
        },
        function() {
            $(this).css('transform', 'translateY(0)');
        }
    );
});
// Add this function to the existing script.js file
function checkModelStatus() {
    $.ajax({
        url: '/api/model_status',
        type: 'GET',
        success: function(response) {
            if (!response.is_trained) {
                showStatus('Model not trained yet. Click "Train Prediction Model" to start.', 'info');
            }
        },
        error: function() {
            console.log('Could not check model status');
        }
    });
}

// Update the document ready function
$(document).ready(function() {
    loadDashboard();
    // Check model status after a short delay
    setTimeout(checkModelStatus, 1000);
    // Refresh data every 30 seconds
    setInterval(loadDashboard, 30000);
});

// Update the trainModel function to handle errors better
function trainModel() {
    showStatus('Training prediction model... This may take a moment.', 'info');
    
    $.ajax({
        url: '/api/refresh_model',
        type: 'GET',
        success: function(response) {
            if (response.status === 'success') {
                showStatus(
                    `Model trained successfully! Train Score: ${(response.train_score * 100).toFixed(1)}%, ` +
                    `Test Score: ${(response.test_score * 100).toFixed(1)}%, ` +
                    `MAE: $${response.mae.toFixed(2)}`, 
                    'success'
                );
            } else {
                showStatus('Error: ' + (response.error || 'Training failed'), 'error');
            }
        },
        error: function(xhr, status, error) {
            console.error('Error training model:', error);
            showStatus('Error training model. Please try again.', 'error');
        }
    });
}

// Update the getPredictions function
function getPredictions() {
    showStatus('Generating price predictions...', 'info');
    
    $.ajax({
        url: '/api/predict',
        type: 'GET',
        success: function(response) {
            if (response.error) {
                showStatus('Error: ' + response.error, 'error');
            } else {
                currentPredictions = response;
                displayPredictions(response);
                showStatus('Predictions generated successfully!', 'success');
            }
        },
        error: function(xhr, status, error) {
            console.error('Error getting predictions:', error);
            showStatus('Error generating predictions. Please try again.', 'error');
        }
    });
}
// Update the loadDashboard function to handle data source
function loadDashboard() {
    showLoading();
    
    $.ajax({
        url: '/api/dashboard',
        type: 'GET',
        success: function(response) {
            currentData = response;
            updateStatistics(response.stats);
            updateCharts(response.plots);
            $('#last-updated').text(response.last_updated);
            
            // Update data source indicator
            updateDataSourceIndicator(response.stats.data_source);
            
            hideLoading();
        },
        error: function(xhr, status, error) {
            console.error('Error loading dashboard:', error);
            alert('Error loading dashboard data. Please try again.');
            hideLoading();
        }
    });
}

function updateDataSourceIndicator(source) {
    const sourceElement = $('#data-source');
    const sourceText = $('#data-source-text');
    
    if (source === 'Sample Data') {
        sourceText.text('Using sample data - Yahoo Finance connection failed');
        sourceElement.addClass('warning').show();
    } else if (source === 'Yahoo Finance') {
        sourceText.text('Using live data from Yahoo Finance');
        sourceElement.removeClass('warning').show();
        // Hide after 5 seconds if it's live data
        setTimeout(() => {
            sourceElement.fadeOut();
        }, 5000);
    } else {
        sourceElement.hide();
    }
}

// Add health check function
function checkHealth() {
    $.ajax({
        url: '/api/health',
        type: 'GET',
        success: function(response) {
            console.log('Health check:', response);
            if (!response.data_available) {
                updateDataSourceIndicator('Sample Data');
            }
        },
        error: function() {
            console.log('Health check failed');
        }
    });
}