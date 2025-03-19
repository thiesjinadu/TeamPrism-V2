// Main JavaScript file for the feedback analysis dashboard

// Show loading overlay
function showLoading() {
    document.getElementById('loading-overlay').classList.add('active');
}

// Hide loading overlay
function hideLoading() {
    document.getElementById('loading-overlay').classList.remove('active');
}

// Initialize charts
function initializeCharts() {
    // Feedback Distribution Chart
    const feedbackCtx = document.getElementById('feedback-distribution').getContext('2d');
    new Chart(feedbackCtx, {
        type: 'doughnut',
        data: {
            labels: feedbackData.labels,
            datasets: [{
                data: feedbackData.data,
                backgroundColor: [
                    '#10B981', // Positive - Green
                    '#EF4444', // Negative - Red
                    '#6B7280', // Neutral - Gray
                    '#F59E0B'  // Mixed - Yellow
                ]
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                }
            }
        }
    });

    // Group Performance Chart
    const performanceCtx = document.getElementById('group-performance').getContext('2d');
    new Chart(performanceCtx, {
        type: 'bar',
        data: {
            labels: performanceData.labels,
            datasets: [{
                label: 'Performance Score',
                data: performanceData.data,
                backgroundColor: '#4F46E5'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

// Handle file upload
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            showLoading();
            
            const formData = new FormData(this);
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (response.ok) {
                    window.location.href = '/';
                } else {
                    throw new Error('Upload failed');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred during upload. Please try again.');
            })
            .finally(() => {
                hideLoading();
            });
        });
    }

    // Initialize charts if we're on the dashboard
    if (document.getElementById('feedback-distribution')) {
        initializeCharts();
    }
});

// Handle model selection
function updateModelSelection() {
    const modelSelect = document.getElementById('model-select');
    if (modelSelect) {
        fetch('/api/v1/models')
            .then(response => response.json())
            .then(models => {
                modelSelect.innerHTML = Object.entries(models)
                    .map(([key, config]) => `<option value="${key}">${config.name}</option>`)
                    .join('');
            })
            .catch(error => console.error('Error fetching models:', error));
    }
}

// Refresh dashboard data
function refreshDashboard() {
    showLoading();
    fetch('/')
        .then(response => response.text())
        .then(html => {
            document.documentElement.innerHTML = html;
            initializeCharts();
        })
        .catch(error => console.error('Error refreshing dashboard:', error))
        .finally(() => {
            hideLoading();
        });
}

// Handle insight card interactions
document.addEventListener('click', function(e) {
    if (e.target.closest('.insight-card')) {
        const card = e.target.closest('.insight-card');
        card.classList.toggle('expanded');
    }
}); 