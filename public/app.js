// Global chart instances
let chartModelComp = null;
let chartRiskDist = null;
let chartHypothesisBins = null;
let chartCostCurve = null;

// Tab switching
function switchTab(tabId) {
  document.querySelectorAll('nav button').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
  
  const selectedBtn = Array.from(document.querySelectorAll('nav button')).find(b => b.getAttribute('onclick').includes(tabId));
  if (selectedBtn) selectedBtn.classList.add('active');
  
  const selectedContent = document.getElementById(`tab-${tabId}`);
  if (selectedContent) selectedContent.classList.add('active');
}

// Fetch and Render Overview Dashboard
async function loadOverviewData() {
  try {
    const resMetrics = await fetch('/api/metrics');
    const resEval = await fetch('/api/evaluation');
    const resSummary = await fetch('/api/test-summary');
    
    if (!resMetrics.ok || !resEval.ok || !resSummary.ok) return;
    
    const metricsData = await resMetrics.json();
    const evalData = await resEval.json();
    const summaryData = await resSummary.json();
    
    // KPI Updates
    document.getElementById('kpi-best-model').innerText = metricsData.meta.best_model_name;
    document.getElementById('kpi-opt-thresh').innerText = evalData.cost_optimization.cost_optimal_threshold.threshold;
    document.getElementById('kpi-savings').innerText = `₹${evalData.cost_optimization.cost_optimal_threshold.savings_vs_no_ml.toLocaleString()}`;
    
    // Chart 1: Model Comparison Bar Chart
    const models = metricsData.models;
    const modelLabels = Object.keys(models);
    const accuracyVals = modelLabels.map(m => models[m].validation_metrics.accuracy);
    const aucVals = modelLabels.map(m => models[m].validation_metrics.roc_auc);
    
    const ctxComp = document.getElementById('chart-model-comparison').getContext('2d');
    if (chartModelComp) chartModelComp.destroy();
    chartModelComp = new Chart(ctxComp, {
      type: 'bar',
      data: {
        labels: modelLabels,
        datasets: [
          { label: 'Accuracy', data: accuracyVals, backgroundColor: '#3b82f6' },
          { label: 'ROC-AUC', data: aucVals, backgroundColor: '#8b5cf6' }
        ]
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#9ca3af' } } },
        scales: {
          x: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: { min: 0.4, max: 0.7, ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });

    // Chart 2: Test Dataset Risk Doughnut Chart
    const dist = summaryData.risk_category_distribution;
    const ctxDist = document.getElementById('chart-risk-dist').getContext('2d');
    if (chartRiskDist) chartRiskDist.destroy();
    chartRiskDist = new Chart(ctxDist, {
      type: 'doughnut',
      data: {
        labels: Object.keys(dist),
        datasets: [{
          data: Object.values(dist),
          backgroundColor: ['#f59e0b', '#10b981', '#f43f5e', '#06b6d4']
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { position: 'right', labels: { color: '#9ca3af' } } }
      }
    });

  } catch (err) {
    console.error("Error loading overview data:", err);
  }
}

// Fetch and Render Hypothesis Proof Data
async function loadHypothesisData() {
  try {
    const res = await fetch('/api/hypothesis');
    if (!res.ok) return;
    const hypData = await res.json();
    
    document.getElementById('hyp-text-summary').innerText = hypData.hypothesis_summary.conclusion_text;
    
    // Bins Chart
    const bins = hypData.binned_analysis;
    const binLabels = bins.map(b => b.bin);
    const binRates = bins.map(b => (b.actual_return_rate * 100).toFixed(1));
    
    const ctxBins = document.getElementById('chart-hypothesis-bins').getContext('2d');
    if (chartHypothesisBins) chartHypothesisBins.destroy();
    chartHypothesisBins = new Chart(ctxBins, {
      type: 'bar',
      data: {
        labels: binLabels,
        datasets: [{
          label: 'Actual Order Return Rate (%)',
          data: binRates,
          backgroundColor: '#06b6d4'
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#9ca3af' } } },
        scales: {
          x: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } },
          y: { min: 40, max: 60, ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } }
        }
      }
    });

    // Model comparison table
    const tableBody = document.getElementById('table-hypothesis-models');
    tableBody.innerHTML = `
      <tr>
        <td><strong>${hypData.baseline_metrics.model}</strong></td>
        <td>${(hypData.baseline_metrics.accuracy * 100).toFixed(1)}%</td>
        <td>${hypData.baseline_metrics.f1}</td>
        <td>${hypData.baseline_metrics.roc_auc}</td>
        <td>${hypData.baseline_metrics.log_loss}</td>
      </tr>
      <tr>
        <td><strong style="color: var(--accent-cyan);">${hypData.full_model_metrics.model}</strong></td>
        <td>${(hypData.full_model_metrics.accuracy * 100).toFixed(1)}%</td>
        <td>${hypData.full_model_metrics.f1}</td>
        <td><strong style="color: var(--risk-low);">${hypData.full_model_metrics.roc_auc}</strong></td>
        <td>${hypData.full_model_metrics.log_loss}</td>
      </tr>
    `;

  } catch (err) {
    console.error("Error loading hypothesis data:", err);
  }
}

// Fetch & Recalculate Cost Optimization Matrix
async function loadCostOptimization() {
  try {
    const res = await fetch('/api/evaluation');
    if (!res.ok) return;
    const evalData = await res.json();
    renderCostCurve(evalData.cost_optimization);
  } catch (err) {
    console.error("Error loading cost curve:", err);
  }
}

function renderCostCurve(costData) {
  document.getElementById('opt-thresh-val').innerText = costData.cost_optimal_threshold.threshold;
  document.getElementById('opt-cost-val').innerText = `₹${costData.cost_optimal_threshold.min_business_cost.toLocaleString()}`;
  document.getElementById('opt-savings-val').innerText = `₹${costData.cost_optimal_threshold.savings_vs_default.toLocaleString()}`;
  
  const curve = costData.cost_curve;
  const threshLabels = curve.map(c => c.threshold);
  const totalCosts = curve.map(c => c.total_business_cost);
  
  const ctxCurve = document.getElementById('chart-cost-curve').getContext('2d');
  if (chartCostCurve) chartCostCurve.destroy();
  chartCostCurve = new Chart(ctxCurve, {
    type: 'line',
    data: {
      labels: threshLabels,
      datasets: [{
        label: 'Total Business Financial Cost (₹)',
        data: totalCosts,
        borderColor: '#f43f5e',
        backgroundColor: 'rgba(244, 63, 94, 0.1)',
        fill: true,
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#9ca3af' } } },
      scales: {
        x: { title: { display: true, text: 'Decision Threshold', color: '#9ca3af' }, ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } },
        y: { title: { display: true, text: 'Business Loss Cost (₹)', color: '#9ca3af' }, ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.05)' } }
      }
    }
  });
}

async function recalculateCostOptimization() {
  const fpCost = parseFloat(document.getElementById('input-fp-cost').value);
  const fnCost = parseFloat(document.getElementById('input-fn-cost').value);
  
  try {
    const res = await fetch('/api/recalculate-cost', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fp_cost: fpCost, fn_cost: fnCost })
    });
    if (!res.ok) return;
    const data = await res.json();
    renderCostCurve(data);
  } catch (err) {
    console.error("Error recalculating cost matrix:", err);
  }
}

// Single Order Real-Time Scorer
async function scoreOrder() {
  const circle = document.getElementById('result-circle');
  const scoreVal = document.getElementById('result-score');
  const scoreBadge = document.getElementById('result-badge');
  const factorsContainer = document.getElementById('result-factors');
  const recBox = document.getElementById('result-recommendation');

  // Set loading state
  if (scoreVal) scoreVal.innerText = "...";
  if (scoreBadge) scoreBadge.innerText = "SCORING";

  const payload = {
    order_id: 10001,
    customer_age: parseInt(document.getElementById('field-age').value) || 30,
    past_purchase_count: parseInt(document.getElementById('field-purchases').value) || 0,
    past_return_rate: parseFloat(document.getElementById('field-return-rate').value) || 0.0,
    product_price: parseFloat(document.getElementById('field-price').value) || 50.0,
    discount_percent: parseFloat(document.getElementById('field-discount').value) || 0.0,
    delivery_delay_days: parseFloat(document.getElementById('field-delay').value) || 0.0,
    product_rating: parseFloat(document.getElementById('field-rating').value) || 4.2,
    session_length_minutes: parseFloat(document.getElementById('field-session').value) || 25.0,
    num_product_views: parseInt(document.getElementById('field-views').value) || 8,
    used_coupon: parseInt(document.getElementById('field-coupon').value),
    product_category: document.getElementById('field-category').value,
    shipping_method: document.getElementById('field-shipping').value,
    payment_method: document.getElementById('field-payment').value,
    device_type: document.getElementById('field-device').value,
    occasion_period: document.getElementById('field-occasion') ? document.getElementById('field-occasion').value : 'none'
  };

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      if (scoreVal) scoreVal.innerText = "ERR";
      if (scoreBadge) scoreBadge.innerText = "API ERROR";
      if (recBox) recBox.innerText = `⚠️ API Error (${res.status}): Make sure the FastAPI backend is running via: python -m uvicorn api.main:app --reload`;
      return;
    }

    const result = await res.json();

    // UI Update
    if (scoreVal) scoreVal.innerText = `${result.risk_score_percent}%`;
    if (scoreBadge) scoreBadge.innerText = result.risk_category;

    if (result.risk_category === 'HIGH') {
      circle.style.borderColor = 'var(--risk-high)';
      circle.style.boxShadow = '0 0 24px rgba(244, 63, 94, 0.4)';
    } else if (result.risk_category === 'MEDIUM') {
      circle.style.borderColor = 'var(--risk-med)';
      circle.style.boxShadow = '0 0 24px rgba(245, 158, 11, 0.4)';
    } else if (result.risk_category === 'INSUFFICIENT DATA') {
      circle.style.borderColor = 'var(--risk-info)';
      circle.style.boxShadow = '0 0 24px rgba(6, 182, 212, 0.4)';
    } else {
      circle.style.borderColor = 'var(--risk-low)';
      circle.style.boxShadow = '0 0 24px rgba(16, 185, 129, 0.4)';
    }

    if (factorsContainer) {
      factorsContainer.innerHTML = result.main_risk_factors.map(f => `
        <div class="factor-item">
          <span style="font-size: 13px; font-weight: 500;">${f.feature}</span>
          <span class="badge ${f.impact === 'HIGH' ? 'badge-high' : 'badge-medium'}">${f.impact} IMPACT</span>
        </div>
      `).join('');
    }

    if (recBox) recBox.innerText = result.recommendation;

  } catch (err) {
    console.error("Error scoring order:", err);
    if (scoreVal) scoreVal.innerText = "OFFLINE";
    if (scoreBadge) scoreBadge.innerText = "NO SERVER";
    if (recBox) recBox.innerText = "⚠️ Could not connect to API backend. Please run: python -m uvicorn api.main:app --reload and open http://127.0.0.1:8000";
  }
}

function loadPresetOrder(type) {
  if (type === 'high') {
    document.getElementById('field-age').value = 25;
    document.getElementById('field-purchases').value = 18;
    document.getElementById('field-return-rate').value = 0.65;
    document.getElementById('field-price').value = 350.0;
    document.getElementById('field-discount').value = 55.0;
    document.getElementById('field-delay').value = 4.5;
    document.getElementById('field-rating').value = 2.1;
    document.getElementById('field-session').value = 8.0;
    document.getElementById('field-views').value = 15;
    document.getElementById('field-coupon').value = 1;
    document.getElementById('field-category').value = 'clothing';
    document.getElementById('field-shipping').value = 'express';
    document.getElementById('field-payment').value = 'credit_card';
    document.getElementById('field-device').value = 'mobile';
    if (document.getElementById('field-occasion')) document.getElementById('field-occasion').value = 'diwali_sale';
  } else if (type === 'low') {
    document.getElementById('field-age').value = 42;
    document.getElementById('field-purchases').value = 25;
    document.getElementById('field-return-rate').value = 0.05;
    document.getElementById('field-price').value = 45.0;
    document.getElementById('field-discount').value = 10.0;
    document.getElementById('field-delay').value = 0.0;
    document.getElementById('field-rating').value = 4.7;
    document.getElementById('field-session').value = 35.0;
    document.getElementById('field-views').value = 3;
    document.getElementById('field-coupon').value = 0;
    document.getElementById('field-category').value = 'electronics';
    document.getElementById('field-shipping').value = 'standard';
    document.getElementById('field-payment').value = 'debit_card';
    document.getElementById('field-device').value = 'desktop';
    if (document.getElementById('field-occasion')) document.getElementById('field-occasion').value = 'none';
  } else {
    // Insufficient / Low History Preset
    document.getElementById('field-age').value = 29;
    document.getElementById('field-purchases').value = 1;
    document.getElementById('field-return-rate').value = 0.0;
    document.getElementById('field-price').value = 85.0;
    document.getElementById('field-discount').value = 5.0;
    document.getElementById('field-delay').value = 0.0;
    document.getElementById('field-rating').value = 4.2;
    document.getElementById('field-session').value = 25.0;
    document.getElementById('field-views').value = 1;
    document.getElementById('field-coupon').value = 1;
    document.getElementById('field-category').value = 'home';
    document.getElementById('field-shipping').value = 'standard';
    document.getElementById('field-payment').value = 'apple_pay';
    document.getElementById('field-device').value = 'mobile';
    if (document.getElementById('field-occasion')) document.getElementById('field-occasion').value = 'none';
  }
  scoreOrder();
}

// Fetch Sample Test Orders Table
async function loadSampleTestOrders() {
  try {
    const res = await fetch('/api/sample-orders');
    if (!res.ok) return;
    const orders = await res.json();
    
    const tableBody = document.getElementById('table-test-orders');
    tableBody.innerHTML = orders.map(o => `
      <tr>
        <td><strong>ORD-${o.order_id}</strong></td>
        <td>${(o.return_probability * 100).toFixed(1)}%</td>
        <td><span class="badge badge-${o.risk_category === 'HIGH' ? 'high' : o.risk_category === 'MEDIUM' ? 'medium' : o.risk_category === 'LOW' ? 'low' : 'info'}">${o.risk_category}</span></td>
        <td style="font-size: 12px; color: var(--text-muted);">${o.top_risk_factors}</td>
        <td style="font-size: 12px;">${o.recommendation}</td>
      </tr>
    `).join('');
  } catch (err) {
    console.error("Error loading sample test orders:", err);
  }
}

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', () => {
  loadOverviewData();
  loadHypothesisData();
  loadCostOptimization();
  loadSampleTestOrders();
  scoreOrder();
});
