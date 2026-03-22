let MAX_POINTS = 30;

const RANGES = {
  Voltage: { min: 200, max: 250 },
  Current: { min: 0, max: 3 },
  Temperature: { min: 20, max: 80 },
  Power: { min: 0, max: 300 },
  Vibration: { min: 0, max: 3 }
};

const COLORS = {
  Voltage: "#16a34a",
  Current: "#2563eb",
  Temperature: "#f59e0b",
  Power: "#7c3aed",
  Vibration: "#ef4444"
};

export function createChart(ctx) {
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "Temperature",
        data: [],
        borderColor: COLORS.Temperature,
        borderWidth: 2,
        tension: 0.3,
        pointRadius: 3,
        pointBackgroundColor: []
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { display: false },
        y: {
          min: RANGES.Temperature.min,
          max: RANGES.Temperature.max
        }
      }
    }
  });
}

export function updateChart(chart, value, isAnomaly) {
  chart.data.labels.push("");

  chart.data.datasets[0].data.push(value || 0);

  chart.data.datasets[0].pointBackgroundColor.push(
    isAnomaly ? "red" : chart.data.datasets[0].borderColor
  );

  if (chart.data.labels.length > MAX_POINTS) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
    chart.data.datasets[0].pointBackgroundColor.shift();
  }

  chart.update("none");
}

export function switchMetric(chart, metric) {
  chart.data.labels = [];
  chart.data.datasets[0].data = [];
  chart.data.datasets[0].pointBackgroundColor = [];

  chart.data.datasets[0].label = metric;
  chart.data.datasets[0].borderColor = COLORS[metric];

  chart.options.scales.y.min = RANGES[metric].min;
  chart.options.scales.y.max = RANGES[metric].max;

  chart.update();
}