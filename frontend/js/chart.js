const MAX_POINTS = 40;

const RANGES = {
  Voltage    : { min: 180, max: 260 },
  Current    : { min: 0,   max: 3 },
  Temperature: { min: 20,  max: 80 },
  Power      : { min: 0,   max: 600 },
  Humidity   : { min: 20,  max: 100 },
  Vibration  : { min: 0,   max: 4095 },
};

const COLORS = {
  Voltage    : "#16a34a",
  Current    : "#2563eb",
  Temperature: "#f59e0b",
  Power      : "#7c3aed",
  Humidity   : "#06b6d4",
  Vibration  : "#6366f1",
};

// ── Feature chart ──
export function createChart(ctx) {
  return new Chart(ctx, {
    type: "line",
    data: {
      labels  : [],
      datasets: [{
        label          : "Temperature",
        data           : [],
        borderColor    : COLORS.Temperature,
        borderWidth    : 2,
        tension        : 0.3,
        pointRadius    : 3,
        pointBackgroundColor: [],
        fill           : false,
      }]
    },
    options: {
      responsive           : true,
      maintainAspectRatio  : false,
      animation            : false,
      plugins: {
        legend: { display: false },
        tooltip: { mode: "index", intersect: false },
      },
      scales: {
        x: { display: false },
        y: {
          min: RANGES.Temperature.min,
          max: RANGES.Temperature.max,
          grid: { color: "rgba(0,0,0,0.05)" },
        }
      }
    }
  });
}

export function updateChart(chart, value, isAnomaly) {
  const ds = chart.data.datasets[0];
  chart.data.labels.push("");
  ds.data.push(value ?? 0);
  ds.pointBackgroundColor.push(isAnomaly ? "#ef4444" : ds.borderColor);

  if (chart.data.labels.length > MAX_POINTS) {
    chart.data.labels.shift();
    ds.data.shift();
    ds.pointBackgroundColor.shift();
  }
  chart.update("none");
}

export function switchMetric(chart, metric) {
  const ds             = chart.data.datasets[0];
  chart.data.labels    = [];
  ds.data              = [];
  ds.pointBackgroundColor = [];
  ds.label             = metric;
  ds.borderColor       = COLORS[metric] || "#6366f1";
  chart.options.scales.y.min = RANGES[metric]?.min ?? 0;
  chart.options.scales.y.max = RANGES[metric]?.max ?? 100;
  chart.update();
}

// ── MSE chart ──
export function createMSEChart(ctx) {
  return new Chart(ctx, {
    type: "line",
    data: {
      labels  : [],
      datasets: [{
        label      : "Reconstruction Error (MSE)",
        data       : [],
        borderColor: "#6366f1",
        borderWidth: 1.5,
        tension    : 0.3,
        pointRadius: 0,
        fill       : {
          target: "origin",
          above : "rgba(99,102,241,0.08)",
        },
      }]
    },
    options: {
      responsive          : true,
      maintainAspectRatio : false,
      animation           : false,
      plugins: {
        legend : { display: true, position: "top" },
        tooltip: { mode: "index", intersect: false },
      },
      scales: {
        x: { display: false },
        y: {
          beginAtZero: true,
          grid: { color: "rgba(0,0,0,0.05)" },
        }
      }
    }
  });
}

export function updateMSEChart(chart, mseHistory) {
  if (!mseHistory || mseHistory.length === 0) return;
  chart.data.labels   = mseHistory.map((_, i) => i);
  chart.data.datasets[0].data = mseHistory;
  chart.update("none");
}

export function setChartBackground(chart, status) {
  if (!chart) return;

  const color =
    status === "Fault"   ? "rgba(239,68,68,0.15)" :
    status === "Warning" ? "rgba(245,158,11,0.15)" :
                           "rgba(0,0,0,0)";

  // create plugin once
  if (!chart.$bgPlugin) {
    chart.$bgPlugin = {
      id: "custom_bg",
      beforeDraw(c) {
        const ctx = c.ctx;
        ctx.save();
        ctx.fillStyle = c.$bgColor || "transparent";
        ctx.fillRect(0, 0, c.width, c.height);
        ctx.restore();
      }
    };
    chart.config.plugins.push(chart.$bgPlugin);
  }

  chart.$bgColor = color;
  chart.update("none");
}