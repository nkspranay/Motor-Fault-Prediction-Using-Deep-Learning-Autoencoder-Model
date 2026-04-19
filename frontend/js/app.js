import { fetchData, fetchHealth } from "./api.js";
import { createChart, updateChart, switchMetric, createMSEChart, updateMSEChart } from "./chart.js";
import { updateStatus, updateMetrics, updateHistory, showToast, setConnected } from "./ui.js";
import { setChartBackground } from "./chart.js";

let chart;
let mseChart;
let currentMetric   = "Temperature";
let lastStatus      = "Normal";
let lastAlertTime   = 0;
const ALERT_COOLDOWN = 15000;

window.onload = async () => {
  const ctx    = document.getElementById("chart").getContext("2d");
  const mseCtx = document.getElementById("mse-chart").getContext("2d");
  chart    = createChart(ctx);
  mseChart = createMSEChart(mseCtx);

  // Check mode on startup
  const health = await fetchHealth();
  if (health) {
    const modeEl = document.getElementById("mode-badge");
    if (modeEl) {
      modeEl.textContent = health.mode === "real" ? "REAL" : "DEMO";
      modeEl.style.background = health.mode === "real" ? "#16a34a" : "#d97706";
    }
  }

  mainLoop();
  setInterval(mainLoop, 1000);
};

async function mainLoop() {
  const data = await fetchData();

  if (!data) {
    setConnected(false);
    return;
  }
  setConnected(true);

  setChartBackground(chart, data.status);

  updateStatus(data.status, data.prediction);
  updateMetrics(data.values);
  updateHistory(data.history);
  updateMSEChart(mseChart, data.mse_history);

  const now = Date.now();

  // ── Toast notifications (non-blocking) ──
  if (data.status !== lastStatus && now - lastAlertTime > ALERT_COOLDOWN) {

    if (data.status === "Warning") {
      showToast("warning", "Warning", "Possible fault developing — monitor closely.");
      playBeep("warning");
    }

    if (data.status === "Fault") {
      const faultList = (data.faults || [])
  .map(f => `${f.feature} (${f.type})`)
  .join(", ") || "Unknown";
      showToast("fault", "Fault Detected", `Affected: ${faultList}`);
      playBeep("fault");
    }

    if (data.status === "Normal" && lastStatus !== "Normal") {
      showToast("normal", "Recovered", "System back to normal.");
    }

    lastStatus    = data.status;
    lastAlertTime = now;
  }

  // Prediction toast (separate cooldown)
  if (data.prediction && data.status === "Normal") {
    const key = "pred_" + Math.floor(now / 30000);   // max once per 30s
    if (!sessionStorage.getItem(key)) {
      sessionStorage.setItem(key, "1");
      showToast("predict", "Prediction", "Rising error pattern detected — possible fault forming.");
    }
  }

  // ── Auto-switch chart to faulted feature ──
  if (data.status === "Fault" && (data.faults || []).length > 0) {
    const faultMetric = data.faults[0]?.feature;
    if (faultMetric !== currentMetric) {
      currentMetric = faultMetric;
      switchMetric(chart, faultMetric);
    }
  }

  const isAnomaly = (data.faults || [])
  .map(f => f.feature)
  .includes(currentMetric);
  updateChart(chart, data.values[currentMetric], isAnomaly);
}

// Public
window.setMetric = function (metric) {
  currentMetric = metric;
  switchMetric(chart, metric);
};

window.showPage = function (page) {
  document.querySelectorAll(".page").forEach(p => p.classList.add("hidden"));
  document.getElementById(page).classList.remove("hidden");

  // Highlight active nav button
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  const btn = document.querySelector(`[onclick="showPage('${page}')"]`);
  if (btn) btn.classList.add("active");
};

// ── Beep ──
function playBeep(type) {
  try {
    const ctx  = new (window.AudioContext || window.webkitAudioContext)();
    const osc  = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);

    osc.frequency.value = type === "fault" ? 880 : type === "warning" ? 660 : 440;
    gain.gain.setValueAtTime(0.3, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.4);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.4);
  } catch (e) {
    // Audio context not available — silent fail
  }
}