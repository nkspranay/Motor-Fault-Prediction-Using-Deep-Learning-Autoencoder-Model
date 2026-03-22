import { fetchData } from "./api.js";
import { createChart, updateChart, switchMetric } from "./chart.js";
import { updateStatus, updateMetrics, updateHistory } from "./ui.js";

let chart;
let currentMetric = "Temperature";

let lastStatus = "Normal";
let lastAlertTime = 0;
const ALERT_COOLDOWN = 10000; // 10 sec

window.onload = () => {
  const ctx = document.getElementById("chart").getContext("2d");
  chart = createChart(ctx);

  mainLoop();
  setInterval(mainLoop, 1000);
};

async function mainLoop() {
  const data = await fetchData();
  if (!data) return;

  updateStatus(data.status);
  updateMetrics(data.values);
  updateHistory(data.history);

  const now = Date.now();

  // -------- ALERT CONTROL --------
  if (
    data.status !== lastStatus &&
    now - lastAlertTime > ALERT_COOLDOWN
  ) {
    if (data.status === "Warning") {
      alert("Warning: Possible upcoming fault");
      playBeep();
    }

    if (data.status === "Fault") {
      alert("Fault detected: " + data.faults.join(", "));
      playBeep();
    }

    lastStatus = data.status;
    lastAlertTime = now;
  }

  // -------- AUTO SWITCH --------
  if (data.status === "Fault" && data.faults.length > 0) {
    const faultMetric = data.faults[0];

    if (faultMetric !== currentMetric) {
      currentMetric = faultMetric;
      switchMetric(chart, faultMetric);
    }
  }

  const isAnomaly = data.faults.includes(currentMetric);

  updateChart(chart, data.values[currentMetric], isAnomaly);
}

window.setMetric = function(metric) {
  currentMetric = metric;
  switchMetric(chart, metric);
};

window.showPage = function(page) {
  document.querySelectorAll(".page").forEach(p => p.classList.add("hidden"));
  document.getElementById(page).classList.remove("hidden");
};

function playBeep() {
  const audio = new Audio("https://www.soundjay.com/buttons/beep-01a.mp3");
  audio.play();
}