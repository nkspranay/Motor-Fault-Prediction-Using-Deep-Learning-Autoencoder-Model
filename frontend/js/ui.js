const UNITS = {
  Voltage    : "V",
  Current    : "A",
  Power      : "W",
  Temperature: "°C",
  Humidity   : "%",
  Vibration  : "",
};

// ── Connection indicator ──
export function setConnected(connected) {
  const el = document.getElementById("connection-status");
  if (!el) return;
  el.textContent    = connected ? "● Connected" : "○ Disconnected";
  el.style.color    = connected ? "#16a34a" : "#ef4444";
}

// ── Status banner ──
export function updateStatus(status, prediction) {
  const el = document.getElementById("status");
  if (!el) return;

  const map = {
    Normal : { color: "#16a34a", bg: "#f0fdf4", label: "Normal" },
    Warning: { color: "#d97706", bg: "#fffbeb", label: "Warning" },
    Fault  : { color: "#dc2626", bg: "#fef2f2", label: "Fault"  },
  };
  const s = map[status] || map.Normal;

  el.style.background = s.bg;
  el.style.borderLeft = `4px solid ${s.color}`;
  el.style.padding    = "12px 16px";
  el.style.borderRadius = "8px";

  let html = `<span style="color:${s.color};font-weight:700;font-size:1.1rem">${s.label}</span>`;

  if (prediction && status === "Normal") {
    html += `<span style="color:#7c3aed;font-size:0.85rem;margin-left:12px">
               ⚑ Pattern detected — possible fault forming
             </span>`;
  }
  el.innerHTML = html;
}

// ── Metric cards ──
export function updateMetrics(values) {
  const el = document.getElementById("metrics");
  if (!el || !values) return;
  el.innerHTML = "";

  for (const [key, val] of Object.entries(values)) {
    el.innerHTML += `
      <div class="metric-card" onclick="window.setMetric('${key}')" title="Click to view chart">
        <div class="metric-label">${key}</div>
        <div class="metric-value">${Number(val).toFixed(2)}<span class="metric-unit"> ${UNITS[key] || ""}</span></div>
      </div>`;
  }
}

// ── Event history ──
export function updateHistory(history) {
  const el = document.getElementById("history-list");
  if (!el || !history) return;
  el.innerHTML = "";

  if (history.length === 0) {
    el.innerHTML = `<div style="color:#94a3b8;text-align:center;padding:20px">No events recorded</div>`;
    return;
  }

  [...history].reverse().forEach(item => {
    const statusColor = {
      Fault  : "#dc2626",
      Warning: "#d97706",
      Normal : "#16a34a",
    }[item.status] || "#6366f1";

    const faultText = (item.faults || []).length > 0
  ? `<div style="color:${statusColor};font-size:0.85rem">
       ⚠ ${(item.faults || [])
            .map(f => `${f.feature} (${f.type})`)
            .join(", ")}
     </div>`
  : "";

    const predText = item.prediction
      ? `<div style="color:#7c3aed;font-size:0.8rem">⚑ Prediction triggered</div>`
      : "";

    el.innerHTML += `
      <div class="history-item">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="color:${statusColor};font-weight:600">${item.status}</span>
          <span style="color:#94a3b8;font-size:0.78rem">${item.time}</span>
        </div>
        ${faultText}
        ${predText}
      </div>`;
  });
}

// ── Toast notification system ──
let toastContainer = null;

function getToastContainer() {
  if (!toastContainer) {
    toastContainer = document.createElement("div");
    toastContainer.id = "toast-container";
    toastContainer.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      z-index: 9999;
      display: flex;
      flex-direction: column;
      gap: 10px;
      max-width: 320px;
    `;
    document.body.appendChild(toastContainer);
  }
  return toastContainer;
}

export function showToast(type, title, message) {
  const container = getToastContainer();

  const colors = {
    fault  : { bg: "#fef2f2", border: "#dc2626", icon: "🔴" },
    warning: { bg: "#fffbeb", border: "#d97706", icon: "🟡" },
    normal : { bg: "#f0fdf4", border: "#16a34a", icon: "🟢" },
    predict: { bg: "#f5f3ff", border: "#7c3aed", icon: "⚑" },
  };
  const c = colors[type] || colors.normal;

  const toast = document.createElement("div");
  toast.style.cssText = `
    background: ${c.bg};
    border-left: 4px solid ${c.border};
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    animation: slideIn 0.3s ease;
    cursor: pointer;
  `;
  toast.innerHTML = `
    <div style="font-weight:700;margin-bottom:4px">${c.icon} ${title}</div>
    <div style="font-size:0.875rem;color:#374151">${message}</div>
  `;

  toast.onclick = () => toast.remove();
  container.appendChild(toast);

  // Auto-dismiss
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.4s";
    setTimeout(() => toast.remove(), 400);
  }, 6000);
}