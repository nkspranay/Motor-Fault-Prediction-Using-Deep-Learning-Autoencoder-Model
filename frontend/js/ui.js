const UNITS = {
  Voltage: "V",
  Current: "A",
  Power: "W",
  Temperature: "°C",
  Vibration: "g"
};

export function updateStatus(status) {
  const el = document.getElementById("status");

  el.innerHTML =
    status === "Normal"
      ? "<span style='color:green'>Normal</span>"
      : status === "Warning"
      ? "<span style='color:orange'>Warning</span>"
      : "<span style='color:red'>Fault</span>";
}

export function updateMetrics(values) {
  const el = document.getElementById("metrics");
  el.innerHTML = "";

  for (let key in values) {
    el.innerHTML += `
      <div class="bg-white p-4 rounded-xl shadow text-center">
        <div class="text-sm text-gray-500">${key}</div>
        <div class="text-xl font-bold">
          ${values[key].toFixed(2)} ${UNITS[key] || ""}
        </div>
      </div>
    `;
  }
}

export function updateHistory(history) {
  const el = document.getElementById("history-list");
  el.innerHTML = "";

  history.slice().reverse().forEach(item => {
    el.innerHTML += `
      <div class="bg-white p-3 rounded shadow">
        <div class="text-sm text-gray-500">${item.time}</div>
        <div style="color:red">Faults: ${item.faults.join(", ")}</div>
      </div>
    `;
  });
}