const alarmBanner = document.getElementById("alarm-banner");
const ackBtn = document.getElementById("ack-btn");

function fmtPercent(v) {
  return v === null || v === undefined ? "--" : `${v.toFixed(1)}%`;
}

function applyStatus(status) {
  document.getElementById("cpu").textContent = fmtPercent(status.cpu_percent);
  document.getElementById("ram").textContent =
    `${fmtPercent(status.ram_percent)} (${Math.round(status.ram_used_mb || 0)} MB)`;
  document.getElementById("temp").textContent =
    status.temperature_c != null ? `${status.temperature_c.toFixed(1)} C` : "--";
  document.getElementById("fps").textContent = (status.fps || 0).toFixed(1);
  document.getElementById("alarm-state").textContent = status.alarm_active ? "ACTIVE" : "Normal";

  alarmBanner.classList.toggle("hidden", !status.alarm_active);

  const list = document.getElementById("detections");
  list.innerHTML = "";
  (status.active_detections || []).forEach((d) => {
    const li = document.createElement("li");
    li.textContent = `${d.class_name} — ${(d.confidence * 100).toFixed(0)}%`;
    list.appendChild(li);
  });
}

function connectStatusSocket() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/status`);
  ws.onmessage = (msg) => applyStatus(JSON.parse(msg.data));
  ws.onclose = () => setTimeout(connectStatusSocket, 2000);
  ws.onerror = () => ws.close();
}

async function refreshEvents() {
  const res = await fetch("/api/events?limit=50");
  const events = await res.json();
  const tbody = document.querySelector("#events-table tbody");
  tbody.innerHTML = "";
  events.forEach((e) => {
    const tr = document.createElement("tr");
    const time = new Date(e.timestamp * 1000).toLocaleString();
    const media = [
      e.snapshot_path ? `<a href="/snapshots/${e.snapshot_path.split("/").pop()}" target="_blank">image</a>` : "",
      e.clip_path ? `<a href="/clips/${e.clip_path.split("/").pop()}" target="_blank">clip</a>` : "",
    ].filter(Boolean).join(" / ");
    tr.innerHTML = `<td>${time}</td><td>${e.class_name}</td><td>${e.severity}</td>` +
      `<td>${(e.confidence * 100).toFixed(0)}%</td><td>${e.zones || ""}</td><td>${media}</td>`;
    tbody.appendChild(tr);
  });
}

ackBtn.addEventListener("click", async () => {
  await fetch("/api/alarm/acknowledge", { method: "POST" });
  refreshEvents();
});

connectStatusSocket();
refreshEvents();
setInterval(refreshEvents, 10000);
