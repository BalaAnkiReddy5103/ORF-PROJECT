async function renderCharts() {
  const barEl = document.getElementById("barChart");
  const lineEl = document.getElementById("lineChart");
  if (!barEl && !lineEl) {
    return;
  }

  try {
    const response = await fetch("/api/performance-data");
    if (!response.ok) {
      return;
    }
    const payload = await response.json();

    if (barEl) {
      new Chart(barEl.getContext("2d"), {
        type: "bar",
        data: {
          labels: payload.labels,
          datasets: [
            { label: "Accuracy", data: payload.accuracy, backgroundColor: "rgba(0,194,168,0.7)" },
            { label: "F1", data: payload.f1, backgroundColor: "rgba(255,107,53,0.7)" }
          ]
        },
        options: {
          responsive: true,
          plugins: { legend: { labels: { color: "#eaf4fa" } } },
          scales: {
            x: { ticks: { color: "#d9e8ef" }, grid: { color: "rgba(255,255,255,0.08)" } },
            y: { ticks: { color: "#d9e8ef" }, grid: { color: "rgba(255,255,255,0.08)" }, min: 0, max: 1 }
          }
        }
      });
    }

    if (lineEl) {
      new Chart(lineEl.getContext("2d"), {
        type: "line",
        data: {
          labels: payload.timeline,
          datasets: [
            { label: "Precision", data: payload.precision, borderColor: "#00c2a8", backgroundColor: "rgba(0,194,168,0.15)", tension: 0.3 },
            { label: "Recall", data: payload.recall, borderColor: "#ffc857", backgroundColor: "rgba(255,200,87,0.15)", tension: 0.3 }
          ]
        },
        options: {
          responsive: true,
          plugins: { legend: { labels: { color: "#eaf4fa" } } },
          scales: {
            x: { ticks: { color: "#d9e8ef", maxRotation: 45, minRotation: 35 }, grid: { color: "rgba(255,255,255,0.08)" } },
            y: { ticks: { color: "#d9e8ef" }, grid: { color: "rgba(255,255,255,0.08)" }, min: 0, max: 1 }
          }
        }
      });
    }
  } catch (_err) {
    // Keep UI functional if chart API is not available.
  }
}

document.addEventListener("DOMContentLoaded", renderCharts);
