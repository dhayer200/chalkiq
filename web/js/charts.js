// ChalkIQ — Charts & Rankings
(function () {
  const DATA_URL = "/assets/data.json";

  fetch(DATA_URL)
    .then((r) => r.json())
    .then((data) => {
      renderCLVChart(data.cumulative_clv);
      renderRankings(data.rankings);
    })
    .catch((err) => console.error("Failed to load data:", err));

  function renderCLVChart(points) {
    const ctx = document.getElementById("clvChart");
    if (!ctx) return;

    const labels = points.map((p) => p.date);
    const values = points.map((p) => p.cum_clv);

    // Find where CLV crosses zero for gradient split
    const maxVal = Math.max(...values);
    const minVal = Math.min(...values);

    new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "Cumulative CLV (%)",
            data: values,
            borderColor: "#22c55e",
            backgroundColor: (context) => {
              const chart = context.chart;
              const { ctx: c, chartArea } = chart;
              if (!chartArea) return "transparent";
              const gradient = c.createLinearGradient(
                0,
                chartArea.top,
                0,
                chartArea.bottom
              );
              gradient.addColorStop(0, "rgba(34, 197, 94, 0.15)");
              gradient.addColorStop(1, "rgba(34, 197, 94, 0)");
              return gradient;
            },
            fill: true,
            tension: 0.3,
            borderWidth: 2,
            pointRadius: 0,
            pointHoverRadius: 4,
            pointHoverBackgroundColor: "#22c55e",
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: true,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: "#1e293b",
            borderColor: "#334155",
            borderWidth: 1,
            titleColor: "#94a3b8",
            bodyColor: "#f1f5f9",
            bodyFont: { family: "JetBrains Mono", size: 13 },
            padding: 12,
            callbacks: {
              label: (ctx) => `CLV: ${ctx.parsed.y > 0 ? "+" : ""}${ctx.parsed.y.toFixed(1)}%`,
            },
          },
        },
        scales: {
          x: {
            grid: { color: "rgba(148,163,184,0.06)" },
            ticks: {
              color: "#64748b",
              font: { size: 11 },
              maxTicksLimit: 8,
              callback: function (val, i) {
                const d = this.getLabelForValue(val);
                return d.slice(5); // MM-DD
              },
            },
          },
          y: {
            grid: { color: "rgba(148,163,184,0.06)" },
            ticks: {
              color: "#64748b",
              font: { family: "JetBrains Mono", size: 11 },
              callback: (v) => (v > 0 ? "+" : "") + v.toFixed(0) + "%",
            },
          },
        },
      },
    });
  }

  function renderRankings(rankings) {
    const tbody = document.getElementById("rankingsBody");
    if (!tbody) return;

    tbody.innerHTML = rankings
      .map((t, i) => {
        const diff = t.elo - 1500;
        const diffStr = (diff > 0 ? "+" : "") + diff;
        const diffColor = diff > 200 ? "text-emerald-400" : diff > 100 ? "text-emerald-500/70" : "text-slate-500";
        const bg = i % 2 === 0 ? "" : "bg-slate-800/20";
        const border = i < rankings.length - 1 ? "border-b border-slate-800/30" : "";
        return `
        <tr class="${bg} ${border} hover:bg-slate-800/40 transition-colors">
          <td class="px-6 py-3 text-slate-500 font-medium">${t.rank}</td>
          <td class="px-6 py-3 text-white font-medium" style="font-family:Inter,system-ui">${t.name}</td>
          <td class="px-6 py-3 text-right font-semibold">${t.elo}</td>
          <td class="px-6 py-3 text-right ${diffColor} hidden sm:table-cell">${diffStr}</td>
        </tr>`;
      })
      .join("");
  }
})();
