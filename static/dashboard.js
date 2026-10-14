/**
 * RetailVision — Executive Dashboard Client Controller
 * Manages live telemetry, Chart.js visualizations, video ingestion, and AI analyst inquiries.
 */

let trafficChartInstance = null;
let segmentsChartInstance = null;
let currentTrafficData = [];
let currentForecastData = null;

document.addEventListener("DOMContentLoaded", () => {
    initDropzone();
    initVideoUpload();
    initAnalystForm();
    initPromptChips();
    initRefreshButton();
    loadAllDashboardData();
});

/* ============================================================
   1. Data Loaders & Telemetry
   ============================================================ */

async function loadAllDashboardData() {
    updateLastUpdatedText();
    await Promise.allSettled([
        loadOverviewMetrics(),
        loadTrafficAndForecast(),
        loadSegmentsData()
    ]);
}

function updateLastUpdatedText() {
    const el = document.getElementById("last-updated-text");
    if (el) {
        const now = new Date();
        el.textContent = `Live Telemetry • Updated ${now.toLocaleTimeString()}`;
    }
}

async function loadOverviewMetrics() {
    try {
        const res = await fetch("/overview/");
        if (!res.ok) return;
        const data = await res.json();

        // Update KPI values
        document.getElementById("val-total-shoppers").textContent = data.total_tracks || "0";
        document.getElementById("badge-video-count").textContent = `${data.video_count || 0} video${data.video_count === 1 ? "" : "s"}`;
        document.getElementById("val-avg-dwell").textContent = `${data.avg_dwell_sec || 0}s`;
        document.getElementById("val-engaged-ratio").textContent = `${data.engaged_pct || 0}%`;
        document.getElementById("val-traffic-count").textContent = data.avg_occupancy ?? "--";
        document.getElementById("badge-peak-occupancy").textContent = `peak ${data.peak_occupancy ?? "--"}`;
        document.getElementById("val-crossings").textContent =
            data.video_count ? `${data.crossings_in || 0} / ${data.crossings_out || 0}` : "--";

        // Update 3x3 Zone Matrix
        updateZoneGrid(data.zones || {});
    } catch (err) {
        console.warn("Could not load overview metrics:", err);
    }
}

function updateZoneGrid(zones) {
    const cells = document.querySelectorAll(".zone-cell");
    let maxCount = 1;
    Object.values(zones).forEach(c => {
        if (c > maxCount) maxCount = c;
    });

    cells.forEach(cell => {
        const zId = cell.getAttribute("data-zone");
        const count = zones[zId] || 0;
        const countEl = cell.querySelector(".zone-count");
        if (countEl) countEl.textContent = count;

        // Dynamic heat intensity
        const intensity = Math.min(count / maxCount, 1);
        if (count > 0) {
            cell.style.backgroundColor = `rgba(99, 102, 241, ${0.1 + intensity * 0.45})`;
            cell.style.borderColor = `rgba(99, 102, 241, ${0.25 + intensity * 0.5})`;
        } else {
            cell.style.backgroundColor = "rgba(255, 255, 255, 0.02)";
            cell.style.borderColor = "rgba(255, 255, 255, 0.08)";
        }
    });
}

async function loadTrafficAndForecast() {
    try {
        const [trafficRes, forecastRes] = await Promise.all([
            fetch("/customer_traffic/"),
            fetch("/forecast/?hours=6")
        ]);

        if (trafficRes.ok) {
            const trafficJson = await trafficRes.json();
            currentTrafficData = (trafficJson.customer_traffic || []).slice().reverse(); // Chronological
        }

        if (forecastRes.ok) {
            currentForecastData = await forecastRes.json();
            const badge = document.getElementById("forecast-status-badge");
            const meta = document.getElementById("forecast-meta-text");
            if (badge && currentForecastData.status) {
                if (currentForecastData.status === "lstm_trained") {
                    badge.textContent = "Deep LSTM Active";
                    badge.style.background = "rgba(16, 185, 129, 0.15)";
                    badge.style.color = "#34d399";
                    if (meta) meta.textContent = `Trained model loss: ${currentForecastData.train_loss}`;
                } else {
                    badge.textContent = "Trend Extrapolation";
                    badge.style.background = "rgba(245, 158, 11, 0.15)";
                    badge.style.color = "#fbbf24";
                    if (meta) meta.textContent = "Awaiting 9+ hours for full LSTM curve";
                }
            }
        }

        renderTrafficChart();
    } catch (err) {
        console.warn("Error fetching traffic/forecast:", err);
    }
}

async function loadSegmentsData() {
    try {
        const res = await fetch("/segments/");
        if (!res.ok) return;
        const rows = await res.json();

        let engagedCount = 0;
        let browserCount = 0;
        let passerCount = 0;

        rows.forEach(r => {
            const seg = (r.segment || "").toLowerCase();
            const n = r.n || 0;
            if (seg.includes("engaged")) engagedCount += n;
            else if (seg.includes("browser")) browserCount += n;
            else if (seg.includes("passer")) passerCount += n;
        });

        document.getElementById("count-engaged").textContent = engagedCount;
        document.getElementById("count-browser").textContent = browserCount;
        document.getElementById("count-passer").textContent = passerCount;

        renderSegmentsDonut(engagedCount, browserCount, passerCount);
    } catch (err) {
        console.warn("Error fetching segments:", err);
    }
}

/* ============================================================
   2. Chart.js Visualizations
   ============================================================ */

function renderTrafficChart() {
    const ctx = document.getElementById("trafficChart")?.getContext("2d");
    if (!ctx) return;

    // Build timeline points
    const histLabels = currentTrafficData.slice(-20).map(item => {
        const parts = (item.timestamp || "").split(" ");
        return parts.length > 1 ? parts[1].substring(0, 5) : item.timestamp;
    });
    const histValues = currentTrafficData.slice(-20).map(item => item.people_count);

    // Build forecast points
    const forecastLabels = (currentForecastData && currentForecastData.future_labels) 
        ? currentForecastData.future_labels 
        : ["+1h", "+2h", "+3h", "+4h", "+5h", "+6h"];
    const forecastValues = (currentForecastData && currentForecastData.forecast_next_hours)
        ? currentForecastData.forecast_next_hours
        : [];

    const allLabels = [...histLabels, ...forecastLabels];
    const historicalSeries = [...histValues, ...new Array(forecastValues.length).fill(null)];
    
    // Connect historical end to forecast start
    const forecastSeries = new Array(Math.max(histValues.length - 1, 0)).fill(null);
    if (histValues.length > 0) {
        forecastSeries.push(histValues[histValues.length - 1]);
    }
    forecastSeries.push(...forecastValues);

    if (trafficChartInstance) {
        trafficChartInstance.destroy();
    }

    trafficChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: allLabels.length > 0 ? allLabels : ["No data yet"],
            datasets: [
                {
                    label: "Observed Store Traffic",
                    data: historicalSeries,
                    borderColor: "#6366f1",
                    backgroundColor: "rgba(99, 102, 241, 0.12)",
                    borderWidth: 2.5,
                    fill: true,
                    tension: 0.35,
                    pointBackgroundColor: "#818cf8",
                    pointRadius: 3
                },
                {
                    label: "LSTM Predicted Footfall",
                    data: forecastSeries,
                    borderColor: "#34d399",
                    backgroundColor: "rgba(16, 185, 129, 0.08)",
                    borderWidth: 2.5,
                    borderDash: [5, 5],
                    fill: true,
                    tension: 0.35,
                    pointBackgroundColor: "#34d399",
                    pointRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        color: "#94a3b8",
                        font: { family: "'Plus Jakarta Sans', sans-serif", size: 12 }
                    }
                },
                tooltip: {
                    backgroundColor: "rgba(15, 20, 32, 0.95)",
                    titleColor: "#f8fafc",
                    bodyColor: "#cbd5e1",
                    borderColor: "rgba(255, 255, 255, 0.1)",
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#64748b", font: { family: "'JetBrains Mono', monospace", size: 10 } }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: "rgba(255, 255, 255, 0.05)" },
                    ticks: { color: "#64748b", font: { family: "'JetBrains Mono', monospace", size: 10 } }
                }
            }
        }
    });
}

function renderSegmentsDonut(engaged, browser, passer) {
    const ctx = document.getElementById("segmentsDonutChart")?.getContext("2d");
    if (!ctx) return;

    if (segmentsChartInstance) {
        segmentsChartInstance.destroy();
    }

    const hasData = (engaged + browser + passer) > 0;
    const dataValues = hasData ? [engaged, browser, passer] : [1, 1, 1];
    const colors = hasData 
        ? ["#a855f7", "#3b82f6", "#f59e0b"] 
        : ["rgba(255, 255, 255, 0.1)", "rgba(255, 255, 255, 0.08)", "rgba(255, 255, 255, 0.05)"];

    segmentsChartInstance = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Engaged Shopper", "Browser", "Passer-By"],
            datasets: [{
                data: dataValues,
                backgroundColor: colors,
                borderColor: "rgba(18, 24, 38, 0.8)",
                borderWidth: 3,
                hoverOffset: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: "rgba(15, 20, 32, 0.95)",
                    callbacks: {
                        label: function(context) {
                            return hasData 
                                ? ` ${context.label}: ${context.raw} persons`
                                : ` No tracking records yet`;
                        }
                    }
                }
            },
            cutout: "70%"
        }
    });
}

/* ============================================================
   3. Drag & Drop and Video Ingestion
   ============================================================ */

function initDropzone() {
    const dropzone = document.getElementById("video-dropzone");
    const fileInput = document.getElementById("video-file-input");
    const banner = document.getElementById("selected-file-banner");
    const fileNameSpan = document.getElementById("selected-file-name");
    const clearBtn = document.getElementById("clear-file-btn");

    if (!dropzone || !fileInput) return;

    ["dragenter", "dragover"].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelection(fileInput.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleFileSelection(fileInput.files[0]);
        }
    });

    clearBtn?.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.value = "";
        banner.style.display = "none";
    });

    function handleFileSelection(file) {
        if (fileNameSpan && banner) {
            fileNameSpan.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(1)} MB)`;
            banner.style.display = "flex";
        }
    }
}

function initVideoUpload() {
    const form = document.getElementById("video-upload-form");
    const progressPanel = document.getElementById("analysis-progress-panel");
    const progressText = document.getElementById("progress-status-text");
    const progressPct = document.getElementById("progress-pct-text");
    const summaryCard = document.getElementById("analysis-summary-card");
    const submitBtn = document.getElementById("start-analysis-btn");
    const useClipToggle = document.getElementById("toggle-use-clip");

    if (!form) return;

    form.onsubmit = async (e) => {
        e.preventDefault();
        const fileInput = document.getElementById("video-file-input");
        if (!fileInput.files || fileInput.files.length === 0) {
            alert("Please select or drop a video file.");
            return;
        }

        const file = fileInput.files[0];
        const useClip = useClipToggle ? useClipToggle.checked : true;

        // UI State: Running
        if (submitBtn) submitBtn.disabled = true;
        if (progressPanel) progressPanel.style.display = "block";
        if (summaryCard) summaryCard.style.display = "none";
        if (progressText) progressText.textContent = `Ingesting ${file.name}...`;
        if (progressPct) progressPct.textContent = "Processing";

        // Step simulation indicators
        animatePipelineBadges();

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch(`/analyze_video/?use_clip=${useClip}&background=true&camera=${encodeURIComponent((document.getElementById("camera-name") || {}).value || "default")}`, {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(errText || "Analysis failed.");
            }

            const { job_id } = await response.json();
            let result = null;
            while (result === null) {
                await new Promise(r => setTimeout(r, 1500));
                const jr = await fetch(`/jobs/${job_id}`);
                if (!jr.ok) throw new Error("Lost track of the analysis job.");
                const job = await jr.json();
                if (progressText) progressText.textContent = job.status === "queued" ? "Queued..." : "Analyzing...";
                if (job.status === "error") throw new Error(job.error || "Analysis failed.");
                if (job.status === "done") result = job.result;
            }

            // Populate Summary Card
            document.getElementById("summary-unique-people").textContent = result.unique_people || result.people_count || 0;
            document.getElementById("summary-duration").textContent = `${result.duration_sec || 0}s`;
            const occ = result.occupancy || {};
            document.getElementById("summary-occupancy").textContent = `${occ.mean ?? "--"} / ${occ.peak ?? "--"}`;
            const cross = Object.values(result.crossings || {});
            const cin = cross.reduce((a, c) => a + c.in, 0), cout = cross.reduce((a, c) => a + c.out, 0);
            document.getElementById("summary-crossings").textContent = cross.length ? `${cin} / ${cout}` : "No line set";
            document.getElementById("summary-dwell").textContent = `${result.avg_dwell_sec || 0}s`;

            const pillList = document.getElementById("summary-segment-pills");
            if (pillList && result.segments) {
                pillList.innerHTML = "";
                Object.entries(result.segments).forEach(([name, count]) => {
                    const pill = document.createElement("span");
                    pill.className = "segment-pill";
                    pill.textContent = `${name}: ${count}`;
                    pillList.appendChild(pill);
                });
            }

            if (summaryCard) summaryCard.style.display = "block";
            if (progressText) progressText.textContent = "Complete!";
            if (progressPct) progressPct.textContent = "100%";

            // Refresh KPIs & charts
            await loadAllDashboardData();
        } catch (err) {
            alert(`Analysis Error: ${err.message}`);
            if (progressText) progressText.textContent = "Error occurred.";
        } finally {
            if (submitBtn) submitBtn.disabled = false;
        }
    };
}

function animatePipelineBadges() {
    const badges = ["step-yolo", "step-clip", "step-kmeans", "step-persist"];
    badges.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.remove("active");
    });

    let current = 0;
    const interval = setInterval(() => {
        if (current > 0) {
            const prev = document.getElementById(badges[current - 1]);
            if (prev) prev.classList.remove("active");
        }
        if (current < badges.length) {
            const next = document.getElementById(badges[current]);
            if (next) next.classList.add("active");
            current++;
        } else {
            clearInterval(interval);
        }
    }, 1500);
}

/* ============================================================
   4. Conversational AI Retail Analyst Terminal
   ============================================================ */

function initAnalystForm() {
    const form = document.getElementById("analyst-query-form");
    const input = document.getElementById("analyst-query-input");
    const chatDisplay = document.getElementById("analyst-chat-display");
    const submitBtn = document.getElementById("analyst-submit-btn");

    if (!form || !input || !chatDisplay) return;

    form.onsubmit = async (e) => {
        e.preventDefault();
        const query = input.value.trim();
        if (!query) return;

        appendChatMessage("user", query);
        input.value = "";
        if (submitBtn) submitBtn.disabled = true;

        // Show thinking indicator
        const thinkingId = appendChatMessage("assistant", "Analyzing telemetry and generating SQL...");

        try {
            const response = await fetch(`/ask/?q=${encodeURIComponent(query)}`);
            if (!response.ok) throw new Error("Could not reach analyst service.");
            const data = await response.json();

            removeChatMessage(thinkingId);

            let contentHtml = `<p>${escapeHtml(data.answer || "No response received.")}</p>`;
            if (data.sql && data.executed === false) {
                // the query was generated but did not run: say so, never "Executed"
                contentHtml += `<span class="sql-badge sql-failed">Not executed. Query was: ${escapeHtml(data.sql)}</span>`;
            } else if (data.sql) {
                const retried = data.attempts > 1 ? ` (corrected after ${data.attempts - 1} retry)` : "";
                contentHtml += `<span class="sql-badge">Executed SQL${retried}: ${escapeHtml(data.sql)}</span>`;
                if (Array.isArray(data.rows)) {
                    const shown = data.rows.slice(0, 5).map(r =>
                        Object.entries(r).map(([k, v]) => `${k}=${v}`).join(", ")).join(" | ");
                    const more = data.truncated || data.rows.length > 5 ? " ..." : "";
                    contentHtml += `<span class="sql-badge sql-result">Result (${data.rows.length} row${data.rows.length === 1 ? "" : "s"}): ${escapeHtml(shown || "no rows")}${more}</span>`;
                }
            }
            if (data.data && data.data.forecast_next_hours) {
                contentHtml += `<span class="sql-badge">Next Forecast: ${data.data.forecast_next_hours.join(", ")} visitors</span>`;
            }

            appendChatMessage("assistant", contentHtml, true);
        } catch (err) {
            removeChatMessage(thinkingId);
            appendChatMessage("assistant", `<p style="color: #ef4444;">Error: ${escapeHtml(err.message)}</p>`, true);
        } finally {
            if (submitBtn) submitBtn.disabled = false;
        }
    };
}

function initPromptChips() {
    const chips = document.querySelectorAll(".prompt-chip");
    const input = document.getElementById("analyst-query-input");
    const form = document.getElementById("analyst-query-form");

    chips.forEach(chip => {
        chip.addEventListener("click", () => {
            const query = chip.getAttribute("data-query");
            if (input && query) {
                input.value = query;
                form?.dispatchEvent(new Event("submit", { cancelable: true }));
            }
        });
    });
}

function initRefreshButton() {
    const btn = document.getElementById("refresh-dashboard-btn");
    btn?.addEventListener("click", async () => {
        btn.classList.add("loading");
        await loadAllDashboardData();
        btn.classList.remove("loading");
    });
}

/* ============================================================
   Utilities
   ============================================================ */

function appendChatMessage(sender, content, isHtml = false) {
    const chatDisplay = document.getElementById("analyst-chat-display");
    if (!chatDisplay) return null;

    const msgId = `msg-${Date.now()}-${Math.random()}`;
    const msg = document.createElement("div");
    msg.id = msgId;
    msg.className = `chat-message ${sender}-message`;

    const avatar = sender === "user" ? "👤" : "🤖";
    const bodyContent = isHtml ? content : `<p>${escapeHtml(content)}</p>`;

    msg.innerHTML = `
        <div class="msg-avatar">${avatar}</div>
        <div class="msg-content">${bodyContent}</div>
    `;

    chatDisplay.appendChild(msg);
    chatDisplay.scrollTop = chatDisplay.scrollHeight;
    return msgId;
}

function removeChatMessage(id) {
    if (!id) return;
    const el = document.getElementById(id);
    if (el) el.remove();
}

function escapeHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}
