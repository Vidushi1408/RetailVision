/* Camera names, camera streams and privacy controls. */
(function () {
    const $ = id => document.getElementById(id);

    async function refreshCameras() {
        try {
            const lines = (await (await fetch("/lines/")).json()).lines || [];
            const cams = new Set(["default"]);
            lines.forEach(l => cams.add(l.camera || "default"));
            const dl = $("camera-list");
            if (dl) { dl.innerHTML = ""; cams.forEach(c => { const o = document.createElement("option"); o.value = c; dl.appendChild(o); }); }
        } catch (e) { /* datalist is a convenience only */ }
    }

    // ---- streams -----------------------------------------------------------
    async function initStreams() {
        const card = $("streams-card");
        if (!card) return;
        let names = [];
        try { names = (await (await fetch("/streams/")).json()).streams || []; } catch (e) {}
        if (!names.length) return;                    // nothing configured: keep the card hidden
        card.style.display = "";
        const sel = $("stream-name");
        names.forEach(n => { const o = document.createElement("option"); o.value = o.textContent = n; sel.appendChild(o); });
        $("stream-run").addEventListener("click", async () => {
            const status = $("stream-status"), btn = $("stream-run");
            btn.disabled = true; status.classList.remove("lines-error");
            const qs = new URLSearchParams({ name: sel.value, seconds: $("stream-seconds").value,
                                             camera: ($("camera-name") || {}).value || "default" });
            try {
                const res = await fetch("/analyze_stream/?" + qs, { method: "POST" });
                if (!res.ok) throw new Error((await res.json()).detail || "Could not start.");
                const { job_id } = await res.json();
                status.textContent = "Recording and analysing...";
                for (;;) {
                    await new Promise(r => setTimeout(r, 2000));
                    const job = await (await fetch("/jobs/" + job_id)).json();
                    if (job.status === "error") throw new Error(job.error || "Analysis failed.");
                    if (job.status === "done") {
                        const r = job.result;
                        status.textContent = `Done: ${r.unique_people} people (est.), occupancy ${r.occupancy.mean}/${r.occupancy.peak}.`;
                        if (typeof loadAllDashboardData === "function") loadAllDashboardData();
                        break;
                    }
                }
            } catch (err) {
                status.textContent = err.message; status.classList.add("lines-error");
            } finally { btn.disabled = false; }
        });
    }

    // ---- privacy -------------------------------------------------------------
    async function initPrivacy() {
        const summary = $("privacy-summary"), status = $("privacy-status");
        if (!summary) return;
        try {
            const p = await (await fetch("/privacy/")).json();
            summary.textContent = "Stored: " + p.stores.join("; ") + ". Never stored: " + p.does_not_store.join("; ") +
                (p.retention_days ? `. Automatic deletion after ${p.retention_days} days.` : ". No automatic deletion is configured.");
        } catch (e) { summary.textContent = "Privacy information is unavailable."; }

        async function act(url, method, okText) {
            status.classList.remove("lines-error");
            const res = await fetch(url, { method });
            if (!res.ok) { status.textContent = "Failed: " + ((await res.json()).detail || res.status); status.classList.add("lines-error"); return; }
            const d = (await res.json()).deleted;
            status.textContent = okText + " (" + Object.entries(d).map(([k, v]) => `${k}: ${v}`).join(", ") + ")";
            if (typeof loadAllDashboardData === "function") loadAllDashboardData();
        }
        $("purge-run").addEventListener("click", () => {
            const days = parseInt($("purge-days").value, 10);
            if (!(days >= 0)) { status.textContent = "Enter a number of days."; return; }
            if (window.confirm(`Delete all analytics older than ${days} days?`)) act("/privacy/purge?days=" + days, "POST", "Deleted");
        });
        $("wipe-run").addEventListener("click", () => {
            if (window.confirm("Delete ALL analytics data? Counting lines are kept. This cannot be undone.")) {
                act("/privacy/data?confirm=yes", "DELETE", "All analytics deleted");
            }
        });
    }

    document.addEventListener("DOMContentLoaded", () => { refreshCameras(); initStreams(); initPrivacy(); });
    window.addEventListener("lines-changed", refreshCameras);
})();
