/* Counting-line setup: draw a segment on a frame and save it via /lines/.
   Coordinates are stored as fractions of the frame (0-1, y down). "in" is the right-hand side
   of the segment when walking from point A to point B (matches retailvision.tracker). */
(function () {
    const canvas = document.getElementById("lines-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const statusEl = document.getElementById("lines-status");
    const listEl = document.getElementById("saved-lines");
    const nameEl = document.getElementById("line-name");
    const MAX_W = 640;

    let frame = null;          // ImageBitmap | HTMLVideoElement-drawn canvas
    let points = [];           // up to two {x, y} in 0-1 coordinates
    let hover = null;
    let saved = [];

    function say(msg, isError) {
        statusEl.textContent = msg;
        statusEl.classList.toggle("lines-error", !!isError);
    }

    function resizeTo(w, h) {
        const scale = Math.min(1, MAX_W / w);
        canvas.width = Math.round(w * scale);
        canvas.height = Math.round(h * scale);
    }

    function arrow(ax, ay, bx, by, color, width) {
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
        const ang = Math.atan2(by - ay, bx - ax), s = 10;
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(bx - s * Math.cos(ang - 0.45), by - s * Math.sin(ang - 0.45));
        ctx.lineTo(bx - s * Math.cos(ang + 0.45), by - s * Math.sin(ang + 0.45));
        ctx.closePath(); ctx.fill();
    }

    function drawLine(p, q, color, label) {
        const W = canvas.width, H = canvas.height;
        const ax = p.x * W, ay = p.y * H, bx = q.x * W, by = q.y * H;
        ctx.strokeStyle = color; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
        [[ax, ay], [bx, by]].forEach(([x, y]) => {
            ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, 5, 0, 6.2832); ctx.fill();
        });
        // "in" arrow: normal pointing to the right-hand side of A->B (image coords, y down)
        const ex = bx - ax, ey = by - ay, len = Math.hypot(ex, ey) || 1;
        const nx = -ey / len, ny = ex / len, mx = (ax + bx) / 2, my = (ay + by) / 2;
        arrow(mx, my, mx + nx * 34, my + ny * 34, color, 3);
        ctx.font = "bold 13px sans-serif"; ctx.fillStyle = "#fff";
        ctx.strokeStyle = "rgba(0,0,0,0.7)"; ctx.lineWidth = 3;
        const text = (label ? label + ": " : "") + "in";
        ctx.strokeText(text, mx + nx * 40 - 8, my + ny * 40 + 4);
        ctx.fillText(text, mx + nx * 40 - 8, my + ny * 40 + 4);
    }

    function render() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (frame) {
            ctx.drawImage(frame, 0, 0, canvas.width, canvas.height);
        } else {
            ctx.fillStyle = "#1b2433"; ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "#9fb0c8"; ctx.font = "16px sans-serif"; ctx.textAlign = "center";
            ctx.fillText("Load a frame, or use a preset line", canvas.width / 2, canvas.height / 2);
            ctx.textAlign = "start";
        }
        saved.forEach(l => drawLine({ x: l.x1, y: l.y1 }, { x: l.x2, y: l.y2 }, "#14b8a6", l.name));
        const a = points[0], b = points[1] || hover;
        if (a && b) drawLine(a, b, "#f59e0b", "new");
        else if (a) {
            ctx.fillStyle = "#f59e0b"; ctx.beginPath();
            ctx.arc(a.x * canvas.width, a.y * canvas.height, 5, 0, 6.2832); ctx.fill();
        }
    }

    function canvasPoint(ev) {
        const r = canvas.getBoundingClientRect();
        return {
            x: Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width)),
            y: Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height)),
        };
    }

    canvas.addEventListener("click", ev => {
        const p = canvasPoint(ev);
        points = points.length >= 2 ? [p] : points.concat([p]);
        hover = null;
        say(points.length === 1 ? "Click the second point." : "Line drawn. Name it and press Save.");
        render();
    });
    canvas.addEventListener("mousemove", ev => {
        if (points.length === 1) { hover = canvasPoint(ev); render(); }
    });

    // ---- frame sources ----------------------------------------------------
    async function tryBrowserVideo(file) {
        const url = URL.createObjectURL(file);
        try {
            const v = document.createElement("video");
            v.muted = true; v.preload = "auto"; v.src = url;
            await new Promise((res, rej) => {
                v.onloadeddata = res; v.onerror = () => rej(new Error("decode"));
                setTimeout(() => rej(new Error("timeout")), 8000);
            });
            v.currentTime = Math.min(1, (v.duration || 1) / 2);
            await new Promise((res, rej) => { v.onseeked = res; setTimeout(() => rej(new Error("seek")), 8000); });
            resizeTo(v.videoWidth, v.videoHeight);
            const off = document.createElement("canvas");
            off.width = canvas.width; off.height = canvas.height;
            off.getContext("2d").drawImage(v, 0, 0, off.width, off.height);
            frame = off;
            return true;
        } catch (e) {
            return false;
        } finally {
            URL.revokeObjectURL(url);
        }
    }

    async function loadFile(file) {
        if (!file) return;
        try {
            if (file.type.startsWith("image/")) {
                frame = await createImageBitmap(file);
                resizeTo(frame.width, frame.height);
            } else if (!(await tryBrowserVideo(file))) {
                // The browser cannot decode this codec; let the server (OpenCV) pull a frame.
                say("Reading a frame on the server...");
                const fd = new FormData(); fd.append("file", file);
                const res = await fetch("/frame/", { method: "POST", body: fd });
                if (!res.ok) throw new Error("Could not read a frame from this file.");
                frame = await createImageBitmap(await res.blob());
                resizeTo(frame.width, frame.height);
            }
            points = []; hover = null;
            say("Frame loaded. Click two points to draw the line.");
            render();
        } catch (err) {
            say(err.message || "Could not load that file.", true);
        }
    }

    document.getElementById("lines-frame-input").addEventListener("change", e => loadFile(e.target.files[0]));
    document.getElementById("lines-use-upload").addEventListener("click", () => {
        const f = document.getElementById("video-file-input").files[0];
        if (!f) return say("Select a clip in the upload box above first.", true);
        loadFile(f);
    });
    document.getElementById("lines-preset-h").addEventListener("click", () => {
        points = [{ x: 0.05, y: 0.5 }, { x: 0.95, y: 0.5 }]; say("Horizontal line set. Press Save."); render();
    });
    document.getElementById("lines-preset-v").addEventListener("click", () => {
        points = [{ x: 0.5, y: 0.05 }, { x: 0.5, y: 0.95 }]; say("Vertical line set. Press Save."); render();
    });
    document.getElementById("line-flip").addEventListener("click", () => {
        if (points.length === 2) { points = [points[1], points[0]]; render(); say("Direction flipped."); }
    });
    document.getElementById("line-clear").addEventListener("click", () => {
        points = []; hover = null; say("Cleared."); render();
    });

    // ---- persistence --------------------------------------------------------
    async function refresh() {
        try {
            const res = await fetch("/lines/");
            saved = (await res.json()).lines || [];
        } catch (e) { saved = []; }
        listEl.innerHTML = "";
        if (!saved.length) {
            const li = document.createElement("li");
            li.className = "saved-empty";
            li.textContent = "No counting lines saved: entrances are not counted until you draw one.";
            listEl.appendChild(li);
        }
        saved.forEach(l => {
            const li = document.createElement("li");
            const span = document.createElement("span");
            span.textContent = l.name + (l.camera && l.camera !== "default" ? "  (camera: " + l.camera + ")" : "");
            const del = document.createElement("button");
            del.type = "button"; del.className = "lines-btn"; del.textContent = "Delete";
            del.setAttribute("aria-label", "Delete line " + l.name);
            del.addEventListener("click", async () => {
                const r = await fetch("/lines/" + encodeURIComponent(l.name), { method: "DELETE" });
                say(r.ok ? "Deleted " + l.name + "." : "Could not delete.", !r.ok);
                refresh();
            });
            li.append(span, del);
            listEl.appendChild(li);
        });
        render();
        window.dispatchEvent(new Event("lines-changed"));
    }

    document.getElementById("line-save").addEventListener("click", async () => {
        if (points.length !== 2) return say("Draw a line first: click two points on the frame.", true);
        const camEl = document.getElementById("line-camera");
        const body = { name: nameEl.value.trim(), camera: (camEl && camEl.value.trim()) || "default", x1: points[0].x, y1: points[0].y, x2: points[1].x, y2: points[1].y };
        const res = await fetch("/lines/", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        if (res.ok) {
            say('Saved "' + body.name + '". It will be used for the next analysis.');
            points = []; hover = null;
            await refresh();
        } else {
            let msg = "Could not save.";
            try { const j = await res.json(); msg = typeof j.detail === "string" ? j.detail : "Check the name and coordinates."; } catch (e) {}
            say(msg, true);
        }
    });

    render();
    refresh();
})();
