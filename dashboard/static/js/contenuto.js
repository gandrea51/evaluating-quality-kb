(function () {
    const plotEl = document.getElementById("plot-data");
    if (!plotEl || typeof Plotly === "undefined") {
        console.error("Mancano i dati (plot-data) o Plotly");
        return;
    }
    const D = JSON.parse(plotEl.textContent);
    const jobEl = document.getElementById("job-data");
    const jobId = jobEl ? JSON.parse(jobEl.textContent).job_id : null;

    const P = D.points;
    const docs = [...new Set(P.map(p => p.doc))];
    const rels = {};                       // key coppia -> relazione (si riempie col polling)
    const relClass = r => r.replace(/\s+/g, "-");

    const colors = {
        "contraddizione": "#b91c1c", "complementare": "#16a34a", "equivalente": "#2563eb",
        "non correlato": "#6b7280", "": "#f59e0b", "invalid": "#a16207"
    };


    function lines(traces, pairs, name, color, dash, width) {
        if (!pairs.length) return;
        const x = [], y = [], z = [];

        pairs.forEach(([i, j]) => {
            x.push(P[i].x, P[j].x, null);
            y.push(P[i].y, P[j].y, null);
            z.push(P[i].z, P[j].z, null);
        });
        traces.push({ type: "scatter3d", mode: "lines", name, x, y, z, hoverinfo: "skip", line: { color, width, dash } });
    }

    function buildTraces() {
        const traces = [];
        docs.forEach(d => {
            const pts = P.filter(p => p.doc === d && !p.frag);
            if (!pts.length) return;
            traces.push({
                type: "scatter3d", mode: "markers", name: d,
                x: pts.map(p => p.x), y: pts.map(p => p.y), z: pts.map(p => p.z),
                text: pts.map(p => p.id + "<br>" + p.text), hoverinfo: "text",
                marker: {
                    size: pts.map(p => p.group >= 0 ? 6 : 3.5), opacity: .9,
                    line: { width: pts.map(p => p.group >= 0 ? 1.5 : 0), color: "#c00" }
                }
            });
        });

        const fr = P.filter(p => p.frag);
        if (fr.length) traces.push({
            type: "scatter3d", mode: "markers", name: "frammenti", 
            x: fr.map(p => p.x), y: fr.map(p => p.y), z: fr.map(p => p.z),
            text: fr.map(p => p.id + "<br>" + P.text), hoverinfo: "text",
            marker: {size: 3, symbol: "x", color: "#9ca3af"}
        });

        lines(traces, D.edges, "quasi-duplicati", "#dc2626", "solid", 4);

        Object.keys(colors).forEach(rel => {
            const ps = D.cands.filter(c => (rels[c.key] || "") === rel).map(c => [c.i, c.j]);
            lines(traces, ps, rel ? "candidata: " + rel : "candidata (non giudicata)", colors[rel], "dash", 3);
        });
        return traces;
    }

    const layout = {
        margin: { l: 0, r: 0, t: 10, b: 0 }, legend: { font: { size: 10 } },
        uirevision: "keep",
        scene: { xaxis: { title: "PC1" }, yaxis: { title: "PC2" }, zaxis: { title: "PC3" } }
    };

    Plotly.newPlot("plot", buildTraces(), layout,  {responsive: true});

    if (!jobId) return;

    const cards = new Map([...document.querySelectorAll(".candidate-card")].map(c => [c.dataset.key, c]));
    const applied = new Set();
    const kpi = document.getElementById("kpi-llm");

    function showJudgment(key, v) {
        const card = cards.get(key);
        if (!card) return;

        const slot = card.querySelector(".rel-slot");
        slot.textContent = "";

        const badge = document.createElement("span");
        badge.className = "badge tag-relation " + relClass(v.relation);
        badge.textContent = v.relation;
        slot.appendChild(badge);

        const box = card.querySelector(".explanation-box");
        if (v.explanation) {
            box.textContent = "";
            const s = document.createElement("strong");
            s.textContent = "Spiegazione Giudice: ";
            box.append(s, document.createTextNode(v.explanation));
            box.className = "explanation-box mt-3 expl-" + relClass(v.relation);
            box.hidded = false;
        }
    }

    function updateKpi(s) {
        const vals = Object.values(s.item || {});
        const contr = vals.filter(v => v.relation === "contraddizione").length;
        document.getElementById("llm-progress".textContent = `${s.done} / ${s.total}`);
        document.getElementById("llm-bar").style.width = (s.total ? 100*s.done/s.total : 0) + "%";

        const sub = document.getElementById("llm-sub");
        const base = `contraddizioni: ${contr}` + (s.done ? ` (${Math.round(100*contr/s.done)}%)`: "");
        if (s.status === "running") {
            sub.textContent = `in corso… ${base} · ${s.elapsed} s`;
        } else if (s.status === "done") {
            sub.textContent = `completato · ${base} · ${s.elapsed} s`;
            kpi.classList.add("done");
        } else {
            sub.textContent = "errore: " + (s.error || s.status);
            kpi.classList.add("error");
        }
    }

    async function poll() {
        let s;
        try {
            const r = await fetch(`/status/${jobId}`);
            s = await r.json();
        } catch (e) {
            console.error("polling fallito", e);
            return setTimeout(poll, 10000);
        }

        let changed = false;
        Object.entries(s.items || {}).forEach(([key, v]) => {
            if (applied.has(key)) return;
            applied.add(key);
            rels[key] = v.relation;
            showJudgment(key, v);
            changed = true;
        });

        if (changed) {
            Plotly.react("plot", buildTraces(), layout);
        }

        updateKpi(s);

        if (s.status === "running") {
            setTimeout(poll, 10000);
        }
    }
    poll();
}) ();