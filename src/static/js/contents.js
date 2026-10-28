(function () {
    // larghezze delle barre di avanzamento (valore in % nel data-width)
    document.querySelectorAll("[data-width]").forEach(el => {
        const val = Math.min(100, Math.max(0, Number(el.dataset.width) || 0));
        el.style.width = `${val}%`;
    });

    const plotEl = document.getElementById("plot-data");

    if (!plotEl || typeof Plotly === "undefined") {
        console.error("Mancano i dati (plot-data) o Plotly");
        return;
    }

    const safeParse = (id, fallback = {}) => {
        const elem = document.getElementById(id);
        if (!elem || !elem.textContent.trim()) return fallback;
        try {
            return JSON.parse(elem.textContent);
        } catch (e) {
            console.error(`Error parsing json per #${id}:`, e);
            return fallback;
        }
    };


    const D = safeParse("plot-data", null);
    if (!D || !D.points) {
        console.error("Dati plot non validi.");
        return;
    }

    const COV = D.cov || {names: [], centroids: [], sizes: []};
    const RED = safeParse("red-data", {nn: []});

    const jobData = safeParse("job-data", {});
    const jobId = jobData.job_id || null;

    const P = D.points;
    const docs = [...new Set(P.map(p => p.doc))];

    const PAL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                 "#bcbd22", "#17becf", "#393b79", "#e6550d", "#31a354", "#756bb1", "#636363", "#843c39"];
    const clusterColor = c => PAL[c % PAL.length];

    const rels = {};
    const relClass = relation => String(relation || "").replace(/\s+/g, "-");

    const colors = {
        "contraddizione": "#b91c1c",
        "complementare": "#16a34a",
        "equivalente": "#2563eb",
        "non correlato": "#6b7280",
        "": "#f59e0b",
        "invalid": "#a16207"
    };

    const pct = v => (v === null || v === undefined) ? "—" : (v * 100).toFixed(1) + "%";
    const setText = (id, text) => { 
        const el = document.getElementById(id); 
        if (el) el.textContent = text; 
    };

    let colorBy = "doc";
    const darkPlotTheme = {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#94a3b8", family: "inherit" },
        xaxis: { gridcolor: "rgba(255,255,255,0.06)", zerolinecolor: "rgba(255,255,255,0.1)" },
        yaxis: { gridcolor: "rgba(255,255,255,0.06)", zerolinecolor: "rgba(255,255,255,0.1)" }
    };

    // MAPPA 3D
    function addLines(traces, pairs, name, color, dash, width) {
        if (!pairs || !pairs.length) return;
        const x = [], y = [], z = [];

        pairs.forEach(([i, j]) => {
            if (!P[i] || !P[j]) return;
            x.push(P[i].x, P[j].x, null);
            y.push(P[i].y, P[j].y, null);
            z.push(P[i].z, P[j].z, null);
        });

        traces.push({
            type: "scatter3d", 
            mode: "lines", 
            name, 
            x, y, z,
            hoverinfo: "skip", 
            line: { color, width, dash }
        });
    }

    function pointTrace(points, name, color) {
        const marker = {
            size: points.map(p => p.group >= 0 ? 6 : 3.5),
            opacity: 0.9,
            line: { width: 1, color: points.map(p => p.group >= 0 ? "#ef4444" : "rgba(0,0,0,0)") }
        };
        if (color) marker.color = color;

        return {
            type: "scatter3d", 
            mode: "markers", 
            name,
            x: points.map(p => p.x), y: points.map(p => p.y), z: points.map(p => p.z),
            text: points.map(p => `${p.id}<br>${p.text}`),
            hoverinfo: "text", 
            marker
        };
    }

    function buildTraces() {
        const traces = [];

        if (colorBy === "doc") {
            docs.forEach(doc => {
                const points = P.filter(p => p.doc === doc && !p.frag);
                if (points.length) traces.push(pointTrace(points, doc));
            });
        } else {
            (COV.names || []).forEach((name, c) => {
                const points = P.filter(p => p.cluster === c && !p.frag);
                if (points.length) traces.push(pointTrace(points, name, clusterColor(c)));
            });

            if (COV.centroids && COV.centroids.length) {
                traces.push({
                    type: "scatter3d", 
                    mode: "markers+text", 
                    name: "Centroidi",
                    x: COV.centroids.map(c => c.x), y: COV.centroids.map(c => c.y), z: COV.centroids.map(c => c.z),
                    text: COV.centroids.map(c => `C${c.cluster + 1}`), 
                    textposition: "top center",
                    hoverinfo: "text",
                    marker: { 
                        size: 9, symbol: "diamond", color: COV.centroids.map(c => clusterColor(c.cluster)),
                        line: { width: 2, color: "#111" } 
                    }
                });
            }
        }

        const fragment = P.filter(p => p.frag);
        if (fragment.length) {
            traces.push({
                type: "scatter3d", 
                mode: "markers", 
                name: "frammenti",
                x: fragment.map(p => p.x), y: fragment.map(p => p.y), z: fragment.map(p => p.z),
                text: fragment.map(p => `${p.id}<br>${p.text}`), 
                hoverinfo: "text",
                marker: { size: 3, symbol: "x", color: "#9ca3af" }
            });
        }

        // quasi-duplicati
        if (D.edges) {
            addLines(traces, D.edges, "quasi-duplicati", "#dc2626", "solid", 4);
        }

        // coppie candidate, colorate per giudizio
        if (D.cands) {
            Object.entries(colors).forEach(([relation, color]) => {
                const pairs = D.cands.filter(c => (rels[c.key] || "") === relation).map(c => [c.i, c.j]);
                addLines(traces, pairs, relation ? `Candidata: ${relation}` : "Candidata non giudicata", color, "dash", 3);
            });
        }
        return traces;
    }

    const layout = {
        margin: { l: 0, r: 0, t: 10, b: 0 },
        legend: { font: { size: 10, color: "#94a3b8" } },
        uirevision: "keep",
        paper_bgcolor: "rgba(0,0,0,0)",
        scene: { 
            xaxis: { title: "PC1", ...darkPlotTheme.xaxis }, 
            yaxis: { title: "PC2", ...darkPlotTheme.yaxis }, 
            zaxis: { title: "PC3", gridcolor: "rgba(255,255,255,0.06)", zerolinecolor: "rgba(255,255,255,0.1)" } 
        }
    };

    Plotly.newPlot("plot", buildTraces(), layout, { responsive: true });

    document.querySelectorAll("[data-colorby]").forEach(btn => {
        btn.addEventListener("click", () => {
            colorBy = btn.dataset.colorby;
            document.querySelectorAll("[data-colorby]").forEach(b => b.classList.toggle("active", b === btn));
            Plotly.react("plot", buildTraces(), layout);
        });
    });

    // COPERTURA
    document.querySelectorAll(".cluster-dot").forEach(el => {
        el.style.background = clusterColor(Number(el.dataset.cluster));
    });

    function drawCoverage() {
        const base = { 
            margin: { l: 50, r: 10, t: 10, b: 60 },
            ...darkPlotTheme
        };

        // chunk per area + soglia di copertura
        const sizesEl = document.getElementById("plot-sizes");
        if (sizesEl && COV.names && COV.sizes) {
            const minChunks = Number(sizesEl.dataset.min) || 0;
            Plotly.newPlot("plot-sizes", [{
                type: "bar", 
                x: COV.names, y: COV.sizes,
                marker: { color: COV.names.map((_, c) => clusterColor(c)) },
                hovertemplate: "%{x}<br>%{y} chunk<extra></extra>"
            }], {
                ...base, 
                yaxis: { title: "Chunk", ...darkPlotTheme.yaxis }, 
                xaxis: { tickangle: -30, ...darkPlotTheme.xaxis },
                shapes: [{ 
                    type: "line", xref: "paper", x0: 0, x1: 1, y0: minChunks, y1: minChunks,
                    line: { color: "#dc2626", dash: "dash", width: 2 } 
                }],
                annotations: [{ 
                    xref: "paper", x: 1, y: minChunks, yanchor: "bottom", xanchor: "right",
                    text: "soglia di copertura", showarrow: false, font: { size: 10, color: "#ef4444" } 
                }]
            }, { responsive: true });
        }

        // silhouette per k
        if (COV.scan && COV.scan.length) {
            Plotly.newPlot("plot-kscan", [
                { 
                    type: "scatter", mode: "lines+markers", name: "Silhouette",
                    x: COV.scan.map(r => r.k), y: COV.scan.map(r => r.silhouette),
                    lin: {color: "#3b82f6"}
                },
                {  
                    type: "scatter", mode: "markers", name: "k selezionato", hoverinfo: "skip",
                    x: [COV.k], y: COV.scan.filter(r => r.k === COV.k).map(r => r.silhouette),
                    marker: { size: 14, color: "#ef4444", symbol: "circle-open", line: { width: 3 } } 
                }
            ], {
                ...base, 
                xaxis: { title: "k", dtick: 1, ...darkPlotTheme.xaxis }, 
                yaxis: { title: "Silhouette", ...darkPlotTheme.yaxis },
                showlegend: false
            }, { responsive: true });
        }

        // heatmap area x documento
        if (COV.heat && COV.heat.docs && COV.heat.docs.length) {
            Plotly.newPlot("plot-heat", [{
                type: "heatmap", 
                z: COV.heat.z, x: COV.heat.docs, y: COV.names, 
                colorscale: "Blues",
                hovertemplate: "%{y}<br>%{x}<br>%{z} chunk<extra></extra>"
            }], { 
                ...base, 
                margin: { l: 150, r: 10, t: 10, b: 120 },
                xaxis: { tickangle: -40, ...darkPlotTheme.xaxis } 
            }, { responsive: true });
        }
    }

    // RIDONDANZA: istogramma delle similarita' del vicino piu' prossimo

    function drawNearest() {
        const el = document.getElementById("plot-nn");
        if (!el || !RED.nn || !RED.nn.length) return;

        const thr = Number(el.dataset.thr) || 0.92;
        const cand = Number(el.dataset.cand) || 0.85;
        const line = (x, color, dash) => ({ type: "line", yref: "paper", x0: x, x1: x, y0: 0, y1: 1,
                                            line: { color, dash, width: 2 } });

        Plotly.newPlot("plot-nn", [{
            type: "histogram", x: RED.nn, nbinsx: 40, marker: { color: "#3b82f6" }
        }], {
            margin: { l: 50, r: 10, t: 10, b: 50 }, ...darkPlotTheme,
            xaxis: { title: "Coseno con il chunk più simile", ...darkPlotTheme.xaxis }, yaxis: { title: "Chunk", ...darkPlotTheme.yaxis },
            shapes: [line(thr, "#f8fafc", "solid"), line(cand, "#f59e0b", "dash")],
            annotations: [
                { x: thr, yref: "paper", y: 1, text: `ridondanza ${thr.toFixed(2)}`, showarrow: false, xanchor: "left", font: { size: 10, color: "#f8fafc" } },
                { x: cand, yref: "paper", y: 0.92, text: `candidati ${cand.toFixed(2)}`, showarrow: false, xanchor: "right", font: { size: 10, color: "#f59e0b" } }
            ]
        }, { responsive: true });
    }

    drawCoverage();
    drawNearest();

    // COERENZA / GIUDICE LLM

    if (!jobId) return;

    const cards = new Map([...document.querySelectorAll(".candidate-card")].map(card => [card.dataset.key, card]));
    const applied = new Set();
    const kpi = document.getElementById("kpi-llm");

    function showJudgment(key, judgement) {
        const card = cards.get(key);
        if (!card) return;

        const slot = card.querySelector(".rel-slot");
        if (slot) {
            slot.textContent = "";
            const badge = document.createElement("span");
            badge.className = `badge tag-relation ${relClass(judgement.relation)}`;
            badge.textContent = judgement.relation;
            slot.appendChild(badge);
        }

        const explanationBox = card.querySelector(".explanation-box");
        if (explanationBox && judgement.explanation) {
            explanationBox.textContent = "";
            const label = document.createElement("strong");
            label.textContent = "Spiegazione del giudice: ";
            explanationBox.append(label, document.createTextNode(judgement.explanation));
            explanationBox.className = `explanation-box mt-3 expl-${relClass(judgement.relation)}`;
            explanationBox.hidden = false;
        }
    }

    function updateKpi(status) {
        const progressEl = document.getElementById("llm-progress");
        const barEl = document.getElementById("llm-bar");
        const subEl = document.getElementById("llm-sub");
        if (!progressEl || !barEl || !subEl) return;

        const total = Number(status.total) || 0;
        const done = Number(status.done) || 0;
        const judgements = Object.values(status.items || {});
        const contradictions = judgements.filter(j => j.relation === "contraddizione").length;

        progressEl.textContent = `${done} / ${total}`;
        barEl.style.width = `${total ? Math.min(100, Math.round((done / total) * 100)) : 0}%`;

        const share = done > 0 ? ` (${Math.round((contradictions / done) * 100)}%)` : "";
        const base = `contraddizioni: ${contradictions}${share}`;

        if (status.status === "running") {
            subEl.textContent = `in corso… ${base} · ${status.elapsed || 0} s`;
        } else if (status.status === "done") {
            subEl.textContent = `completato · ${base} · ${status.elapsed || 0} s`;
            if (kpi) kpi.classList.add("done");
        } else {
            subEl.textContent = `errore: ${status.error || status.status}`;
            if (kpi) kpi.classList.add("error");
        }
    }

    // valori finali: Coerenza = 1 - C, C su coppie e su chunk, lista dei conflitti
    function renderCoerenza(c) {
        if (!c) return;

        setText("coe-value", pct(c.coerenza));
        setText("coe-value-2", pct(c.coerenza));
        setText("coe-cpairs", pct(c.C_pairs));
        setText("coe-cchunks", pct(c.C_chunks));
        setText("coe-n", `${c.n_contradictions || 0} / ${c.n_judged || 0}`);

        const note = document.getElementById("coe-note");
        if (note) {
            note.textContent = "";
            note.append(
                document.createTextNode(`${c.n_contradictions || 0} / ${c.n_judged || 0}`),
                document.createElement("br"),
                document.createTextNode("coppie contraddittorie")
            );
        }

        const list = document.getElementById("conflicts-list");
        const empty = document.getElementById("conflicts-empty");
        if (!list) return;

        const conflicts = c.conflicts || [];
        if (empty) {
            empty.textContent = conflicts.length ? "" : "Nessuna contraddizione rilevata.";
            empty.hidden = conflicts.length > 0;
        }

        conflicts.forEach(cf => {
            const art = document.createElement("article");
            art.className = "candidate-card";

            const head = document.createElement("div");
            head.className = "candidate-header d-flex flex-wrap gap-3 mb-2";

            const sim = document.createElement("strong");
            sim.textContent = `Similarità ${(cf.sim || 0).toFixed(3)}`;

            const ids = document.createElement("span");
            ids.className = "text-muted";
            ids.textContent = `${cf.a} ↔ ${cf.b}`;

            const tag = document.createElement("span");
            tag.className = "badge tag-relation contraddizione";
            tag.textContent = cf.cross_doc ? "contraddizione · documenti diversi" : "contraddizione · stesso documento";

            head.append(sim, ids, tag);

            const row = document.createElement("div");
            row.className = "row g-3 candidate-texts";
            [["A", cf.text_a], ["B", cf.text_b]].forEach(([letter, text]) => {
                const col = document.createElement("div");
                col.className = "col-md-6";
                const box = document.createElement("div");
                box.className = "chunk-box candidate-chunk";
                const title = document.createElement("div");
                title.className = "chunk-heading";
                title.textContent = `Testo ${letter}`;
                const p = document.createElement("p");
                p.className = "chunk-text";
                p.textContent = text || "";
                box.append(title, p);
                col.appendChild(box);
                row.appendChild(col);
            });

            const expl = document.createElement("div");
            expl.className = "explanation-box mt-3 expl-contraddizione";
            const label = document.createElement("strong");
            label.textContent = "Spiegazione del giudice: ";
            expl.append(label, document.createTextNode(cf.explanation || ""));

            art.append(head, row, expl);
            list.appendChild(art);
        });
    }

    async function poll() {
        let status;
        try {
            const response = await fetch(`/status/${jobId}`);
            if (response.status === 404) {
                // Se 404 riprova dopo 3 secondi invece di interrompersi (il thread potrebbe essere all'avvio)
                return setTimeout(poll, 3000);
            }
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            status = await response.json();
        } catch (error) {
            console.error("Polling fallito, nuovo tentativo tra 5s:", error);
            return setTimeout(poll, 5000);
        }

        let changed = false;
        Object.entries(status.items || {}).forEach(([key, judgement]) => {
            if (applied.has(key)) return;
            applied.add(key);
            rels[key] = judgement.relation;
            showJudgment(key, judgement);
            changed = true;
        });

        if (changed) Plotly.react("plot", buildTraces(), layout);

        updateKpi(status);

        if (status.status === "done") {
            renderCoerenza(status.coerenza);
        } else if (status.status === "running") {
            setTimeout(poll, 3000);
        }
    }
    poll();
})();